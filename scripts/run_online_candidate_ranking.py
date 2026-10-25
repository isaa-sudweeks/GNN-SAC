"""Frozen source-only candidate ranking using an explicitly supplied campaign runtime."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np


CONDITIONS = ('shared', 'separate')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def selected_states(length: int, collapsed: bool) -> list[dict]:
    """Select only realized-trajectory indices, never counterfactual outcomes."""
    if length < 1:
        raise ValueError('An episode must contain at least one action')
    labels = defaultdict(list)
    labels[0].append('initial')
    if length > 20:
        labels[20].append('t20')
    labels[max(0, length - (5 if collapsed else 10))].append(
        'precollapse' if collapsed else 'late_survivor')
    return [dict(timestep=t, strata=s) for t, s in sorted(labels.items())]


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def ranking_metrics(scores: np.ndarray, predicted_failure: np.ndarray,
                    truth: np.ndarray, failure: np.ndarray,
                    displacement: np.ndarray, selected: int | None = None) -> dict:
    """Finite-set decision metrics; a constant ranking is explicitly uninformative."""
    if not len(scores) or not all(len(a) == len(scores) for a in
                                 (predicted_failure, truth, failure, displacement)):
        raise ValueError('Candidate arrays must have equal nonzero lengths')
    if not all(np.isfinite(a).all() for a in (scores, truth, displacement)):
        raise ValueError('Nonfinite candidates require an explicit failure record')
    chosen = int(np.argmax(scores)) if selected is None else int(selected)
    if chosen < 0 or chosen >= len(scores):
        raise ValueError('Selected candidate index is out of range')
    a, b = average_ranks(scores), average_ranks(truth)
    informative = bool(np.ptp(a) and np.ptp(b))
    safe = ~failure.astype(bool)
    return dict(candidates=len(scores), selected=chosen,
        spearman=float(np.corrcoef(a, b)[0, 1]) if informative else None,
        informative=informative, regret=float(truth.max() - truth[chosen]),
        selected_true_score=float(truth[chosen]), selected_displacement=float(displacement[chosen]),
        selected_collapse=bool(failure[chosen]), safe_candidates=int(safe.sum()),
        positive_progress_safe_candidates=int((safe & (displacement > 0)).sum()),
        avoidable_selected_collapse=bool(failure[chosen] and safe.any()),
        safe_opportunity=bool(safe.any()), mixed_candidates=bool(safe.any() and failure.any()),
        true_failures=int(failure.sum()), predicted_failures=int(predicted_failure.sum()),
        false_negatives=int((failure & ~predicted_failure).sum()),
        false_positives=int((~failure & predicted_failure).sum()))


def model_digest(model) -> str:
    digest = hashlib.sha256()
    for key, tensor in model.state_dict().items():
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def predict_failures(learner, state: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    import torch
    with torch.no_grad():
        imagined = torch.tensor(state, dtype=torch.float32)[None].repeat(len(candidates), 1, 1)
        actions = torch.tensor(candidates, dtype=torch.float32)[..., None]
        failed = torch.zeros(len(candidates), dtype=torch.bool)
        for t in range(learner.cem.horizon):
            imagined, progress = learner.model.dynamics.step(imagined, actions[:, t], learner.graph)
            if not (torch.isfinite(imagined).all() and torch.isfinite(progress).all()):
                raise FloatingPointError('Nonfinite imagined state/progress; diagnostic failure retained')
            failed |= learner.collapsed(imagined)
            imagined = torch.nan_to_num(imagined, nan=0., posinf=0., neginf=0.)
        return failed.numpy()


def propose(learner, state, seed, previous, cem_plan) -> tuple[np.ndarray, int]:
    batches = []
    def score(sequences):
        batches.append(sequences.copy())
        return learner.score(state, sequences)
    result = cem_plan(score, len(learner.metadata['action_mask']), learner.cem,
        np.random.default_rng(seed), action_mask=np.asarray(learner.metadata['action_mask']),
        previous_plan=previous)
    candidates = np.concatenate(batches)
    matches = np.flatnonzero(np.all(candidates == result.actions, axis=(1, 2)))
    if not len(matches):
        raise AssertionError('Native CEM selection was not actually evaluated')
    return candidates, int(matches[0])


def branch_rollout(branch, snapshot, actions, metadata, cfg, actual_step) -> dict:
    branch.restore(snapshot)
    mapping = np.asarray(metadata['node_to_physical'])
    rewards, rows, elapsed = [], [], []
    for action in actions:
        row, detail, done = actual_step(branch.planning_env, metadata, mapping, action[:, None], cfg)
        if not all(np.isfinite(row[k]).all() for k in ('state', 'next_state', 'progress')):
            raise FloatingPointError('Nonfinite actual simulator transition; evidence retained')
        if not np.isfinite(row['reward']):
            raise FloatingPointError('Nonfinite actual simulator reward')
        rows.append(row)
        elapsed.append(detail['dt'])
        rewards.append(float(row['reward']))
        if row['truncated']:
            raise AssertionError('Unexpected wrapper truncation within diagnostic horizon')
        if done:
            break
    return dict(score=float(sum(cfg['gamma'] ** t * r for t, r in enumerate(rewards))),
        collapse=bool(rows[-1]['terminated']),
        displacement=float(sum(r['progress'][0] for r in rows) / metadata['initial_diagonal']),
        state=np.asarray([r['state'] for r in rows]), next_state=np.asarray([r['next_state'] for r in rows]),
        reward=np.asarray(rewards), progress=np.asarray([r['progress'] for r in rows]),
        elapsed=np.asarray(elapsed))


def verify_restore(branch, snapshot, actions, expected, metadata, cfg, actual_step) -> dict:
    first = branch_rollout(branch, snapshot, actions[:1], metadata, cfg, actual_step)
    for key in ('state', 'next_state', 'progress', 'reward'):
        if not np.array_equal(first[key][0], expected[key]):
            raise AssertionError(f'Full snapshot continuation mismatch: {key}')
    forward = branch_rollout(branch, snapshot, actions, metadata, cfg, actual_step)
    branch_rollout(branch, snapshot, np.zeros_like(actions), metadata, cfg, actual_step)
    repeat = branch_rollout(branch, snapshot, actions, metadata, cfg, actual_step)
    if not all(np.array_equal(forward[k], repeat[k]) for k in forward):
        raise AssertionError('Candidate-order independence failed')
    return dict(exact_continuation=True, exact_order_independence=True)


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row['topology'], row['condition'])].append(row)
    output = []
    for (topology, condition), cells in sorted(groups.items()):
        native = [c['native'] for c in cells]
        correlations = [c['pooled']['spearman'] for c in cells if c['pooled']['informative']]
        true_failures = sum(c['pooled']['true_failures'] for c in cells)
        output.append(dict(topology=topology, condition=condition, states=len(cells),
            native_selected_collapses=sum(c['selected_collapse'] for c in native),
            native_avoidable_collapses=sum(c['avoidable_selected_collapse'] for c in native),
            safe_opportunities=sum(c['safe_opportunity'] for c in native),
            native_mean_regret=float(np.mean([c['regret'] for c in native])),
            pooled_selected_collapses=sum(c['pooled']['selected_collapse'] for c in cells),
            median_spearman=float(np.median(correlations)) if correlations else None,
            false_negatives=sum(c['pooled']['false_negatives'] for c in cells),
            true_failures=true_failures,
            false_positives=sum(c['pooled']['false_positives'] for c in cells)))
    return output


def report(output: Path, rows: list[dict], episodes: list[dict], elapsed: float, complete: bool) -> None:
    summary = summarize(rows)
    by_seed = [dict(seed=seed, **cell) for seed in sorted({r['seed'] for r in rows})
               for cell in summarize([r for r in rows if r['seed'] == seed])]
    strata = defaultdict(list)
    for row in rows:
        strata[(row['seed'], row['generator'], '+'.join(row['strata']))].append(row)
    stratified = [dict(seed=seed, generator=generator, stratum=stratum, **cell)
                  for (seed, generator, stratum), cells in sorted(strata.items())
                  for cell in summarize(cells)]
    write_json(output / 'results.json', dict(complete=complete, elapsed_seconds=elapsed,
        episodes=episodes, rows=rows, summary=summary, by_seed=by_seed, stratified=stratified))
    lines = ['# Frozen online candidate-ranking results', '',
        f'Status: {"complete" if complete else "partial"}; elapsed {elapsed:.1f} seconds.', '',
        'Source-development diagnostic; no fitting, unseen topology or model-free comparison.', '',
        '| Topology | Model | States | Native collapse | Avoidable collapse | Safe opportunities | Mean regret | Median Spearman | Missed candidate collapses |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        correlation = row['median_spearman']
        lines.append(f"| {row['topology']} | {row['condition']} | {row['states']} | "
            f"{row['native_selected_collapses']} | {row['native_avoidable_collapses']} | "
            f"{row['safe_opportunities']} | {row['native_mean_regret']:.6f} | "
            f"{correlation if correlation is not None else 'uninformative'} | "
            f"{row['false_negatives']}/{row['true_failures']} |")
    lines += ['', '## Per-training-seed results', '',
        '| Topology | Seed | Model | States | CEM-selected collapses | Avoidable collapses | Mean regret |',
        '|---|---:|---|---:|---:|---:|---:|']
    for cell in by_seed:
        lines.append(f"| {cell['topology']} | {cell['seed']} | {cell['condition']} | {cell['states']} | "
            f"{cell['native_selected_collapses']} | {cell['native_avoidable_collapses']} | {cell['native_mean_regret']:.6f} |")
    lines += ['', 'Native means CEM-selected within each model proposal set, recreated with common RNG and generator warm start; it is not the original episode action.',
        'Original episode plans are preserved separately. Regret uses the shared candidate pool.',
        'Avoidable means that a sequence survives the 10-step branch in this finite pool.',
        'States overlap within trajectories; candidates are not independent observations.',
        'No surviving candidate does not establish unavoidable collapse. Branch survival is only two seconds.',
        'Branches execute sequences open-loop; deployed MPC replans after every first action. This is not a causal test of closed-loop failure.',
        'All-candidate metrics include duplicates; unique/provenance sensitivity metrics are saved per state.',
        'Per-state, per-seed, generator and stratum data are retained in results.json.',
        'No claim about dynamics versus model-free transfer follows from this source-only experiment.', '']
    (output / 'report.md').write_text('\n'.join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config/online_candidate_ranking.yaml')
    parser.add_argument('--benchmark', action='store_true', help='One episode and one selected state only')
    args = parser.parse_args()
    runtime, campaign, output = args.runtime_root.resolve(), args.campaign_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(runtime), str(runtime / 'sac')]
    import torch
    from omegaconf import OmegaConf
    from dynamics.online import OnlineLearner
    from dynamics.planner import cem_plan
    from dynamics.simulator import SimulatorBranch
    from scripts.collect_dynamics_training_data import make_training_env, graph_metadata, observable_state
    from scripts.run_online_dynamics_pilot import actual_step, random_sequence

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol_path = campaign / 'protocol.json'
    source_protocol = json.loads(protocol_path.read_text())
    cfg = source_protocol['settings']
    diagnostic_cfg = dict(OmegaConf.to_container(OmegaConf.load(args.config)))
    expected = ['tetrahedron', 'henneberg_n6_2tube_1', 'henneberg_n8_3tube_64']
    if cfg['topologies'] != expected or cfg['gamma'] != .995 or cfg['steps'] != 20000:
        raise ValueError('Requires the fixed original gamma=.995 source campaign')
    inputs = {str(protocol_path): sha256(protocol_path)}
    for relative, digest in source_protocol['sources'].items():
        path = campaign / 'source-snapshot' / relative
        if sha256(path) != digest:
            raise ValueError(f'Original campaign source hash changed: {relative}')
        inputs[str(path)] = digest
    runtime_sources = {str(path): sha256(path) for folder in ('sac/dynamics', 'env')
                       for path in (runtime / folder).rglob('*.py')}
    for path in (runtime / 'scripts').glob('*dynamics*.py'):
        runtime_sources[str(path)] = sha256(path)
    # Existing publication changed trailing whitespace only in training.py.
    import ast
    for relative in source_protocol['sources']:
        if ast.dump(ast.parse((runtime / relative).read_text())) != ast.dump(
                ast.parse((campaign / 'source-snapshot' / relative).read_text())):
            raise ValueError(f'Executed runtime differs semantically from campaign: {relative}')
    versions = {name: importlib.metadata.version(name) for name in source_protocol['dependencies']}
    if versions != source_protocol['dependencies']:
        raise ValueError('Runtime dependency versions differ from source campaign')
    # Verify reset exclusion against all recorded evaluation seeds and fitting reset ranges.
    prior_seeds = set()
    for path in campaign.glob('*-seed*/report.json'):
        inputs[str(path)] = sha256(path)
        data = json.loads(path.read_text())
        def collect(value):
            if isinstance(value, dict):
                if isinstance(value.get('seed'), int):
                    prior_seeds.add(value['seed'])
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)
        collect(data)
    planned = [diagnostic_cfg['reset_seed_base'] + i * 10000 + s * 100 + e
               for i in range(3) for s in cfg['seeds']
               for e in range(diagnostic_cfg['episodes_per_seed_topology_generator'])]
    if prior_seeds.intersection(planned) or min(planned) <= 3000000:
        raise ValueError('Diagnostic resets overlap previous data roles')
    plan_path = Path(__file__).resolve().parents[1] / 'docs/plans/online_candidate_ranking.md'
    source_hashes = {str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
                     str(plan_path): sha256(plan_path), str(args.config.resolve()): sha256(args.config), **runtime_sources}
    write_json(output / 'protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        settings=cfg, inputs=inputs, runtime_sources=source_hashes, dependencies=versions,
        reset_seeds=planned, reset_exclusion_verified=True, benchmark=args.benchmark,
        runtime_root=str(runtime), campaign_root=str(campaign), no_updates=True,
        diagnostic_settings=diagnostic_cfg, candidate_count=73, episode_steps=80, planning_horizon=10,
        cap_seconds=diagnostic_cfg['cap_seconds'], claim='Source-only prerequisite, not a model-free transfer comparison'))
    started = time.perf_counter()
    rows, episodes, checks = [], [], []
    try:
        for seed in cfg['seeds']:
            records = {}
            for condition in CONDITIONS:
                path = campaign / f'{condition}-seed{seed}' / 'checkpoint-020000.pt'
                inputs[str(path)] = sha256(path)
                manifest = json.loads((path.parent / 'resume.json').read_text())
                if inputs[str(path)] != manifest['sha256'] or manifest['step'] != 20000:
                    raise ValueError('Final checkpoint differs from saved completion hash')
                saved = torch.load(path, map_location='cpu', weights_only=False)
                if saved['step'] != 20000 or saved['settings'] != cfg:
                    raise ValueError('Checkpoint budget/settings mismatch')
                for record in saved['topologies'].values():
                    rt = record['runtime']
                    fitting_seeds = range(rt['reset_base'], rt['reset_base'] + rt['episode'] + 1)
                    if any(value in fitting_seeds for value in planned):
                        raise ValueError('Diagnostic reset overlaps actual fitting episode seed')
                records[condition] = {topology: dict(metadata=record['metadata'], model=record['model'])
                    for topology, record in saved['topologies'].items()}
                del saved
            for topology_index, topology in enumerate(expected):
                metadata = records['shared'][topology]['metadata']
                if metadata != records['separate'][topology]['metadata']:
                    raise ValueError('Shared/separate graph metadata differ')
                mapping = np.asarray(metadata['node_to_physical'])
                mask = np.asarray(metadata['action_mask'])
                initial = np.zeros((len(mask), 6), dtype=np.float32)
                learners = {}
                for condition in CONDITIONS:
                    learner = OnlineLearner(metadata, initial, cfg, seed, False)
                    learner.model.load_state_dict(records[condition][topology]['model'])
                    learner.model.eval()
                    learner.model.requires_grad_(False)
                    learners[condition] = learner
                digests = {c: model_digest(l.model) for c, l in learners.items()}
                for episode_index in range(diagnostic_cfg['episodes_per_seed_topology_generator']):
                    reset_seed = diagnostic_cfg['reset_seed_base'] + topology_index * 10000 + seed * 100 + episode_index
                    paired_start = None
                    for generator in CONDITIONS:
                        env = make_training_env(topology, OmegaConf.create(cfg))
                        auxiliary = make_training_env(topology, OmegaConf.create(cfg))
                        try:
                            auxiliary.reset()
                            env.reset()
                            obs, _ = env.unwrapped.reset(seed=reset_seed)
                            actual_metadata = graph_metadata(env.unwrapped, obs)
                            excluded = {'control_interval_seconds', 'initial_edge_length'}
                            if {k: v for k, v in actual_metadata.items() if k not in excluded} != {
                                    k: v for k, v in metadata.items() if k not in excluded}:
                                raise ValueError('Runtime environment graph/physics metadata changed')
                            branch = SimulatorBranch(env, auxiliary)
                            state, xy = observable_state(env.unwrapped, mapping)
                            start_hash = hashlib.sha256(state.tobytes() + xy.tobytes()).hexdigest()
                            if paired_start is None:
                                paired_start = start_hash
                            elif start_hash != paired_start:
                                raise AssertionError('Generator reset states are not paired')
                            rng = np.random.default_rng(diagnostic_cfg['trajectory_planner_seed_base'] + topology_index * 10000 + seed * 100 + episode_index)
                            snapshots, states, previous_plans, actions, trace, episode_plans = [], [], [], [], [], []
                            previous = None
                            for t in range(diagnostic_cfg['episode_steps']):
                                snapshots.append(branch.capture())
                                state, _ = observable_state(env.unwrapped, mapping)
                                states.append(state.copy())
                                previous_plans.append(None if previous is None else previous.copy())
                                plan = learners[generator].plan(state, rng, previous)
                                previous = plan.actions.copy()
                                episode_plans.append(plan.actions.copy())
                                action = plan.actions[0].reshape(-1, 1)
                                row, _, done = actual_step(env, metadata, mapping, action, cfg)
                                actions.append(action[:, 0].copy())
                                trace.append(row)
                                if done:
                                    break
                            identity = f'{topology}-seed{seed}-reset{episode_index}-{generator}'
                            folder = output / identity
                            folder.mkdir()
                            np.savez_compressed(folder / 'episode.npz', plans=np.asarray(episode_plans),
                                **{key: np.asarray([r[key] for r in trace]) for key in trace[0]})
                            selection = selected_states(len(trace), bool(trace[-1]['terminated']))
                            episodes.append(dict(topology=topology, seed=seed, generator=generator,
                                reset_seed=reset_seed, initial_state_sha256=start_hash, transitions=len(trace),
                                checkpoint_initial_edge_length=metadata['initial_edge_length'],
                                reset_initial_edge_length=actual_metadata['initial_edge_length'],
                                collapsed=bool(trace[-1]['terminated']), states=selection,
                                trace=str((folder / 'episode.npz').relative_to(output))))
                            for selection_index, selected in enumerate(selection):
                                if time.perf_counter() - started > diagnostic_cfg['cap_seconds']:
                                    raise TimeoutError('Fixed elapsed cap reached; partial results retained')
                                t = selected['timestep']
                                state, snapshot = states[t], snapshots[t]
                                live_before, live_xy_before = observable_state(env.unwrapped, mapping)
                                live_time_before = float(env.unwrapped.mj_model.data.time)
                                live_qpos = env.unwrapped.mj_model.data.qpos.copy()
                                live_qvel = env.unwrapped.mj_model.data.qvel.copy()
                                state_started = time.perf_counter()
                                proposal_seed = diagnostic_cfg['candidate_seed_base'] + topology_index * 100000 + seed * 10000 + episode_index * 1000 + t
                                chunks, native = [], {}
                                for c, learner in learners.items():
                                    candidates, chosen = propose(learner, state, proposal_seed,
                                        previous_plans[t], cem_plan)
                                    native[c] = sum(len(v) for v in chunks) + chosen
                                    chunks.append(candidates)
                                randoms = np.asarray([random_sequence(np.random.default_rng(proposal_seed + 100 + i), 10, mask)[..., 0]
                                                      for i in range(diagnostic_cfg['random_candidates'])])
                                candidates = np.concatenate([*chunks, np.zeros((1, 10, len(mask))), randoms])
                                provenance = np.asarray(['shared_cem'] * 32 + ['separate_cem'] * 32 + ['zero'] + ['random'] * 8)
                                if candidates.shape != (73, 10, len(mask)) or not np.isfinite(candidates).all():
                                    raise AssertionError('Common candidate accounting failed')
                                if np.max(np.abs(candidates)) > 1 or np.any(candidates[..., mask == 0]):
                                    raise AssertionError('Invalid action bounds or masks')
                                score = {c: learner.score(state, candidates) for c, learner in learners.items()}
                                predicted = {c: predict_failures(learner, state, candidates) for c, learner in learners.items()}
                                # Match the campaign's actual next transition, then test branch order.
                                continuation = np.asarray(actions[t:t + 10])
                                check = verify_restore(branch, snapshot, continuation, trace[t], metadata, cfg, actual_step)
                                checks.append(dict(episode=identity, timestep=t, **check))
                                truth_records = [branch_rollout(branch, snapshot, sequence, metadata, cfg, actual_step)
                                                 for sequence in candidates]
                                truth = np.asarray([r['score'] for r in truth_records])
                                failures = np.asarray([r['collapse'] for r in truth_records])
                                moves = np.asarray([r['displacement'] for r in truth_records])
                                live_after, live_xy_after = observable_state(env.unwrapped, mapping)
                                if not (np.array_equal(live_before, live_after) and
                                        np.array_equal(live_xy_before, live_xy_after) and
                                        np.array_equal(live_qpos, env.unwrapped.mj_model.data.qpos) and
                                        np.array_equal(live_qvel, env.unwrapped.mj_model.data.qvel) and
                                        live_time_before == float(env.unwrapped.mj_model.data.time)):
                                    raise AssertionError('Counterfactual branches modified the live environment')
                                stem = folder / f'state-{t:03d}'
                                with stem.with_suffix('.pkl').open('wb') as stream:
                                    pickle.dump(dict(snapshot=snapshot, metadata=metadata, truth=truth_records), stream)
                                np.savez_compressed(stem.with_suffix('.npz'), candidates=candidates, state=state,
                                    provenance=provenance, generator_previous_plan=previous_plans[t],
                                    original_generator_plan=episode_plans[t],
                                    simulator_scores=truth, simulator_collapse=failures, simulator_displacement=moves,
                                    **{f'{c}_scores': score[c] for c in CONDITIONS},
                                    **{f'{c}_collapse': predicted[c] for c in CONDITIONS})
                                for c in CONDITIONS:
                                    _, unique_indices, inverse = np.unique(candidates, axis=0, return_index=True, return_inverse=True)
                                    unique_metrics = ranking_metrics(score[c][unique_indices], predicted[c][unique_indices],
                                        truth[unique_indices], failures[unique_indices], moves[unique_indices],
                                        int(inverse[native[c]]))
                                    by_provenance = {}
                                    for label in np.unique(provenance):
                                        ix = np.flatnonzero(provenance == label)
                                        by_provenance[str(label)] = ranking_metrics(score[c][ix], predicted[c][ix],
                                            truth[ix], failures[ix], moves[ix])
                                    rows.append(dict(topology=topology, seed=seed, generator=generator,
                                        reset_seed=reset_seed, timestep=t, strata=selected['strata'], condition=c,
                                        native=ranking_metrics(score[c], predicted[c], truth, failures, moves, native[c]),
                                        pooled=ranking_metrics(score[c], predicted[c], truth, failures, moves),
                                        unique=unique_metrics, by_provenance=by_provenance,
                                        artifact=str(stem.with_suffix('.npz').relative_to(output)),
                                        wall_seconds=time.perf_counter() - state_started))
                                report(output, rows, episodes, time.perf_counter() - started, False)
                                print(json.dumps(dict(episode=identity, timestep=t, states=len(rows) // 2,
                                    elapsed_seconds=time.perf_counter() - started)), flush=True)
                                if args.benchmark:
                                    break
                            if any(model_digest(l.model) != digests[c] for c, l in learners.items()):
                                raise AssertionError('Frozen model changed during evaluation')
                        finally:
                            env.close()
                            auxiliary.close()
                        if args.benchmark:
                            break
                    if args.benchmark:
                        break
                if args.benchmark:
                    break
            if args.benchmark:
                break
        all_hashes = {**inputs, **source_hashes}
        preserved = all(sha256(Path(path)) == digest for path, digest in all_hashes.items())
        if not preserved:
            raise AssertionError('Source/checkpoint inputs changed')
        write_json(output / 'verification.json', dict(passed=True, input_hashes=all_hashes,
            hashes_preserved=True, frozen_models=True, reset_pairing=True, restoration=checks,
            no_fitting=True, no_unseen_topologies=True, episodes=len(episodes), states=len(rows) // 2))
        report(output, rows, episodes, time.perf_counter() - started, True)
    except Exception:
        write_json(output / 'failure.json', dict(traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started))
        report(output, rows, episodes, time.perf_counter() - started, False)
        raise


if __name__ == '__main__':
    main()
