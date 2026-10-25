"""Fixed-replay sampling ablation followed by frozen source closed-loop evaluation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np

from failure_balanced_helpers import ReplaySampler, restored_history
from run_online_candidate_ranking import (branch_rollout, model_digest, predict_failures,
    propose, ranking_metrics, sha256, verify_restore, write_json)


MODULES = ('model', 'actor', 'target', 'target_actor', 'optimizer', 'actor_optimizer')


def bootstrap(runtime: Path) -> dict[str, Any]:
    sys.path[:0] = [str(runtime), str(runtime / 'sac')]
    import torch
    from omegaconf import OmegaConf
    from dynamics.online import OnlineLearner, OnlineReplay
    from dynamics.multitopology import share_dynamics, update_round
    from dynamics.planner import cem_plan
    from dynamics.simulator import SimulatorBranch
    from scripts.collect_dynamics_training_data import make_training_env, observable_state, graph_metadata
    from scripts.run_online_dynamics_pilot import actual_step, random_sequence
    from scripts.run_dynamics_multi_diagnostic import cohorts, prediction_metrics
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return locals()


def load_group(api: dict, campaign: Path, seed: int, arm: str, options: dict) -> tuple:
    torch = api['torch']
    path = campaign / f'shared-seed{seed}' / 'checkpoint-020000.pt'
    manifest = json.loads((path.parent / 'resume.json').read_text())
    if sha256(path) != manifest['sha256'] or manifest['step'] != 20000:
        raise ValueError('Original final checkpoint hash/budget mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    cfg = saved['settings']
    if cfg['gamma'] != .995 or saved['step'] != 20000 or cfg['training_horizon'] != 5:
        raise ValueError('Incorrect source campaign')
    learners, replays, samplers = {}, {}, {}
    for topology, record in saved['topologies'].items():
        arrays = record['replay']
        learner = api['OnlineLearner'](record['metadata'], arrays['state'][0], cfg, seed, False)
        for name in MODULES:
            getattr(learner, name).load_state_dict(record[name])
        learner.updates = record['updates']
        if learner.updates != 20000:
            raise ValueError('Incorrect topology view update count')
        replay = api['OnlineReplay'](len(arrays['state']), len(arrays['state'][0]))
        for key, values in arrays.items():
            getattr(replay, key)[:len(values)].copy_(torch.as_tensor(values))
        replay.size = len(arrays['state'])
        sampler = ReplaySampler(arrays, cfg['training_horizon'], arm, record['learner_rng'],
            options['batch_size'], options['failure_distance'], options['failure_fraction'])
        if not np.array_equal(sampler.valid, replay.starts(cfg['training_horizon']).numpy()):
            raise AssertionError('Sampler valid windows differ from native replay')
        learner.rng = sampler
        learners[topology], replays[topology], samplers[topology] = learner, replay, sampler
    if len({model_digest(l.model) for l in learners.values()}) != 1:
        raise AssertionError('Source shared topology views have different parameters')
    api['share_dynamics'](learners)
    del saved
    first = next(iter(learners.values()))
    steps = {int(value['step']) for value in first.optimizer.state.values()}
    if steps != {60000}:
        raise AssertionError(f'Incorrect original shared Adam accounting: {steps}')
    return cfg, learners, replays, samplers


def save_fit(api: dict, output: Path, round_number: int, learners: dict, samplers: dict,
             identity: dict, elapsed: float) -> None:
    first = next(iter(learners.values()))
    saved = dict(round=round_number, identity=identity, elapsed=elapsed,
        torch_rng=api['torch'].get_rng_state(), records={})
    for topology, learner in learners.items():
        if samplers[topology].phase != 0:
            raise AssertionError('Can checkpoint only a complete topology round')
        saved['records'][topology] = {name: getattr(learner, name).state_dict() for name in MODULES}
        saved['records'][topology].update(updates=learner.updates, sampler=samplers[topology].state_dict())
    temporary = output / 'resume.tmp'
    api['torch'].save(saved, temporary)
    temporary.replace(output / 'resume.pt')
    write_json(output / 'resume.json', dict(round=round_number, sha256=sha256(output / 'resume.pt')))
    if round_number == identity['rounds']:
        # The final checkpoint also contains complete restart state.
        import shutil
        shutil.copy2(output / 'resume.pt', output / 'final-model.pt')


def restore_fit(api: dict, output: Path, learners: dict, samplers: dict, identity: dict) -> tuple[int, float]:
    manifest = json.loads((output / 'resume.json').read_text())
    if sha256(output / 'resume.pt') != manifest['sha256']:
        raise ValueError('Restart checkpoint hash mismatch')
    saved = api['torch'].load(output / 'resume.pt', map_location='cpu', weights_only=False)
    if saved['identity'] != identity or saved['round'] != manifest['round']:
        raise ValueError('Restart source/config/identity mismatch')
    for topology, learner in learners.items():
        record = saved['records'][topology]
        for name in MODULES:
            getattr(learner, name).load_state_dict(record[name])
        learner.updates = record['updates']
        samplers[topology].load_state_dict(record['sampler'])
    api['torch'].set_rng_state(saved['torch_rng'])
    return saved['round'], saved['elapsed']


def collect(api: dict, learner: Any, cfg: dict, resets: list[int], output: Path,
            candidate_validation: bool) -> list[dict]:
    """Use literal paired reset IDs; save no data to fitting replay."""
    output.mkdir(parents=True, exist_ok=False)
    metadata = learner.metadata
    topology = metadata['topology']
    mapping = np.asarray(metadata['node_to_physical'])
    mask = np.asarray(metadata['action_mask'])
    env = api['make_training_env'](topology, api['OmegaConf'].create(cfg))
    auxiliary = api['make_training_env'](topology, api['OmegaConf'].create(cfg)) if candidate_validation else None
    records = []
    before = model_digest(learner.model)
    try:
        if auxiliary:
            auxiliary.reset()
        for episode, reset_seed in enumerate(resets):
            env.reset(); obs, _ = env.unwrapped.reset(seed=reset_seed)
            actual_metadata = api['graph_metadata'](env.unwrapped, obs)
            excluded = {'initial_edge_length', 'control_interval_seconds'}
            if {k: v for k, v in actual_metadata.items() if k not in excluded} != {
                    k: v for k, v in metadata.items() if k not in excluded}:
                raise AssertionError('Current simulator graph/routing/physics differs from checkpoint')
            interval = float(env.unwrapped.nsubsteps * env.unwrapped.mj_model.model.opt.timestep)
            if interval != metadata['control_interval_seconds'] or not np.isclose(interval, .2):
                raise AssertionError('Control interval changed')
            rng = np.random.default_rng(reset_seed + 1000000)
            branch = api['SimulatorBranch'](env, auxiliary) if auxiliary else None
            snapshots, previous_plans, actions, rows, details, latencies = [], [], [], [], [], []
            previous = None
            state, xy = api['observable_state'](env.unwrapped, mapping)
            start_hash = hashlib.sha256(state.tobytes() + xy.tobytes()).hexdigest()
            for t in range(cfg['episode_steps']):
                if time.time() > cfg.get('_deadline', float('inf')):
                    raise TimeoutError('Global experiment elapsed cap reached')
                state, _ = api['observable_state'](env.unwrapped, mapping)
                if branch:
                    snapshots.append(branch.capture())
                    previous_plans.append(None if previous is None else previous.copy())
                started = time.perf_counter()
                plan = learner.plan(state, rng, previous)
                latencies.append(time.perf_counter() - started)
                previous = plan.actions.copy()
                action = plan.actions[0].reshape(-1, 1)
                row, detail, done = api['actual_step'](env, metadata, mapping, action, cfg)
                row['truncated'] = bool(row['truncated'] or (t + 1 == cfg['episode_steps'] and not row['terminated']))
                actions.append(action[:, 0].copy()); rows.append(row); details.append(detail)
                if done or row['truncated']:
                    break
            arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
            arrays['episode'] = np.full(len(rows), episode, dtype=np.int64)
            arrays['dt'] = np.asarray([r['dt'] for r in details])
            path = output / f'episode-{episode:02d}.npz'
            np.savez_compressed(path, **arrays)
            record = dict(seed=reset_seed, transitions=len(rows), collapsed=bool(rows[-1]['terminated']),
                displacement=float(arrays['progress'][:, 0].sum() / metadata['initial_diagonal']),
                task_return=float(arrays['reward'].sum()),
                discounted_return=float(sum(cfg['gamma'] ** t * r['reward'] for t, r in enumerate(rows))),
                raw_return=float(sum(r['raw_reward'] for r in details)),
                collapse_time_seconds=float(arrays['dt'].sum()) if rows[-1]['terminated'] else None,
                initial_state_sha256=start_hash, median_planning_seconds=float(np.median(latencies)),
                trace=path.name, trace_sha256=sha256(path))
            if branch:
                t = max(0, len(rows) - 5) if record['collapsed'] else min(20, len(rows) - 1)
                state = arrays['state'][t]
                proposed, _ = propose(learner, state, reset_seed + 2000000, previous_plans[t], api['cem_plan'])
                randoms = np.asarray([api['random_sequence'](np.random.default_rng(reset_seed + 3000000 + i), 10, mask)[..., 0] for i in range(8)])
                candidates = np.concatenate([proposed, np.zeros((1, 10, len(mask))), randoms])
                live_qpos, live_qvel = env.unwrapped.mj_model.data.qpos.copy(), env.unwrapped.mj_model.data.qvel.copy()
                live_time = float(env.unwrapped.mj_model.data.time)
                restoration = verify_restore(branch, snapshots[t], np.asarray(actions[t:t + 10]),
                    rows[t], metadata, cfg, api['actual_step'])
                truth = []
                for candidate in candidates:
                    if time.time() > cfg.get('_deadline', float('inf')):
                        raise TimeoutError('Global experiment cap reached during candidate branches')
                    truth.append(branch_rollout(branch, snapshots[t], candidate, metadata, cfg, api['actual_step']))
                if not (np.array_equal(live_qpos, env.unwrapped.mj_model.data.qpos) and
                        np.array_equal(live_qvel, env.unwrapped.mj_model.data.qvel) and
                        live_time == float(env.unwrapped.mj_model.data.time)):
                    raise AssertionError('Validation branches modified live simulator state')
                candidate_path = output / f'candidates-{episode:02d}.npz'
                np.savez_compressed(candidate_path, state=state, candidates=candidates,
                    scores=np.asarray([r['score'] for r in truth]),
                    collapse=np.asarray([r['collapse'] for r in truth]),
                    displacement=np.asarray([r['displacement'] for r in truth]))
                # Retain exact per-step simulator reward/dt evidence for independent audit.
                import pickle
                with (output / f'branches-{episode:02d}.pkl').open('wb') as stream:
                    pickle.dump(truth, stream)
                record.update(candidate_timestep=t, candidates=candidate_path.name,
                    candidate_sha256=sha256(candidate_path), restoration=restoration)
            records.append(record)
            write_json(output / 'episodes.json', records)
    finally:
        env.close()
        if auxiliary:
            auxiliary.close()
    if model_digest(learner.model) != before:
        raise AssertionError('Frozen evaluation changed model parameters/buffers')
    return records


def validate(api: dict, learner: Any, validation: Path, deadline: float = float('inf')) -> dict:
    predictions, rankings = [], []
    for record in json.loads((validation / 'episodes.json').read_text()):
        if time.time() > deadline:
            raise TimeoutError('Global experiment cap reached during validation')
        path = validation / record['trace']
        if sha256(path) != record['trace_sha256']:
            raise ValueError('Validation trace changed')
        with np.load(path) as raw:
            arrays = dict(raw)
        for horizon in (1, 5, 10):
            for cohort, starts in api['cohorts'](arrays, horizon).items():
                metrics = api['prediction_metrics'](learner.model.dynamics, arrays, starts,
                    horizon, learner.metadata, learner.graph)
                predictions.append(dict(reset_seed=record['seed'], horizon=horizon, cohort=cohort, **metrics))
        path = validation / record['candidates']
        if sha256(path) != record['candidate_sha256']:
            raise ValueError('Validation candidates changed')
        with np.load(path) as values:
            state, candidates = values['state'], values['candidates']
            scores = learner.score(state, candidates)
            failed = predict_failures(learner, state, candidates)
            metrics = ranking_metrics(scores, failed, values['scores'], values['collapse'], values['displacement'])
            _, unique = np.unique(candidates, axis=0, return_index=True)
            sensitivity = ranking_metrics(scores[unique], failed[unique], values['scores'][unique],
                values['collapse'][unique], values['displacement'][unique])
        rankings.append(dict(reset_seed=record['seed'], timestep=record['candidate_timestep'],
            pooled=metrics, unique=sensitivity))
    return dict(predictions=predictions, rankings=rankings)


def worker(mode: str, runtime: str, campaign: str, root: str, seed: int, arm: str,
           options: dict, identity: dict, resume: bool = False, deadline: float = float('inf')) -> dict:
    from contextlib import redirect_stdout, redirect_stderr
    root = Path(root)
    folder = root / (f'prepare-seed{seed}' if mode == 'prepare' else f'{arm}-seed{seed}')
    folder.mkdir(parents=True, exist_ok=resume)
    with (folder / 'worker.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
        begun = time.perf_counter()
        try:
            api = bootstrap(Path(runtime))
            cfg, learners, replays, samplers = load_group(api, Path(campaign), seed, arm, options)
            cfg = dict(cfg, episode_steps=options['episode_steps'], _deadline=deadline)
            starting_model = model_digest(next(iter(learners.values())).model)
            if mode == 'prepare':
                results = {}
                for index, (topology, learner) in enumerate(learners.items()):
                    validation = [options['validation_seed_base'] + index * 10000 + seed * 100 + i for i in range(options['validation_episodes'])]
                    evaluation = [options['evaluation_seed_base'] + index * 10000 + seed * 100 + i for i in range(options['evaluation_episodes'])]
                    for role, resets in (('validation', validation), ('evaluation', evaluation)):
                        target = folder / topology / role
                        target.parent.mkdir(exist_ok=True)
                        if resume and (target / 'episodes.json').exists():
                            records = json.loads((target / 'episodes.json').read_text())
                            if len(records) != len(resets):
                                raise ValueError('Partial episode generation requires a fresh recorded output path')
                        else:
                            records = collect(api, learner, cfg, resets, target, role == 'validation')
                        results.setdefault(topology, {})[role] = records
                    results[topology]['diagnostic'] = validate(api, learner, folder / topology / 'validation', deadline)
                report = dict(status='complete', mode=mode, seed=seed, topologies=results,
                              elapsed_seconds=time.perf_counter() - begun)
                write_json(folder / 'report.json', report)
                return report
            manifest_path = root / 'prepared-data-manifest.json'
            manifest = json.loads(manifest_path.read_text())
            if not all(sha256(Path(p)) == digest for p, digest in manifest.items()):
                raise ValueError('Frozen validation/reference data changed before fitting')
            local_identity = dict(identity, seed=seed, arm=arm, rounds=options['rounds'],
                                  prepared_manifest_sha256=sha256(manifest_path))
            start, elapsed = 0, 0.
            if resume and (folder / 'resume.json').exists():
                start, elapsed = restore_fit(api, folder, learners, samplers, local_identity)
            starting_normalization = {name: {key: value.clone() for key, value in l.model.dynamics.named_buffers()}
                                      for name, l in learners.items()}
            history = restored_history(json.loads((folder / 'progress.json').read_text())['history'], start) if start else []
            losses = []
            for round_number in range(start + 1, options['rounds'] + 1):
                if time.time() > deadline or elapsed + time.perf_counter() - begun > options['cap_seconds']:
                    raise TimeoutError('Declared fitting cap reached; checkpoint retained')
                result = api['update_round'](learners, replays)
                losses.append(result)
                if round_number % 100 == 0 or round_number == options['rounds']:
                    history.append(dict(round=round_number, elapsed_seconds=elapsed + time.perf_counter() - begun,
                        losses={name: {key: float(np.mean([r[name][key] for r in losses])) for key in result[name]}
                                for name in result}))
                    losses.clear()
                    write_json(folder / 'progress.json', dict(history=history))
                if round_number % options['checkpoint_every'] == 0 or round_number == options['rounds']:
                    save_fit(api, folder, round_number, learners, samplers, local_identity,
                             elapsed + time.perf_counter() - begun)
            first = next(iter(learners.values()))
            adam_steps = {int(value['step']) for value in first.optimizer.state.values()}
            if adam_steps != {60000 + 3 * options['rounds']}:
                raise AssertionError('Final shared Adam accounting failed')
            for name, learner in learners.items():
                if learner.updates != 20000 + options['rounds']:
                    raise AssertionError('Final topology view accounting failed')
                for key, value in learner.model.dynamics.named_buffers():
                    if not api['torch'].equal(value, starting_normalization[name][key]):
                        raise AssertionError('Frozen normalization changed during fitting')
                if any(counter['draws'] != options['rounds'] * 32 for counter in samplers[name].counts):
                    raise AssertionError('Replay exposure count mismatch')
            results = {}
            for index, (topology, learner) in enumerate(learners.items()):
                learner.model.eval()
                validation = root / f'prepare-seed{seed}' / topology / 'validation'
                diagnostic = validate(api, learner, validation, deadline)
                resets = [options['evaluation_seed_base'] + index * 10000 + seed * 100 + i for i in range(options['evaluation_episodes'])]
                target = folder / topology / 'evaluation'
                target.parent.mkdir(exist_ok=True)
                if resume and (target / 'episodes.json').exists():
                    evaluation = json.loads((target / 'episodes.json').read_text())
                    if len(evaluation) != len(resets):
                        raise ValueError('Partial evaluation retained; cannot silently overwrite')
                else:
                    evaluation = collect(api, learner, cfg, resets, target, False)
                results[topology] = dict(diagnostic=diagnostic, evaluation=evaluation,
                    sampling=samplers[topology].state_dict(), availability=samplers[topology].availability(),
                    model_sha256=model_digest(learner.model), updates=learner.updates)
                reference = json.loads((root / f'prepare-seed{seed}' / topology / 'evaluation' / 'episodes.json').read_text())
                if [(r['seed'], r['initial_state_sha256']) for r in evaluation] != [
                        (r['seed'], r['initial_state_sha256']) for r in reference]:
                    raise AssertionError('Actual evaluation initial states are not paired')
            if not all(sha256(Path(p)) == digest for p, digest in manifest.items()) or sha256(manifest_path) != local_identity['prepared_manifest_sha256']:
                raise ValueError('Frozen validation/reference data changed during fitting/evaluation')
            report = dict(status='complete', seed=seed, arm=arm, rounds=options['rounds'],
                starting_model_sha256=starting_model,
                adam_steps=sorted(adam_steps), topologies=results, history=history,
                elapsed_seconds=elapsed + time.perf_counter() - begun)
            write_json(folder / 'report.json', report)
            return report
        except Exception:
            write_json(folder / 'failure.json', dict(traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - begun))
            traceback.print_exc()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-root', type=Path, required=True)
    parser.add_argument('--campaign-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'config/failure_balanced_replay.yaml')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    api = bootstrap(args.runtime_root.resolve())
    options = api['OmegaConf'].to_container(api['OmegaConf'].load(args.config))
    if args.benchmark:
        options = dict(options, rounds=options['benchmark_rounds'], checkpoint_every=options['benchmark_rounds'])
    args.output.mkdir(parents=True, exist_ok=args.resume)
    campaign = args.campaign_root.resolve()
    prior = json.loads((campaign / 'protocol.json').read_text())
    import ast
    inputs = {str(campaign / 'protocol.json'): sha256(campaign / 'protocol.json')}
    for relative, digest in prior['sources'].items():
        original = campaign / 'source-snapshot' / relative
        if sha256(original) != digest or ast.dump(ast.parse(original.read_text())) != ast.dump(ast.parse((args.runtime_root / relative).read_text())):
            raise ValueError('Original executed sources differ from runtime')
        inputs[str(original)] = digest
    versions = {name: importlib.metadata.version(name) for name in prior['dependencies']}
    if versions != prior['dependencies']:
        raise ValueError('Original dependency versions differ')
    planned = {role: [options[f'{role}_seed_base'] + i * 10000 + s * 100 + e
        for i in range(3) for s in options['seeds'] for e in range(options[f'{role}_episodes'])]
        for role in ('validation', 'evaluation')}
    if set(planned['validation']) & set(planned['evaluation']):
        raise ValueError('Validation/evaluation reset overlap')
    prior_resets = set()
    def scan(value):
        if isinstance(value, dict):
            for key in ('seed', 'reset_seed'):
                if isinstance(value.get(key), int):
                    prior_resets.add(value[key])
            for child in value.values(): scan(child)
        elif isinstance(value, list):
            for child in value: scan(child)
    for path in campaign.glob('*-seed*/report.json'):
        scan(json.loads(path.read_text())); inputs[str(path)] = sha256(path)
    # Experiment17 and all previous fitting runs used lower namespaces; additionally
    # check actual checkpoint fitting ranges and original reporting records.
    for seed in options['seeds']:
        path = campaign / f'shared-seed{seed}' / 'checkpoint-020000.pt'
        inputs[str(path)] = sha256(path)
        saved = api['torch'].load(path, map_location='cpu', weights_only=False)
        for record in saved['topologies'].values():
            rt = record['runtime']
            fitting = range(rt['reset_base'], rt['reset_base'] + rt['episode'] + 1)
            if any(s in fitting or s in prior_resets or s < 20000000 for values in planned.values() for s in values):
                raise ValueError('New source reset overlaps prior fitting/evaluation')
        del saved
    here = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), here / 'scripts/failure_balanced_helpers.py',
               here / 'scripts/run_online_candidate_ranking.py', here / 'docs/plans/failure_balanced_replay.md', args.config]
    sources += [p for folder in ('sac/dynamics', 'env') for p in (args.runtime_root / folder).rglob('*.py')]
    sources += [args.runtime_root / 'scripts' / name for name in
                ('collect_dynamics_training_data.py', 'run_online_dynamics_pilot.py', 'run_dynamics_multi_diagnostic.py')]
    identity = dict(inputs=inputs, sources={str(p.resolve()): sha256(p) for p in sources},
                    options=options, dependencies=versions)
    protocol = dict(identity=identity, reset_seeds=planned, no_unseen=True,
        fixed_replay_continuation=True, original_settings=prior['settings'],
        created_utc=datetime.now(timezone.utc).isoformat(), benchmark=args.benchmark)
    protocol['started_epoch'] = time.time()
    path = args.output / 'protocol.json'
    if args.resume:
        original = json.loads(path.read_text())
        if original['identity'] != identity:
            raise ValueError('Cannot resume changed source/config/checkpoints')
        protocol = original
    else:
        write_json(path, protocol)
    begun = time.perf_counter()
    if args.benchmark:
        # A benchmark fits only; it does not generate or inspect validation outcomes.
        cfg, learners, replays, samplers = load_group(api, campaign, 0, 'uniform', options)
        for _ in range(options['rounds']):
            api['update_round'](learners, replays)
        elapsed = time.perf_counter() - begun
        write_json(args.output / 'benchmark.json', dict(rounds=options['rounds'], topology_updates=3 * options['rounds'],
            elapsed_seconds=elapsed, estimated_aggregate_fitting_seconds=elapsed * 2000 / options['rounds'] * 6,
            method='One native-width shared model, three source batches per round, one CPU thread'))
        return
    context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=options['workers'], mp_context=context) as pool:
        for mode in ('prepare', 'fit'):
            jobs = [(seed, 'uniform') for seed in options['seeds']] if mode == 'prepare' else [
                (seed, arm) for seed in options['seeds'] for arm in options['arms']]
            futures = []
            for seed, arm in jobs:
                folder = args.output / (f'prepare-seed{seed}' if mode == 'prepare' else f'{arm}-seed{seed}')
                if args.resume and (folder / 'report.json').exists():
                    continue
                futures.append(pool.submit(worker, mode, str(args.runtime_root.resolve()), str(campaign),
                    str(args.output.resolve()), seed, arm, options, identity, args.resume,
                    protocol['started_epoch'] + options['cap_seconds']))
            for future in as_completed(futures):
                result = future.result()
                print(json.dumps(dict(mode=mode, seed=result['seed'], arm=result.get('arm'),
                    status=result['status'], elapsed_seconds=time.perf_counter() - begun)), flush=True)
            if mode == 'prepare':
                prepared = {str(p.resolve()): sha256(p) for folder in args.output.glob('prepare-seed*')
                            for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json', '.npz', '.pkl')}
                manifest_path = args.output / 'prepared-data-manifest.json'
                if manifest_path.exists():
                    if json.loads(manifest_path.read_text()) != prepared:
                        raise ValueError('Frozen prepared-data manifest changed on restart')
                else:
                    write_json(manifest_path, prepared)
    if not all(sha256(Path(p)) == digest for p, digest in {**inputs, **identity['sources']}.items()):
        raise AssertionError('Executed sources or input checkpoints changed')
    write_json(args.output / 'completion.json', dict(complete=True, elapsed_seconds=time.perf_counter() - begun,
        input_hashes_preserved=True, fitting_jobs=6, added_shared_adam_steps=36000, no_experiment3=True))


if __name__ == '__main__':
    main()
