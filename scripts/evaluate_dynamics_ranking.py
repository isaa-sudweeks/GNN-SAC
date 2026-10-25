"""Source-only model/simulator candidate ranking on trusted indexed snapshots.

This diagnostic uses actual CEM proposals. Its small source state sample is not
the predeclared 20-state-per-topology held-out evaluation and cannot pass that gate.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'sac')]
from dynamics.dataset import file_sha256, load_dataset, SOURCE_TOPOLOGIES
from dynamics.planner import CEMConfig, cem_plan, planning_score
from dynamics.rigidity import normalized_rigidity
from dynamics.simulator import SimulatorBranch
from dynamics.training import load_model, tensor_graph
from scripts.collect_dynamics_training_data import make_training_env, observable_state


@torch.no_grad()
def model_scores(model, initial: np.ndarray, sequences: np.ndarray, episode: dict, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    """Score recursive predictions using only current state, static graph and commands."""
    graph = tensor_graph(episode)
    n, horizon = sequences.shape[:2]
    state = torch.as_tensor(np.repeat(initial[None], n, axis=0), dtype=torch.float32)
    actions = torch.as_tensor(sequences, dtype=torch.float32)
    displacement = np.zeros(n)
    efforts = np.zeros(n)
    failed = np.zeros(n, dtype=bool)
    elapsed = np.zeros(n)
    dt = float(episode['dt'][0])
    physical_order = graph['physical_order_index'].cpu().numpy()
    incidence = np.asarray(episode['routing_incidence'])
    ctrlrange = np.asarray(episode['ctrlrange'])
    bounds = np.asarray(episode['action_mask'])
    for step in range(horizon):
        active = ~failed
        predicted, progress = model.step(state, actions[:, step], graph)
        values = predicted.numpy()
        finite = np.isfinite(values).all(axis=(1, 2)) & np.isfinite(progress.numpy()).all(axis=1)
        rigidity = np.asarray([normalized_rigidity(p[physical_order, :3], np.asarray(episode['structural_edges'], dtype=int),
            float(episode['initial_rigidity_reference']), axis_indices=tuple(episode['axis_indices']),
            weighted=bool(episode['wcrm'])) if good else np.nan for p, good in zip(values, finite)])
        commands = sequences[:, step, :, 0] * bounds * float(episode['speed'])
        routed = np.clip(commands @ incidence.T, ctrlrange[:, 0], ctrlrange[:, 1])
        efforts[active] += np.square(routed[active]).mean(axis=1) * dt
        displacement[active] += progress.numpy()[active, 0]
        elapsed[active] += dt
        failed |= active & (~finite | ~np.isfinite(rigidity) | (rigidity < float(settings["collapse_threshold"])))
        state = predicted
    scores = np.asarray([planning_score(float(move) if np.isfinite(move) else 0., float(episode['initial_diagonal']),
        float(effort), effort_weight=float(settings["objective"]["effort_weight"]), collapsed=bool(collapse),
        remaining_seconds=max(0., horizon * dt - seconds),
        failure_cost=float(settings["objective"]["failure_cost"]),
        failure_cost_per_second=float(settings["objective"]["failure_cost_per_second"]))
        for move, effort, collapse, seconds in zip(displacement, efforts, failed, elapsed)])
    return scores, failed


def run(data: Path, checkpoint: Path, output: Path, topologies: list[str], states: int = 4, config: Path = ROOT / "config/dynamics_ranking.yaml") -> dict:
    """Retain every candidate and collapse outcome; do not tune from this report."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    collection_protocol = json.loads((data/'collection-protocol.json').read_text())
    settings = OmegaConf.create(collection_protocol['settings'])
    ranking = OmegaConf.to_container(OmegaConf.load(config))
    cfg = CEMConfig(**ranking["cem"])
    protocol = dict(topologies=topologies, role='evaluation', states_per_topology=states, settings=ranking,
        selection='First declared evaluation episodes; snapshot timestep20 or last available <=20',
        candidates='Actual 32x2 CEM proposals scored by frozen model; same64 simulator branches',
        horizon=10, control_interval_seconds=.2, planner_seed=51000,
        objective='forward / diagonal - .1 integrated mean clipped routed actuator command squared; failure1 plus remaining seconds',
        limits='Source diagnostic only; no teacher/near-failure coverage guarantee; no tuning or holdout gate claim',
        checkpoint_sha256=file_sha256(checkpoint), dataset_manifest_sha256=file_sha256(data/'manifest.json'),
        collection_protocol_sha256=file_sha256(data/'collection-protocol.json'))
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    model = load_model(checkpoint)
    torch.set_num_threads(1)
    manifest = json.loads((data/'manifest.json').read_text())
    episodes = load_dataset(data, roles=['evaluation'])
    results = []
    started = time.perf_counter()
    for topology in topologies:
        selected = [e for e in episodes if e['topology'] == topology][:states]
        live, planning = make_training_env(topology, settings), make_training_env(topology, settings)
        try:
            live.reset(); planning.reset()
            branch = SimulatorBranch(live, planning)
            for episode in selected:
                record = next(r for r in manifest['episodes'] if r['file'] == str(Path(episode['file']).relative_to(data)))
                snapshot_file = data/record['snapshots']
                if file_sha256(snapshot_file) != record['snapshots_sha256']:
                    raise ValueError('Snapshot hash mismatch')
                with gzip.open(snapshot_file,'rb') as stream:
                    snapshots = pickle.load(stream)  # explicitly trusted local output
                timestep = max(t for t in record['snapshot_timesteps'] if t <= int(ranking['snapshot_timestep']))
                snapshot = snapshots[record['snapshot_timesteps'].index(timestep)]
                branch.restore(snapshot)
                restored, _ = observable_state(planning.unwrapped, np.asarray(episode['node_to_physical']))
                if not np.allclose(restored, episode['state'][timestep], atol=1e-10):
                    raise ValueError('Restored ranking state does not match indexed observation')
                batches = []
                def evaluate(sequences):
                    shaped = sequences.reshape(-1, cfg.horizon, *live.action_space.shape)
                    batches.append(shaped.copy())
                    return model_scores(model, episode['state'][timestep], shaped, episode, ranking)[0]
                seed = int(ranking["planner_seed"]) + int(episode['seed'])
                plan = cem_plan(evaluate, int(np.prod(live.action_space.shape)), cfg,
                    np.random.default_rng(seed), action_mask=np.asarray(episode['action_mask']))
                sequences = np.concatenate(batches)
                predicted_scores, predicted_failures = model_scores(model, episode['state'][timestep], sequences, episode, ranking)
                true_failures = []
                def score(result, horizon):
                    collapse = result.terminated or result.nonfinite
                    true_failures.append(collapse)
                    effort = sum(float(info['energy_penalty_raw']) * elapsed / len(episode['routing_incidence'])
                                 for info, elapsed in zip(result.infos, np.diff(result.elapsed_times)))
                    return planning_score(float(result.com_positions[-1,0]-result.com_positions[0,0]),
                        float(episode['initial_diagonal']), effort, effort_weight=float(ranking['objective']['effort_weight']), collapsed=collapse,
                        remaining_seconds=max(0., horizon*float(episode['dt'][0])-result.elapsed_times[-1]),
                        failure_cost=float(ranking['objective']['failure_cost']),
                        failure_cost_per_second=float(ranking['objective']['failure_cost_per_second']))
                true_scores = branch.evaluate(snapshot, sequences, score)
                true_failures = np.asarray(true_failures, dtype=bool)
                informative = np.ptp(predicted_scores) > 1e-12 and np.ptp(true_scores) > 1e-12
                correlation = float(spearmanr(predicted_scores, true_scores).statistic) if informative else None
                best = int(np.argmax(predicted_scores))
                row = dict(topology=topology, seed=episode['seed'], timestep=timestep, planner_seed=seed,
                    candidates=len(sequences), spearman=correlation, informative=bool(informative),
                    selected_index=best, simulator_best_index=int(np.argmax(true_scores)),
                    selected_sequence_regret=float(true_scores.max()-true_scores[best]),
                    selected_collapses=bool(true_failures[best]),
                    collapse_false_negatives=int((true_failures & ~predicted_failures).sum()),
                    true_collapses=int(true_failures.sum()), predicted_collapses=int(predicted_failures.sum()),
                    model_scores=predicted_scores.tolist(), simulator_scores=true_scores.tolist(),
                    true_failure_flags=true_failures.tolist(), predicted_failure_flags=predicted_failures.tolist())
                np.savez_compressed(output/f'{topology}-{episode["episode_id"]}-candidates.npz', sequences=sequences)
                results.append(row)
                print(json.dumps({k: row[k] for k in ['topology','seed','spearman','selected_sequence_regret','selected_collapses']}),flush=True)
                report=dict(protocol=protocol, results=results, wall_seconds=time.perf_counter()-started,
                    median_spearman=float(np.median([r['spearman'] for r in results if r['informative']])) if any(r['informative'] for r in results) else None,
                    noninformative_states=sum(not r['informative'] for r in results),
                    claim_boundary='Small source-only ranking diagnostic; frozen checkpoint; no transfer/control result')
                (output/'report.json').write_text(json.dumps(report,indent=2))
        finally:
            live.close();planning.close()
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--topologies',nargs='+',choices=SOURCE_TOPOLOGIES,default=['tetrahedron'])
    parser.add_argument('--states',type=int,default=4)
    parser.add_argument('--config',type=Path,default=ROOT/'config/dynamics_ranking.yaml')
    args=parser.parse_args()
    run(args.data.resolve(), args.checkpoint, args.output, args.topologies, args.states, args.config)
