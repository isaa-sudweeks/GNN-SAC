"""Run fixed-budget source-only online dynamics/MPC and reward/value experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'sac')]
from dynamics.dataset import file_sha256, load_dataset
from dynamics.online import OnlineLearner, OnlineReplay
from dynamics.planner import interpolate_knots
from dynamics.training import evaluate_recursive
from scripts.collect_dynamics_training_data import make_training_env, graph_metadata, observable_state, write_json


def random_sequence(rng, steps: int, mask: np.ndarray) -> np.ndarray:
    return np.clip(interpolate_knots(rng.normal(0, .5, (max(2, steps // 5), len(mask))), steps), -1, 1)[:, :, None] * mask[None, :, None]


def actual_step(env, metadata: dict, mapping: np.ndarray, action: np.ndarray, cfg: dict):
    """Execute a real command; label physical progress and the common task reward."""
    base = env.unwrapped
    state, xy = observable_state(base, mapping)
    before = float(base.mj_model.data.time)
    obs, raw, done, info = env.step(action)
    nxt, next_xy = observable_state(base, mapping)
    elapsed = float(base.mj_model.data.time) - before
    progress = next_xy - xy
    effort = float(np.square(base.node_velocity_controller.latest_edge_commands).mean()) * elapsed
    base_reward = float(progress[0] / metadata['initial_diagonal'] - cfg['objective']['effort_weight'] * effort)
    terminated = bool(info['terminated'])
    reward = base_reward - cfg['objective']['failure_cost'] * terminated
    row = dict(state=state, next_state=nxt, action=action, progress=progress, reward=reward,
               base_reward=base_reward, terminated=terminated, truncated=bool(info['truncated']))
    detail = dict(global_xy=xy, next_global_xy=next_xy, dt=elapsed, raw_reward=float(raw),
                  next_rigidity=float(obs['rigidity'][0]), command_integral=effort)
    return row, detail, done


def evaluate(learner, condition: str, cfg: dict, seed: int, steps: int, output: Path) -> dict:
    """Paired independent resets with frozen weights and no exploration or replay writes."""
    env = make_training_env(cfg['topology'], OmegaConf.create(cfg))
    metadata = learner.metadata
    mapping = np.asarray(metadata['node_to_physical'])
    mask = np.asarray(metadata['action_mask'])
    rows, latencies = [], []
    try:
        for episode in range(cfg['evaluation_episodes']):
            reset_seed = 900000 + seed * 10000 + episode
            env.reset(); env.unwrapped.reset(seed=reset_seed)
            rng = np.random.default_rng(950000 + seed * 10000 + episode)
            random_actions = random_sequence(rng, cfg['episode_steps'], mask)
            previous = None
            trace, details, discounted, raw_return = [], [], 0., 0.
            for t in range(cfg['episode_steps']):
                state, _ = observable_state(env.unwrapped, mapping)
                if condition == 'random':
                    action = random_actions[t]
                else:
                    started = time.perf_counter()
                    plan = learner.plan(state, rng, previous)
                    latencies.append(time.perf_counter() - started)
                    previous, action = plan.actions, plan.actions[0].reshape(-1, 1)
                row, detail, done = actual_step(env, metadata, mapping, action, cfg)
                trace.append(row); details.append(detail)
                discounted += cfg['gamma'] ** t * row['reward']
                raw_return += detail['raw_reward']
                if done:
                    break
            name = f'eval-{steps}-{episode}.npz'
            arrays = {k: np.asarray([r[k] for r in trace]) for k in trace[0]}
            arrays.update({k: np.asarray([r[k] for r in details]) for k in details[0]})
            np.savez_compressed(output / name, **arrays)
            rows.append(dict(seed=reset_seed, transitions=len(trace),
                displacement=float(sum(r['progress'][0] for r in trace) / metadata['initial_diagonal']),
                discounted_task_return=discounted, task_return=float(sum(r['reward'] for r in trace)),
                raw_return=raw_return, collapsed=bool(trace[-1]['terminated']),
                initial_state_sha256=hashlib.sha256(trace[0]['state'].tobytes() + details[0]['global_xy'].tobytes()).hexdigest(),
                trace=name, sha256=file_sha256(output / name)))
    finally:
        env.close()
    return dict(steps=steps, episodes=rows, median_displacement=float(np.median([r['displacement'] for r in rows])),
                mean_displacement=float(np.mean([r['displacement'] for r in rows])),
                collapses=sum(r['collapsed'] for r in rows),
                median_planning_seconds=float(np.median(latencies)) if latencies else 0.,
                p95_planning_seconds=float(np.percentile(latencies, 95)) if latencies else 0.)


def run_one(cfg: dict, condition: str, seed: int, output: Path, diagnostics: list[dict]) -> dict:
    output.mkdir()
    torch.manual_seed(seed)
    env = make_training_env(cfg['topology'], OmegaConf.create(cfg))
    begun = time.perf_counter()
    training_episodes, evaluations, history = [], [], []
    losses, planning_times = [], []
    episode_id, t, episode_return, episode_move, episode_raw = -1, 0, 0., 0., 0.
    action_rng = np.random.default_rng(500000 + seed)
    explore_rng = np.random.default_rng(600000 + seed)
    previous = None
    try:
        env.reset(); obs, _ = env.unwrapped.reset(seed=700000 + seed * 10000)
        metadata = graph_metadata(env.unwrapped, obs)
        metadata['control_interval_seconds'] = float(env.unwrapped.nsubsteps * env.unwrapped.mj_model.model.opt.timestep)
        mapping = np.asarray(metadata['node_to_physical'])
        state, _ = observable_state(env.unwrapped, mapping)
        mask = np.asarray(metadata['action_mask'])
        learner = OnlineLearner(metadata, state, cfg, seed, condition == 'reward_value_mpc')
        replay = OnlineReplay(cfg['steps'], len(mapping))
        write_json(output / 'metadata.json', metadata)
        done = True
        for step in range(1, cfg['steps'] + 1):
            if done:
                episode_id += 1
                env.reset(); env.unwrapped.reset(seed=700000 + seed * 10000 + episode_id)
                sequence = random_sequence(action_rng, cfg['episode_steps'], mask)
                t, episode_return, episode_move, episode_raw, previous = 0, 0., 0., 0., None
            state, _ = observable_state(env.unwrapped, mapping)
            if step <= cfg['warmup'] or condition == 'random':
                action = sequence[t]
            else:
                started = time.perf_counter()
                plan = learner.plan(state, action_rng, previous)
                planning_times.append(time.perf_counter() - started)
                previous = plan.actions
                action = np.clip(plan.actions[0].reshape(-1, 1) + explore_rng.normal(0, cfg['exploration_std'], (len(mask), 1)), -1, 1) * mask[:, None]
            row, detail, done = actual_step(env, metadata, mapping, action, cfg)
            t += 1
            row['truncated'] = bool(row['truncated'] or (t == cfg['episode_steps'] and not row['terminated']))
            done = bool(done or row['truncated'])
            replay.append(**row, episode=episode_id)
            episode_return += row['reward']
            episode_move += row['progress'][0] / metadata['initial_diagonal']
            episode_raw += detail['raw_reward']
            if done or step == cfg['steps']:
                training_episodes.append(dict(episode=episode_id, seed=700000 + seed * 10000 + episode_id,
                    ending_step=step, transitions=t, displacement=float(episode_move), task_return=float(episode_return),
                    raw_return=float(episode_raw), collapsed=row['terminated'], truncated=row['truncated'],
                    budget_cutoff=bool(step == cfg['steps'] and not done)))
            if step == cfg['warmup']:
                learner.normalize_from_warmup(replay)
                np.savez_compressed(output / 'warmup.npz', **replay.arrays())
                for _ in range(cfg['initial_updates']):
                    losses.append(learner.update(replay))
            elif step > cfg['warmup']:
                for _ in range(cfg['updates_per_step']):
                    losses.append(learner.update(replay))
            if step in cfg['evaluate_at']:
                evaluations.append(evaluate(learner, condition, cfg, seed, step, output))
                print(json.dumps(dict(condition=condition, seed=seed, evaluation=evaluations[-1],
                    updates=learner.updates, wall_seconds=time.perf_counter() - begun)), flush=True)
            if step % 80 == 0 or step == cfg['steps']:
                history.append(dict(step=step, updates=learner.updates,
                    losses={key: float(np.mean([r[key] for r in losses])) for key in losses[0]} if losses else {},
                    wall_seconds=time.perf_counter() - begun))
                losses.clear()
                write_json(output / 'progress.json', dict(history=history, evaluations=evaluations, training_episodes=training_episodes))
        np.savez_compressed(output / 'replay.npz', **replay.arrays())
        torch.save(dict(model=learner.model.state_dict(), actor=learner.actor.state_dict(),
                        target=learner.target.state_dict(), target_actor=learner.target_actor.state_dict(),
                        optimizer=learner.optimizer.state_dict(), actor_optimizer=learner.actor_optimizer.state_dict(),
                        settings=cfg, condition=condition, seed=seed, updates=learner.updates), output / 'final-model.pt')
        prediction = evaluate_recursive(learner.model.dynamics, diagnostics, [1, 5, 10, 20]) if diagnostics else {}
        prediction['claim_boundary'] = 'Common source-only smooth-random diagnostic episodes; no fitting/checkpoint selection; no transfer claim'
        write_json(output / 'prediction.json', prediction)
        report = dict(condition=condition, seed=seed, interactions=replay.size, updates=learner.updates,
            history=history, evaluations=evaluations, training_episodes=training_episodes,
            checkpoint_sha256=file_sha256(output / 'final-model.pt'), replay_sha256=file_sha256(output / 'replay.npz'),
            warmup_sha256=file_sha256(output / 'warmup.npz'),
            median_train_planning_seconds=float(np.median(planning_times)) if planning_times else 0.,
            wall_seconds=time.perf_counter() - begun,
            prediction_summary=prediction.get('summary', []), status='complete')
        write_json(output / 'report.json', report)
        return report
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/dynamics_online.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-data', type=Path)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config))
    if cfg['topology'] != 'tetrahedron':
        raise ValueError('This separately scoped experiment is source tetrahedron only')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(1)
    diagnostics = [e for e in load_dataset(args.diagnostic_data, ['evaluation']) if e['topology'] == cfg['topology']] if args.diagnostic_data else []
    protocol = dict(settings=cfg, created_utc=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        config_sha256=file_sha256(args.config),
        sources={str(p.relative_to(ROOT)): file_sha256(p) for p in [Path(__file__).resolve(), ROOT / 'sac/dynamics/online.py', ROOT / 'sac/dynamics/model.py']},
        diagnostic_data=str(args.diagnostic_data), diagnostic_episode_hashes=[file_sha256(Path(e['file'])) for e in diagnostics],
        objective='progress / initial diagonal - 0.1 * integral mean squared routed commands - 1 on true collapse',
        planning='discounted H=10 predicted rewards, plus gamma^H terminal min twin Q only in reward/value condition',
        reward_value_method='Explicit observable GNN, scalar Huber reward/twin TD Q, delayed deterministic actor and Polyak targets; TD-MPC2-inspired, not full TD-MPC2',
        termination='True terminal stops bootstrap; artificial 80-step truncation keeps bootstrap; no model state enters real replay',
        data='Same per-seed 160 real random warmup transitions; then each MPC condition executes its learned planner with exploration; no simulator MPC or teacher training data',
        selection='Fixed final checkpoints; evaluate at declared budgets only; no best-test checkpoint or hyperparameter selection',
        stop='Exactly declared seeds/conditions/steps; fail loudly on numerical errors; no automatic tuning extension',
        budget='Smoke benchmark first; initial target <=30 CPU wall minutes for nine runs, revise estimate before expansion',
        limitations=['One source topology, three training seeds and three paired evaluation resets per seed',
                     'Approximate discounted tail value is not an exact infinite-horizon guarantee',
                     'Reward/value arm changes both task supervision and planning tail; cannot isolate their individual effects',
                     'No held-out or final-test topology access'])
    write_json(args.output / 'protocol.json', protocol)
    reports = []
    for seed in cfg['seeds']:
        for condition in cfg['conditions']:
            result = run_one(cfg, condition, seed, args.output / f'{condition}-seed{seed}', diagnostics)
            reports.append(result)
            write_json(args.output / 'report.json', dict(protocol=protocol, runs=reports))


if __name__ == '__main__':
    main()
