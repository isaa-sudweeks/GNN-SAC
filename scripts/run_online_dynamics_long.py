"""Fixed-method, restartable 20k online source experiment.

The original short-run implementation is preserved. This runner adds persistence,
exact-prefix verification, independent workers, and fresh final evaluation only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'sac')]
from dynamics.dataset import file_sha256, load_dataset
from dynamics.online import OnlineLearner, OnlineReplay
from dynamics.simulator import SimulatorBranch
from dynamics.training import evaluate_recursive
from scripts.run_online_dynamics_pilot import actual_step, random_sequence, evaluate
from scripts.collect_dynamics_training_data import make_training_env, graph_metadata, observable_state, write_json


def save_resume(output: Path, saved: dict, *, milestone: bool) -> None:
    """Atomically replace the latest full state; retain declared milestone states."""
    temporary = output / 'resume.tmp'
    torch.save(saved, temporary)
    destination = output / 'resume.pt'
    temporary.replace(destination)
    digest = file_sha256(destination)
    if milestone:
        import shutil
        shutil.copy2(destination, output / f"checkpoint-{saved['step']:06d}.pt")
    write_json(output / 'resume.json', dict(file='resume.pt', sha256=digest, step=saved['step']))


def load_resume(output: Path, cfg: dict) -> dict:
    """Load only a trusted local checkpoint matching the exact declared protocol."""
    manifest = json.loads((output / 'resume.json').read_text())
    path = output / manifest['file']
    if file_sha256(path) != manifest['sha256']:
        raise ValueError('Resume checkpoint hash mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['settings'] != cfg or saved['step'] != manifest['step']:
        raise ValueError('Resume settings/step mismatch')
    return saved


def nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(nested_equal(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return left == right


def compare_prefix(learner, replay, output: Path, old: Path) -> None:
    """Prove the 1,600-step prefix matches the prior completed experiment."""
    saved = torch.load(old / 'final-model.pt', map_location='cpu', weights_only=False)
    checks = {name: nested_equal(getattr(learner, name).state_dict(), saved[name]) for name in
              ('model', 'actor', 'target', 'target_actor', 'optimizer', 'actor_optimizer')}
    with np.load(old / 'replay.npz') as arrays:
        checks['real_replay'] = all(np.array_equal(replay.arrays()[k], arrays[k]) for k in arrays.files)
    old_report = json.loads((old / 'report.json').read_text())
    checks['evaluation_arrays'] = True
    for row in old_report['evaluations'][-1]['episodes']:
        with np.load(old / row['trace']) as previous, np.load(output / row['trace']) as current:
            checks['evaluation_arrays'] &= all(np.array_equal(previous[k], current[k]) for k in previous.files)
    write_json(output / 'prefix-comparison.json', dict(step=1600, checks=checks, passed=all(checks.values())))
    if not all(checks.values()):
        raise ValueError(f'Long-run prefix changed: {checks}')


def run_long_one(cfg: dict, condition: str, seed: int, output: Path, diagnostics: list[dict], *, resume: bool = False, stop_after: int | None = None, prefix_root: Path | None = None) -> dict:
    output.mkdir(exist_ok=resume)
    torch.manual_seed(seed)
    env = make_training_env(cfg['topology'], OmegaConf.create(cfg))
    auxiliary = make_training_env(cfg['topology'], OmegaConf.create(cfg))
    auxiliary.reset()
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
        start_step = 0
        resumed_wall = 0.
        state_capture = SimulatorBranch(env, auxiliary)
        state_restore = SimulatorBranch(auxiliary, env)
        if resume:
            saved = load_resume(output, cfg)
            metadata = saved['metadata']
            for name in ('model', 'actor', 'target', 'target_actor', 'optimizer', 'actor_optimizer'):
                getattr(learner, name).load_state_dict(saved[name])
            learner.updates = saved['updates']
            for name, values in saved['replay'].items():
                getattr(replay, name)[:len(values)].copy_(torch.as_tensor(values))
            replay.size = saved['step']
            training_episodes, evaluations, history = saved['training_episodes'], saved['evaluations'], saved['history']
            losses, planning_times = saved['losses'], saved['planning_times']
            episode_id, t, episode_return, episode_move, episode_raw = saved['episode_runtime']
            done, sequence, previous = saved['done'], saved['sequence'], saved['previous']
            action_rng.bit_generator.state = saved['action_rng']
            explore_rng.bit_generator.state = saved['explore_rng']
            learner.rng.bit_generator.state = saved['learner_rng']
            state_restore.restore(saved['simulator'])
            torch.set_rng_state(saved['torch_rng'])
            np.random.set_state(saved['numpy_rng'])
            random.setstate(saved['python_rng'])
            start_step, resumed_wall = saved['step'], saved['wall_seconds']
        for step in range(start_step + 1, cfg['steps'] + 1):
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
                evaluations[-1]['terminal_calibration'] = terminal_calibration(learner, replay)
                if step == 1600 and prefix_root is not None:
                    compare_prefix(learner, replay, output, prefix_root / f'{condition}-seed{seed}')
                print(json.dumps(dict(condition=condition, seed=seed, evaluation=evaluations[-1],
                    updates=learner.updates, wall_seconds=resumed_wall + time.perf_counter() - begun)), flush=True)
            if step % 80 == 0 or step == cfg['steps']:
                history.append(dict(step=step, updates=learner.updates,
                    losses={key: float(np.mean([r[key] for r in losses])) for key in losses[0]} if losses else {},
                    wall_seconds=resumed_wall + time.perf_counter() - begun))
                losses.clear()
                write_json(output / 'progress.json', dict(history=history, evaluations=evaluations, training_episodes=training_episodes))
            if step in cfg['evaluate_at'] or step % int(cfg['checkpoint_every']) == 0 or step == stop_after:
                saved = dict(settings=cfg, condition=condition, seed=seed, step=step, updates=learner.updates,
                    metadata=metadata, replay=replay.arrays(),
                    **{name: getattr(learner, name).state_dict() for name in
                       ('model', 'actor', 'target', 'target_actor', 'optimizer', 'actor_optimizer')},
                    training_episodes=training_episodes, evaluations=evaluations, history=history,
                    losses=losses, planning_times=planning_times,
                    episode_runtime=(episode_id, t, episode_return, episode_move, episode_raw),
                    done=done, sequence=sequence, previous=previous,
                    action_rng=action_rng.bit_generator.state, explore_rng=explore_rng.bit_generator.state,
                    learner_rng=learner.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                    simulator=state_capture.capture(), wall_seconds=resumed_wall + time.perf_counter() - begun)
                save_resume(output, saved, milestone=step in cfg['evaluate_at'])
            if step == stop_after:
                return dict(status='paused_for_resume_test', interactions=replay.size)
        fresh_cfg = dict(cfg, evaluation_episodes=int(cfg['fresh_evaluation_episodes']))
        fresh_output = output / 'fresh-final'
        fresh_output.mkdir(exist_ok=resume)
        fresh_evaluation = evaluate(learner, condition, fresh_cfg, seed + 30, cfg['steps'], fresh_output)
        fresh_random_output = output / 'fresh-random'
        fresh_random_output.mkdir(exist_ok=resume)
        fresh_random = evaluate(learner, 'random', fresh_cfg, seed + 30, cfg['steps'], fresh_random_output)
        write_json(output / 'fresh-evaluation.json', fresh_evaluation)
        write_json(output / 'fresh-random.json', fresh_random)
        np.savez_compressed(output / 'replay.npz', **replay.arrays())
        torch.save(dict(model=learner.model.state_dict(), actor=learner.actor.state_dict(),
                        target=learner.target.state_dict(), target_actor=learner.target_actor.state_dict(),
                        optimizer=learner.optimizer.state_dict(), actor_optimizer=learner.actor_optimizer.state_dict(),
                        settings=cfg, condition=condition, seed=seed, updates=learner.updates), output / 'final-model.pt')
        prediction = evaluate_recursive(learner.model.dynamics, diagnostics, [1, 5, 10, 20]) if diagnostics else {}
        prediction['claim_boundary'] = 'Common source-only smooth-random diagnostic episodes; no fitting/checkpoint selection; no transfer claim'
        write_json(output / 'prediction.json', prediction)
        report = dict(condition=condition, seed=seed, interactions=replay.size, updates=learner.updates,
            history=history, evaluations=evaluations, fresh_evaluation=fresh_evaluation, fresh_random=fresh_random, training_episodes=training_episodes,
            checkpoint_sha256=file_sha256(output / 'final-model.pt'), replay_sha256=file_sha256(output / 'replay.npz'),
            warmup_sha256=file_sha256(output / 'warmup.npz'),
            median_train_planning_seconds=float(np.median(planning_times)) if planning_times else 0.,
            wall_seconds=resumed_wall + time.perf_counter() - begun,
            prediction_summary=prediction.get('summary', []), status='complete')
        write_json(output / 'report.json', report)
        return report
    finally:
        env.close()
        auxiliary.close()



def terminal_calibration(learner: OnlineLearner, replay: OnlineReplay) -> dict:
    """Describe min-Q error on realized fitting terminals without modifying learning."""
    if not learner.use_value:
        return dict(applicable=False)
    indices = torch.nonzero(replay.terminated[:replay.size], as_tuple=True)[0]
    if not len(indices):
        return dict(applicable=True, terminal_transitions=0)
    predictions = []
    with torch.no_grad():
        for chunk in indices.split(256):
            _, first, second = learner.model.task(replay.state[chunk], replay.action[chunk], learner.graph)
            predictions.append(torch.minimum(first, second) * float(learner.settings['reward_scale']))
    predicted = torch.cat(predictions)
    truth = replay.reward[indices]
    return dict(applicable=True, terminal_transitions=len(indices), mean_predicted_q=float(predicted.mean()),
                mean_terminal_reward=float(truth.mean()), mae=float((predicted-truth).abs().mean()),
                mean_optimism=float((predicted-truth).mean()))


def worker(cfg: dict, condition: str, seed: int, output: str, diagnostic_data: str,
           prefix_root: str, resume: bool) -> dict:
    """Isolate each run's Torch/RNG state in a separate spawned CPU process."""
    from contextlib import redirect_stdout, redirect_stderr
    root = Path(output)
    folder = root / f'{condition}-seed{seed}'
    torch.set_num_threads(1)
    with (root / f'{condition}-seed{seed}.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            if (folder / 'report.json').exists() and resume:
                return json.loads((folder / 'report.json').read_text())
            diagnostics = [e for e in load_dataset(Path(diagnostic_data), ['evaluation']) if e['topology'] == cfg['topology']] if diagnostic_data else []
            return run_long_one(cfg, condition, seed, folder, diagnostics,
                                resume=resume and folder.exists(), prefix_root=Path(prefix_root) if prefix_root else None)
        except Exception:
            failure = dict(condition=condition, seed=seed, status='failed', traceback=traceback.format_exc())
            write_json(root / f'{condition}-seed{seed}-failure.json', failure)
            traceback.print_exc()
            return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'config/dynamics_online_long.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-data', type=Path)
    parser.add_argument('--prefix-root', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config))
    if cfg['topology'] != 'tetrahedron' or set(cfg['conditions']) != {'dynamics_mpc', 'reward_value_mpc'}:
        raise ValueError('Long experiment is restricted to the two approved tetrahedron conditions')
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    sources = [Path(__file__).resolve(), ROOT/'scripts/run_online_dynamics_pilot.py', ROOT/'sac/dynamics/online.py',
               ROOT/'sac/dynamics/model.py', ROOT/'sac/dynamics/simulator.py']
    protocol = dict(settings=cfg, sources={str(p.relative_to(ROOT)):file_sha256(p) for p in sources},
        config_sha256=file_sha256(args.config), created_utc=datetime.now(timezone.utc).isoformat(),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dependencies={p:importlib.metadata.version(p) for p in ['torch','numpy','mujoco','mujoco-truss-gen','omegaconf']},
        diagnostic_data=str(args.diagnostic_data), prefix_root=str(args.prefix_root),
        protocol='Fixed-method longer rerun from identical seeds; verify exact 1600-step prefix. No loss/architecture/exploration changes.',
        selection='Fixed checkpoints at 1600/4000/8000/12000/20000; no checkpoint selection by evaluation.',
        evaluation='Three existing paired resets at each checkpoint; five fresh paired resets plus matched random control only at final budget.',
        restart='Full real replay, models, optimizers, RNGs, episode counters, CEM plan and native simulator/wrapper snapshot; hashes checked.',
        budget='Six runs, at most three independent CPU workers with one Torch thread each; target <=3 aggregate run wall hours; stop at 20k each.',
        claim_boundary='Source tetrahedron only; no multi-topology expansion, held-out fitting/tuning or final-test access.')
    protocol_path = args.output/'protocol.json'
    if args.resume:
        original = json.loads(protocol_path.read_text())
        for key in ('settings','sources','config_sha256','diagnostic_data','prefix_root'):
            if original[key] != protocol[key]:
                raise ValueError(f'Cannot resume with changed {key}')
        protocol = original
    else:
        write_json(protocol_path, protocol)
    reports = []
    context = multiprocessing.get_context('spawn')
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(cfg['workers']), mp_context=context) as executor:
        pending = {executor.submit(worker,cfg,condition,seed,str(args.output),str(args.diagnostic_data) if args.diagnostic_data else '',
                                   str(args.prefix_root) if args.prefix_root else '',args.resume): (condition,seed)
                   for condition in cfg['conditions'] for seed in cfg['seeds']}
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                condition, seed = pending.pop(future)
                try:
                    reports.append(future.result())
                except Exception:
                    reports.append(dict(condition=condition,seed=seed,status='failed',traceback=traceback.format_exc()))
                write_json(args.output/'report.json',dict(protocol=protocol,runs=reports))
            progress = []
            for condition,seed in pending.values():
                path = args.output/f'{condition}-seed{seed}'/'progress.json'
                if path.exists():
                    current=json.loads(path.read_text())
                    progress.append(dict(condition=condition,seed=seed,step=current['history'][-1]['step']))
            heartbeat=dict(completed=len(reports),pending=len(pending),progress=progress,wall_seconds=time.perf_counter()-started)
            write_json(args.output/'heartbeat.json',heartbeat)
            print(json.dumps(heartbeat),flush=True)
    if any(r['status'] != 'complete' for r in reports):
        raise SystemExit('At least one run failed; evidence retained, no methodological changes attempted')


if __name__ == '__main__':
    main()
