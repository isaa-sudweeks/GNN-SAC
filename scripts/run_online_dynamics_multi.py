"""Restartable, fixed-budget shared versus separate source dynamics-MPC experiment."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
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
sys.path[:0] = [str(ROOT), str(ROOT/'sac')]
from dynamics.dataset import SOURCE_TOPOLOGIES, file_sha256
from dynamics.multitopology import normalize_warmups, share_dynamics, update_round
from dynamics.online import OnlineLearner, OnlineReplay
from dynamics.simulator import SimulatorBranch
from dynamics.training import evaluate_recursive
from scripts.collect_dynamics_training_data import make_training_env, graph_metadata, observable_state, write_json
from scripts.run_online_dynamics_pilot import actual_step, evaluate, random_sequence
from scripts.run_online_dynamics_long import save_resume, load_resume

MODULES = ('model', 'actor', 'target', 'target_actor', 'optimizer', 'actor_optimizer')


def prediction(learner, folder: Path, evaluation: dict) -> dict:
    episodes = []
    for index, row in enumerate(evaluation['episodes']):
        with np.load(folder/row['trace']) as arrays:
            episodes.append({**learner.metadata, **dict(arrays), 'role': 'evaluation', 'episode_id': index})
    result = evaluate_recursive(learner.model.dynamics, episodes, [1, 5, 10, 20])
    return dict(summary=[r for r in result['summary'] if r['predictor'] == 'gnn'],
                distribution='Own frozen controller trajectories; not a common distribution comparison')


def run_group(cfg: dict, condition: str, seed: int, output: Path, *, resume: bool = False,
              stop_after: int | None = None) -> dict:
    """One seed/condition, three live graphs, checkpointed at completed rounds."""
    output.mkdir(parents=True, exist_ok=resume)
    envs, auxiliaries, learners, replays, runtime = {}, {}, {}, {}, {}
    begun = time.perf_counter()
    history, evaluations, fresh = [], {}, {}
    elapsed, start = 0., 0
    try:
        for index, topology in enumerate(cfg['topologies']):
            folder = output/topology
            folder.mkdir(exist_ok=resume)
            env = envs[topology] = make_training_env(topology, OmegaConf.create(cfg))
            aux = auxiliaries[topology] = make_training_env(topology, OmegaConf.create(cfg))
            aux.reset()
            reset_base = 700000 + seed*10000 + index*1000000
            env.reset(); obs, _ = env.unwrapped.reset(seed=reset_base)
            metadata = graph_metadata(env.unwrapped, obs)
            metadata['control_interval_seconds'] = float(env.unwrapped.nsubsteps*env.unwrapped.mj_model.model.opt.timestep)
            mapping = np.asarray(metadata['node_to_physical'])
            state, _ = observable_state(env.unwrapped, mapping)
            learners[topology] = OnlineLearner(metadata, state, cfg, seed, False)
            # Every view starts with exactly the same seeded parameters, independent of graph size.
            replays[topology] = OnlineReplay(cfg['steps'], len(mapping))
            runtime[topology] = dict(episode=-1, t=0, done=True, previous=None, sequence=None,
                reset_base=reset_base, action_rng=np.random.default_rng(500000+seed+index*1000000),
                explore_rng=np.random.default_rng(600000+seed+index*1000000), planning_times=[])
            write_json(folder/'metadata.json', metadata)
        if condition == 'shared':
            share_dynamics(learners)
        statistics = None
        initial = {name: {k: v.clone() for k,v in learner.model.dynamics.state_dict().items()}
                   for name,learner in learners.items()}
        if not resume:
            torch.save(initial, output/'initial-dynamics.pt')
        if resume:
            saved = load_resume(output, cfg)
            if saved['condition'] != condition or saved['seed'] != seed:
                raise ValueError('Run identity mismatch')
            for name, learner in learners.items():
                record = saved['topologies'][name]
                if record['metadata'] != learner.metadata:
                    raise ValueError('Simulator/metadata mismatch')
                for key in MODULES:
                    getattr(learner,key).load_state_dict(record[key])
                learner.updates = record['updates']
                learner.rng.bit_generator.state = record['learner_rng']
                replay = replays[name]
                for key, values in record['replay'].items():
                    getattr(replay,key)[:len(values)].copy_(torch.as_tensor(values))
                replay.size = saved['step']
                runtime[name] = record['runtime']
                SimulatorBranch(auxiliaries[name], envs[name]).restore(record['simulator'])
            torch.set_rng_state(saved['torch_rng']); np.random.set_state(saved['numpy_rng']); random.setstate(saved['python_rng'])
            history, evaluations, statistics = saved['history'], saved['evaluations'], saved['statistics']
            start, elapsed = saved['step'], saved['wall_seconds']
        for step in range(start+1, cfg['steps']+1):
            for name,learner in learners.items():
                env, replay, rt = envs[name], replays[name], runtime[name]
                mapping = np.asarray(learner.metadata['node_to_physical'])
                mask = np.asarray(learner.metadata['action_mask'])
                if rt['done']:
                    rt['episode'] += 1
                    env.reset(); env.unwrapped.reset(seed=rt['reset_base']+rt['episode'])
                    rt.update(t=0, previous=None, sequence=random_sequence(rt['action_rng'], cfg['episode_steps'], mask))
                state, _ = observable_state(env.unwrapped, mapping)
                if step <= cfg['warmup']:
                    action = rt['sequence'][rt['t']]
                else:
                    started = time.perf_counter()
                    plan = learner.plan(state, rt['action_rng'], rt['previous'])
                    rt['planning_times'].append(time.perf_counter()-started)
                    rt['previous'] = plan.actions
                    action = np.clip(plan.actions[0].reshape(-1,1)+rt['explore_rng'].normal(0,cfg['exploration_std'],(len(mask),1)), -1,1)*mask[:,None]
                row, _, done = actual_step(env, learner.metadata, mapping, action, cfg)
                rt['t'] += 1
                row['truncated'] = bool(row['truncated'] or (rt['t'] == cfg['episode_steps'] and not row['terminated']))
                rt['done'] = bool(done or row['truncated'])
                replay.append(**row, episode=rt['episode'])
            # Collect the complete round first. Every optimizer sees exactly balanced exposure.
            if step == cfg['warmup']:
                statistics = normalize_warmups(learners, replays)
                write_json(output/'normalization.json', statistics)
                for name,replay in replays.items():
                    np.savez_compressed(output/name/'warmup.npz', **replay.arrays())
                for _ in range(cfg['initial_updates']):
                    losses = update_round(learners,replays)
            elif step > cfg['warmup']:
                for _ in range(cfg['updates_per_step']):
                    losses = update_round(learners,replays)
            if step in cfg['evaluate_at']:
                for index,(name,learner) in enumerate(learners.items()):
                    local = dict(cfg, topology=name)
                    evaluation = evaluate(learner,'dynamics_mpc',local,seed+index*100,step,output/name)
                    evaluation['prediction'] = prediction(learner,output/name,evaluation)
                    evaluations.setdefault(name,[]).append(evaluation)
            if step % 80 == 0 or step == stop_after or step == cfg['steps']:
                history.append(dict(step=step,updates={name:l.updates for name,l in learners.items()},
                    losses=losses if step >= cfg['warmup'] else {}, wall_seconds=elapsed+time.perf_counter()-begun))
                write_json(output/'progress.json',dict(condition=condition,seed=seed,history=history,evaluations=evaluations))
            if step in cfg['evaluate_at'] or step % cfg['checkpoint_every'] == 0 or step == stop_after:
                saved = dict(settings=cfg,condition=condition,seed=seed,step=step,history=history,evaluations=evaluations,
                    statistics=statistics,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
                    wall_seconds=elapsed+time.perf_counter()-begun,topologies={})
                for name,learner in learners.items():
                    saved['topologies'][name] = dict(metadata=learner.metadata,updates=learner.updates,
                        **{key:getattr(learner,key).state_dict() for key in MODULES},
                        replay=replays[name].arrays(),runtime=runtime[name],learner_rng=learner.rng.bit_generator.state,
                        simulator=SimulatorBranch(envs[name],auxiliaries[name]).capture())
                save_resume(output,saved,milestone=step in cfg['evaluate_at'])
            if step == stop_after:
                return dict(status='paused_for_resume_test',step=step)
        for index,(name,learner) in enumerate(learners.items()):
            folder = output/name/'fresh-final'; folder.mkdir(exist_ok=resume)
            local = dict(cfg,topology=name,evaluation_episodes=cfg['fresh_evaluation_episodes'])
            fresh[name] = evaluate(learner,'dynamics_mpc',local,seed+index*100+30,cfg['steps'],folder)
            fresh[name]['prediction'] = prediction(learner,folder,fresh[name])
            np.savez_compressed(output/name/'replay.npz',**replays[name].arrays())
        report = dict(condition=condition,seed=seed,status='complete',evaluations=evaluations,fresh=fresh,
            history=history,statistics=statistics,interactions={n:r.size for n,r in replays.items()},
            updates={n:l.updates for n,l in learners.items()},
            planning_seconds={n:dict(median=float(np.median(rt['planning_times'])),p95=float(np.percentile(rt['planning_times'],95))) for n,rt in runtime.items()},
            wall_seconds=elapsed+time.perf_counter()-begun)
        write_json(output/'report.json',report)
        return report
    finally:
        for env in [*envs.values(),*auxiliaries.values()]:
            env.close()


def worker(cfg: dict, condition: str, seed: int, root: str, resume: bool) -> dict:
    torch.set_num_threads(1)
    root = Path(root)
    folder = root/f'{condition}-seed{seed}'
    with (root/f'{condition}-seed{seed}.log').open('a') as log, redirect_stdout(log), redirect_stderr(log):
        try:
            if resume and (folder/'report.json').exists():
                return json.loads((folder/'report.json').read_text())
            return run_group(cfg,condition,seed,folder,resume=resume and (folder/'resume.json').exists())
        except Exception:
            result = dict(condition=condition,seed=seed,status='failed',traceback=traceback.format_exc())
            write_json(root/f'{condition}-seed{seed}-failure-{time.time_ns()}.json',result)
            traceback.print_exc()
            return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'config/dynamics_online_multi.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    cfg=OmegaConf.to_container(OmegaConf.load(args.config))
    if len(cfg['topologies']) != 3 or len(set(cfg['topologies'])) != 3 or not set(cfg['topologies']) <= set(SOURCE_TOPOLOGIES):
        raise ValueError('Exactly three distinct source topologies required')
    if cfg['conditions'] != ['shared','separate']:
        raise ValueError('Only shared versus separate dynamics MPC authorized')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    sources=[Path(__file__).resolve(),ROOT/'scripts/run_online_dynamics_pilot.py',ROOT/'scripts/run_online_dynamics_long.py',ROOT/'scripts/report_online_dynamics_multi.py',
             ROOT/'scripts/collect_dynamics_training_data.py',*sorted((ROOT/'sac/dynamics').glob('*.py'))]
    benchmark_path=ROOT/'outputs/dynamics-pilot/online-multi-benchmark-2026-10-24/benchmark.json'
    benchmark=json.loads(benchmark_path.read_text())
    protocol=dict(benchmark=benchmark,settings=cfg,sources={str(p.relative_to(ROOT)):file_sha256(p) for p in sources},
        config_sha256=file_sha256(args.config),created_utc=datetime.now(timezone.utc).isoformat(),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dependencies={p:importlib.metadata.version(p) for p in ['torch','numpy','mujoco','mujoco-truss-gen','omegaconf']},
        accounting='Per topology: 20000 real transitions, 20000 updates, 640000 one-step draws, 640000 five-step windows (3200000 recursive target exposures). Shared optimizer 60000 updates; each separate optimizer 20000. Strict round robin.',
        normalization='Per-seed equal-topology moments of matched 160-transition random fitting warmups. Same common physical-unit mean/std/delta RMS and mean initial diagonal for all four models. Frozen thereafter.',
        stop='Fixed 20000 transitions/topology; numerical failure stops affected group and is retained. No tuning, checkpoint selection, budget extension or unseen topology access.',
        selection='Fixed monitoring checkpoints and independent fresh final resets; no outcome-based checkpoint selection.',
        claim='Shared learning across three trained source topologies only. See docs/plans/gnn_dynamics_online_multi.md and benchmark.json.')
    if args.resume:
        original=json.loads((args.output/'protocol.json').read_text())
        for key in ('settings','sources','config_sha256','dependencies','benchmark'):
            if original[key] != protocol[key]:
                raise ValueError(f'Cannot resume changed {key}')
        protocol=original
    else:
        write_json(args.output/'protocol.json',protocol)
    begun=time.perf_counter(); reports=[]
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(worker,cfg,c,s,str(args.output),args.resume) for s in cfg['seeds'] for c in cfg['conditions']]
        for future in as_completed(futures):
            reports.append(future.result())
            write_json(args.output/'report.json',dict(protocol=protocol,runs=reports,wall_seconds=time.perf_counter()-begun))
            print(json.dumps(dict(completed=len(reports),status=reports[-1]['status'],wall_seconds=time.perf_counter()-begun)),flush=True)
    if any(r['status'] != 'complete' for r in reports):
        raise SystemExit('Run failures retained; no automatic repair')
    from scripts.report_online_dynamics_multi import report
    report(args.output)


if __name__ == '__main__':
    main()
