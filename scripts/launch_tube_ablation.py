#!/usr/bin/env python3
"""Launch matched signed, membership and optional tube-physics screens on ORC."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'sac'):
    sys.path.insert(0, str(path))
from common.parser import parse_cfg
from common.gnn_actor_critic import GNNActorCritic


def build_replay_plan(cache_dir: Path, samples: int = 16384) -> list[dict]:
    """All development teachers share one deterministic clean replay per topology."""
    if samples < 1:
        raise ValueError('Replay sample count must be positive.')
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        cfg = compose(config_name='config', overrides=['distillation=kl_paper_v4'])
        return [dict(topology=name, source=str(source),
                     output=str(cache_dir / 'clean-replay-v1' / f'samples_{samples}_seed_314159' / f'{name}.pt'),
                     samples=samples, seed=314159)
                for name, source in cfg.distillation.teachers.items()]


def build_commands(run_root: Path, cache_dir: Path, seeds: list[int],
                   stage: str = 'offline', exp_name: str = 'tube-v1',
                   arms: tuple[str, ...] = ('signed', 'membership'),
                   replay_samples: int = 16384) -> tuple[list[str], list[dict]]:
    """Preserve the routing protocol; the offline screen collects no SAC transitions."""
    if stage not in {'offline', 'online'}:
        raise ValueError('stage must be offline or online')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct seeds.')
    if not arms or len(set(arms)) != len(arms) or set(arms) - {'signed', 'membership', 'physics'}:
        raise ValueError('Provide distinct arms from signed, membership, physics.')
    common = [
        'sac_backend=gnn', 'platform=supercomputer', 'sim_backend=mjx',
        'cross_validation=random_5fold', 'cross_validation.held_out_group=fold_0',
        'distillation=kl_paper_v4', 'distillation.reconstruct_control_metadata=true',
        f'distillation.offline_only={str(stage == "offline").lower()}',
        f'run_root={run_root}', f'distillation.cache_dir={cache_dir}',
        f'wandb_dir={run_root}', 'enable_wandb=true', 'set_wandb_offline=true',
        'steps=133334', 'distillation.decay_steps=75000000',
        'distillation.pretrain_updates=10000',
        'message_attention=true', 'use_virtual_node=true',
        'checkpoint_freq=800000', 'checkpoint_keep_last=5',
        'eval_freq=200000', 'eval_episodes=5', 'save_video=false',
        'hydra.launcher.array_parallelism=3',
        f'++tube_experiment_base={exp_name}',
    ]
    if 'physics' in arms:
        replay_plan = build_replay_plan(cache_dir, replay_samples)
        mapping = ','.join(f"{task['topology']}:{task['output']}" for task in replay_plan)
        common.extend(['distillation.eval_freq=2000',
                       'domain_randomization_params.observation_noise.enabled=false',
                       'domain_randomization_params.action_noise.enabled=false',
                       'domain_randomization_params.length_scale.enabled=false',
                       'distillation.teachers={' + mapping + '}'])
    jobs = []
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        for seed in seeds:
            for arm in arms:
                cfg = parse_cfg(compose(config_name='config', overrides=[
                    *common, f'+tube_ablation={arm}', f'seed={seed}']))
                if cfg.steps != 2000010 or len(cfg.truss_topologies) != 15:
                    raise ValueError('Unexpected topology-scaled experiment budget.')
                # Shapes are fixed by the raw native/MJX graph contract.
                cfg.obs_dim, cfg.action_dim = cfg.node_feature_dim, 1
                model = GNNActorCritic(cfg)
                parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                jobs.append(dict(arm=arm, seed=seed, stage=stage,
                                 steps=0 if stage == 'offline' else cfg.steps,
                                 planned_online_steps=cfg.steps,
                                 trainable_parameters=parameters,
                                 wandb_dir=str(cfg.wandb_dir),
                                 enable_wandb=cfg.enable_wandb,
                                 set_wandb_offline=cfg.set_wandb_offline,
                                 training_topologies=cfg.truss_topologies,
                                 heldout_topologies=cfg.eval_extra_topologies,
                                 graph_features=cfg.graph_features,
                                 exp_name=cfg.exp_name))
    command = [sys.executable, str(ROOT / 'sac/train.py'), '-m', *common,
               'seed=' + ','.join(map(str, seeds)), '+tube_ablation=' + ','.join(arms)]
    return command, jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, help='Default: ~/nobackup/autodelete/GNN-SAC/runs/tube-<stage>-v1.')
    parser.add_argument('--cache-dir', type=Path, default=Path.home() / 'nobackup/autodelete/gnn-sac-tube-cache')
    parser.add_argument('--stage', choices=['offline', 'online'], default='offline')
    parser.add_argument('--exp-name', default='tube-v1', help='Experiment base name; appends the arm name.')
    parser.add_argument('--arms', nargs='+', choices=['signed', 'membership', 'physics'], default=['signed', 'membership'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    parser.add_argument('--replay-samples', type=int, default=16384,
                        help='Clean transitions per frozen teacher when physics is included.')
    parser.add_argument('--execute', action='store_true', help='Submit; default writes a manifest only.')
    args = parser.parse_args()
    if args.run_root is None:
        args.run_root = Path.home() / f'nobackup/autodelete/GNN-SAC/runs/tube-{args.stage}-v1'
    args.run_root = args.run_root.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    command, jobs = build_commands(args.run_root, args.cache_dir, args.seeds, args.stage, args.exp_name, tuple(args.arms), args.replay_samples)
    args.run_root.mkdir(parents=True, exist_ok=True)
    manifest = dict(command=command, jobs=jobs,
                    training_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    protocol='random_5fold/fold_0; fixed teachers, attention/global node, 10000 offline updates; KL decay 75M')
    (args.run_root / 'tube_experiment_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    replay_plan = build_replay_plan(args.cache_dir, args.replay_samples) if 'physics' in args.arms else []
    replay_manifest = args.run_root / 'clean_replay_manifest.json'
    if replay_plan:
        replay_manifest.write_text(json.dumps(replay_plan, indent=2) + '\n')
        manifest['clean_replay_manifest'] = str(replay_manifest)
        (args.run_root / 'tube_experiment_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(shlex.join(command), flush=True)
    if args.execute:
        # Slurm exports this environment to workers. Keep W&B's artifact staging
        # and cache on the same storage as its offline run records.
        environment = dict(os.environ)
        for name, directory in {
            'WANDB_CACHE_DIR': args.run_root / 'cache/wandb',
            'WANDB_DATA_DIR': args.run_root / 'cache/wandb-data',
            'TMPDIR': args.run_root / 'tmp',
        }.items():
            directory.mkdir(parents=True, exist_ok=True)
            environment[name] = str(directory)
        if replay_plan:
            preparation = [sys.executable, str(ROOT / 'scripts/prepare_tube_replay.py'),
                           '--manifest', str(replay_manifest)]
            print('Preparing shared clean replay on CPU; GPU jobs start after this array succeeds.', flush=True)
            subprocess.run([
                'sbatch', '--wait', '--parsable', '--account=nusey', '--qos=standby',
                '--time=04:00:00', '--cpus-per-task=4', '--mem=64G',
                f'--array=0-{len(replay_plan) - 1}%4', '--job-name=tube-clean-replay',
                '--chdir=' + str(ROOT),
                '--output=' + str(args.run_root / 'clean-replay-%A_%a.log'),
                '--wrap=' + shlex.join(preparation),
            ], cwd=ROOT, env=environment, check=True)
        result = subprocess.run(command, cwd=ROOT, env=environment)
        diagnostics = []
        for directory in sorted(args.run_root.glob('truss-graph/*/seed_*/*/checkpoints')):
            if directory.parents[2].name not in {job['exp_name'] for job in jobs}:
                continue
            if not (directory / 'distillation.pt').exists():
                continue
            output = directory.parent / 'tube_diagnostics.json'
            diagnostic = [sys.executable, str(ROOT / 'scripts/evaluate_routing_ablation.py'),
                          '--checkpoint-dir', str(directory), '--output', str(output)]
            submitted = subprocess.run([
                'sbatch', '--parsable', '--account=nusey', '--qos=standby',
                '--time=04:00:00', '--cpus-per-task=4', '--mem=64G',
                '--job-name=tube-diagnostics',
                '--output=' + str(directory.parent / 'diagnostics-%j.log'),
                '--wrap=' + shlex.join(diagnostic),
            ], text=True, capture_output=True)
            diagnostics.append(dict(checkpoint_dir=str(directory), output=str(output),
                                    returncode=submitted.returncode, stdout=submitted.stdout.strip(),
                                    stderr=submitted.stderr.strip()))
            print('DIAGNOSTICS', directory, submitted.returncode,
                  submitted.stdout, submitted.stderr, flush=True)
        manifest['diagnostic_submissions'] = diagnostics
        (args.run_root / 'tube_experiment_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        failed = not diagnostics or any(job['returncode'] for job in diagnostics)
        raise SystemExit(result.returncode or int(failed))


if __name__ == '__main__':
    main()
