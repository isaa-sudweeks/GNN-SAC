#!/usr/bin/env python3
"""Launch the matched signed versus tube-membership screen on ORC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sac'))
from common.parser import parse_cfg
from common.gnn_actor_critic import GNNActorCritic


def build_commands(run_root: Path, cache_dir: Path, seeds: list[int],
                   stage: str = 'offline') -> tuple[list[str], list[dict]]:
    """Preserve the routing protocol; the offline screen collects no SAC transitions."""
    if stage not in {'offline', 'online'}:
        raise ValueError('stage must be offline or online')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide distinct seeds.')
    common = [
        'sac_backend=gnn', 'platform=supercomputer', 'sim_backend=mjx',
        'cross_validation=random_5fold', 'cross_validation.held_out_group=fold_0',
        'distillation=kl_paper_v4', 'distillation.reconstruct_control_metadata=true',
        f'distillation.offline_only={str(stage == "offline").lower()}',
        f'run_root={run_root}', f'distillation.cache_dir={cache_dir}',
        'steps=133334', 'distillation.decay_steps=75000000',
        'distillation.pretrain_updates=10000',
        'message_attention=true', 'use_virtual_node=true',
        'checkpoint_freq=800000', 'checkpoint_keep_last=5',
        'eval_freq=200000', 'eval_episodes=5', 'save_video=false',
        'hydra.launcher.array_parallelism=3',
    ]
    jobs = []
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        for seed in seeds:
            for arm in ('signed', 'membership'):
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
                                 training_topologies=cfg.truss_topologies,
                                 heldout_topologies=cfg.eval_extra_topologies,
                                 graph_features=cfg.graph_features,
                                 exp_name=cfg.exp_name))
    command = [sys.executable, str(ROOT / 'sac/train.py'), '-m', *common,
               'seed=' + ','.join(map(str, seeds)), '+tube_ablation=signed,membership']
    return command, jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['offline', 'online'], default='offline')
    parser.add_argument('--execute', action='store_true', help='Submit; default writes a manifest only.')
    args = parser.parse_args()
    command, jobs = build_commands(args.run_root, args.cache_dir, [1, 2, 3], args.stage)
    args.run_root.mkdir(parents=True, exist_ok=True)
    manifest = dict(command=command, jobs=jobs,
                    training_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    protocol='random_5fold/fold_0; fixed teachers, attention/global node, 10000 offline updates; KL decay 75M')
    (args.run_root / 'tube_experiment_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(shlex.join(command), flush=True)
    if args.execute:
        result = subprocess.run(command, cwd=ROOT)
        diagnostics = []
        for directory in sorted(args.run_root.glob('truss-graph/tube-v1-*/seed_*/*/checkpoints')):
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
