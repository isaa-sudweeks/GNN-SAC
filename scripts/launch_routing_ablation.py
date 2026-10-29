#!/usr/bin/env python3
"""Compose a controlled 3-arm x 3-seed development-fold experiment on ORC."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'sac'))
from common.parser import parse_cfg


def build_commands(run_root: Path, cache_dir: Path, seeds: list[int]) -> tuple[list[str], list[dict]]:
    """Use 15 training topologies and an explicit 75M-transition KL decay."""
    common = [
        'sac_backend=gnn', 'platform=supercomputer', 'sim_backend=mjx',
        'cross_validation=random_5fold', 'cross_validation.held_out_group=fold_0',
        'distillation=kl_paper_v4', 'distillation.reconstruct_control_metadata=true',
        f'run_root={run_root}', f'distillation.cache_dir={cache_dir}',
        'steps=133334',  # parser multiplies by 15 -> 2,000,010 total transitions.
        'distillation.decay_steps=75000000', 'distillation.pretrain_updates=10000',
        'message_attention=true', 'use_virtual_node=true',
        'checkpoint_freq=800000', 'checkpoint_keep_last=5',
        'eval_freq=200000', 'eval_episodes=5', 'save_video=false',
        'hydra.launcher.array_parallelism=3',
    ]
    jobs = []
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        for seed in seeds:
            for arm in ('baseline', 'edge_types', 'signed'):
                raw = compose(config_name='config', overrides=[*common, f'+routing_ablation={arm}', f'seed={seed}'])
                cfg = parse_cfg(raw)
                if cfg.steps != 2000010 or len(cfg.truss_topologies) != 15:
                    raise ValueError('Unexpected topology-scaled experiment budget.')
                if cfg.distillation['decay_steps'] != 75000000:
                    raise ValueError('KL schedule changed.')
                jobs.append(dict(arm=arm, seed=seed, steps=cfg.steps,
                                 training_topologies=cfg.truss_topologies,
                                 heldout_topologies=cfg.eval_extra_topologies,
                                 graph_features=cfg.graph_features,
                                 exp_name=cfg.exp_name))
    command = [sys.executable, str(ROOT / 'sac/train.py'), '-m', *common,
               'seed='+','.join(map(str,seeds)), '+routing_ablation=baseline,edge_types,signed']
    return command, jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Submit; default only writes a manifest.')
    args = parser.parse_args()
    command, jobs = build_commands(args.run_root, args.cache_dir, [1,2,3])
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root/'experiment_manifest.json').write_text(json.dumps(dict(
        command=command, jobs=jobs, training_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol='random_5fold/fold_0; matched teachers, 10000 offline updates, 2000010 total transitions; KL decay 75M',
    ), indent=2)+'\n')
    print(shlex.join(command), flush=True)
    if args.execute:
        result = subprocess.run(command, cwd=ROOT)
        # Run diagnostics on compute nodes, including available checkpoints from failed jobs.
        for directory in sorted((args.run_root/'truss-graph').glob('routing-v1-*/seed_*/*/checkpoints')):
            output = directory.parent/'routing_diagnostics.json'
            diagnostic = [sys.executable, str(ROOT/'scripts/evaluate_routing_ablation.py'),
                          '--checkpoint-dir', str(directory), '--output', str(output)]
            submitted = subprocess.run([
                'sbatch', '--parsable', '--account=nusey', '--qos=standby', '--time=04:00:00',
                '--cpus-per-task=4', '--mem=64G', '--job-name=routing-diagnostics',
                '--output='+str(directory.parent/'diagnostics-%j.log'),
                '--wrap='+shlex.join(diagnostic)], text=True, capture_output=True)
            print('DIAGNOSTICS',directory,submitted.returncode,submitted.stdout,submitted.stderr,flush=True)
        raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
