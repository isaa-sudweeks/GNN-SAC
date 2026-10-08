#!/usr/bin/env python3
"""GPU-node canary: tiny native fixture teachers, then signed MJX distillation."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

import torch
from hydra import compose, initialize_config_dir

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'sac'):sys.path.insert(0,str(path))
from common.parser import parse_cfg
from common.gnn_buffer import GNNBuffer
from env import make_env
from gnn_sac import GNNSAC
from trainer.base import Trainer


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--tube-nodes', action='store_true', help='Exercise tube membership student with legacy teachers.')
    parser.add_argument('--cpu', action='store_true', help='Run native two-topology canary locally.')
    args=parser.parse_args();args.run_root.mkdir(parents=True,exist_ok=True)
    mapping={}
    for topology in ('tetrahedron','octahedron'):
        with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
            cfg=parse_cfg(compose(config_name='config',overrides=[
                'sac_backend=gnn','device=cpu','enable_wandb=false','save_video=false',
                'domain_randomization=false','max_steps=2','nsubsteps=1',
                f'truss_topology={topology}',f'work_dir={args.run_root}/fixture-{topology}',
                'buffer_size=8','batch_size=2','mpl_dims=[8]','message_hidden_dims=[8]',
                'head_hidden_dims=[8]','log_std_min=-2','log_std_max=0']))
        env=make_env(cfg)
        try:
            agent=GNNSAC(cfg);buffer=GNNBuffer(cfg);obs=env.reset()
            episode=[dict(obs=obs,action=torch.zeros(obs.num_nodes,1),reward=torch.tensor(0.),terminated=torch.tensor(False))]
            for _ in range(2):
                action=agent.act(obs);obs,reward,_,info=env.step(action)
                episode.append(dict(obs=obs,action=action,reward=reward,terminated=info['terminated']))
            buffer.add(episode)
            # Distillation only needs these three full-trainer sections.
            path=args.run_root/f'{topology}.pt'
            torch.save(dict(config=vars(cfg),agent=agent.training_state_dict(),buffer=buffer.state_dict()),path)
            mapping[topology]=str(path)
        finally:env.close()
    teacher_override='++distillation.teachers={'+','.join(f'{key}:{value}' for key,value in mapping.items())+'}'
    command=[sys.executable,str(ROOT/'sac/train.py'),'sac_backend=gnn','platform=supercomputer',
             'sim_backend=mjx','mjx_impl=warp','num_envs=4','truss_topologies=[tetrahedron,octahedron]',
             'distillation=kl','distillation.reconstruct_control_metadata=true',teacher_override,
             'distillation.pretrain_updates=2','distillation.batch_size=2','distillation.shard_size=2',
             'distillation.checkpoint_freq=0','steps=16','batch_size=2','buffer_size=32',
             'seed_steps=1','pretrain_steps=1','max_steps=2','nsubsteps=1','eval_episodes=1','eval_freq=100',
             'domain_randomization=false','enable_wandb=false','save_video=false','save_agent=false',
             'checkpoint_freq=0','save_csv=false','resume_from_checkpoint=null',
             'graph_features.node_roles=true','graph_features.edge_roles=true',
             'graph_features.edge_distance=true','graph_features.edge_direction=true',
             'mpl_dims=[8]','message_hidden_dims=[8]','head_hidden_dims=[8]',
             'log_std_min=-2','log_std_max=0',f'run_root={args.run_root}',
             f'work_dir={args.run_root}/student','exp_name=routing-canary']
    if args.tube_nodes:
        command.append('graph_features.tube_nodes=true')
    if args.cpu:
        command = [item for item in command if not item.startswith(('platform=', 'sim_backend=', 'mjx_impl=', 'num_envs='))]
        command = ['+' + item if item.startswith('run_root=') else item for item in command]
        command.extend(['device=cpu', 'sim_backend=mujoco', 'num_envs=1'])
    subprocess.run(command,cwd=ROOT,check=True)
    (args.run_root/'PASSED').write_text(f'Signed-routing tube_nodes={args.tube_nodes} cpu={args.cpu} offline + online canary completed.\n')


if __name__=='__main__':main()
