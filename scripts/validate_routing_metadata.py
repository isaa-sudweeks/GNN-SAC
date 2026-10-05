#!/usr/bin/env python3
"""Check every development topology and optionally validate saved teacher contracts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from torch_geometric.data import Data

ROOT=Path(__file__).resolve().parents[1]
for path in (ROOT,ROOT/'sac'):sys.path.insert(0,str(path))
from common.parser import parse_cfg
from common.distillation import reconstruct_control_metadata, replay_contract_observation, replay_observations
from env.mujoco_gen.topology_envs import MujocoPresetGraphEnv


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-contracts',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();results=[]
    with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
        raw=compose(config_name='config',overrides=['sac_backend=gnn','cross_validation=random_5fold',
                    'cross_validation.held_out_group=fold_0','distillation=kl_paper_v4',
                    'device=cpu',f'work_dir={args.output.parent}', 'graph_features.edge_roles=true','graph_features.edge_direction=true'])
        cfg=parse_cfg(raw)
    for topology,path in cfg.distillation['teachers'].items():
        if topology in cfg.cross_validation['final_test']:raise ValueError('Final-test leakage.')
        local=SimpleNamespace(**vars(cfg));local.truss_topology=topology;local.truss_topologies=None;local.domain_randomization=False
        env=MujocoPresetGraphEnv(local)
        try:
            obs,_=env.reset(seed=0)
            row=dict(topology=topology,nodes=len(obs['x']),signed_messages=int((obs['edge_direction']!=0).sum()))
        finally:env.close()
        if args.teacher_contracts:
            state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            replay=state['buffer']
            if 'buffers' in replay:
                matches=[v for k,v in replay['buffers'].items() if k.split(':')[-1]==topology]
                if not matches and len(replay['buffers'])==1:matches=list(replay['buffers'].values())
                if len(matches)!=1:raise ValueError('Ambiguous teacher replay.')
                replay=matches[0]
            graph=next(replay_observations(replay));graph=replay_contract_observation(replay,graph)
            metadata=reconstruct_control_metadata(SimpleNamespace(**state['config']),cfg,graph)
            row['teacher_contract']='validated';row['teacher']=str(path)
            del state,replay,graph,metadata
        results.append(row);print(json.dumps(row),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
