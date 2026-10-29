#!/usr/bin/env python3
"""Read-only, fixed-seed development diagnostics for routing-ablation checkpoints."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'sac'):
    sys.path.insert(0,str(path))
from common.distillation import attach_control_metadata, gaussian_forward_kl, reconstruct_control_metadata, replay_contract_observation
from common.gnn_actor_critic import GNNActorCritic
from common.graph_transforms import graph_feature_flags, policy_action_mask, prepare_graph
from env.mujoco_gen.topology_envs import MujocoPresetGraphEnv


def replay_samples(replay: dict, count: int, seed: int) -> list[Data]:
    """Uniform valid replay rows, selected without materializing the full replay."""
    size, capacity, end = (int(replay[k]) for k in ('size','capacity','idx'))
    if not 0 < size <= capacity:
        raise ValueError('Empty or invalid diagnostic teacher replay.')
    generator=torch.Generator().manual_seed(seed)
    offsets=torch.randint(size,(count,),generator=generator).tolist()
    indices=[((end if size==capacity else 0)+i)%capacity for i in offsets]
    if int(replay.get('format_version',0)) not in (3,4):
        return [replay['obs'][i].clone() for i in indices]
    fields=replay['replay_buffer']['_storage']['_storage'];static=replay['static'];result=[]
    for i in indices:
        graph=Data(x=fields['obs_x'][i].clone(),edge_index=static['edge_index'])
        mask=fields['obs_action_mask'][i] if 'obs_action_mask' in fields else static['action_mask']
        if mask is not None:graph.action_mask=mask.clone()
        for key in ('edge_role','edge_direction'):
            if static.get(key) is not None:graph[key]=static[key]
        if static['has_rigidity']:graph.rigidity=fields['obs_rigidity'][i].clone()
        result.append(graph)
    return result


def prepared(config, graph):
    return prepare_graph(graph,use_virtual_node=bool(config.use_virtual_node),**graph_feature_flags(config))


def select_checkpoints(directory: Path) -> list[tuple[str,Path]]:
    result=[]
    offline=directory/'distillation.pt'
    if offline.exists():result.append(('offline',offline))
    numbered=sorted((int(p.stem.removeprefix('step_')),p) for p in directory.glob('step_*.pt') if not p.name.endswith('.agent.pt'))
    for label,target in [('800k',800000),('final',2000010)]:
        candidates=[(step,p) for step,p in numbered if step>=target]
        if candidates:result.append((label,candidates[0][1]))
    return result


@torch.no_grad()
def evaluate_checkpoint(path: Path, samples: int=256, episodes: int=5) -> dict:
    """No gradients, optimizer updates, checkpoint selection, or final-test access."""
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    config=deepcopy(state['config']);config['device']='cpu';cfg=SimpleNamespace(**config)
    student=GNNActorCritic(cfg);student.load_state_dict(state['agent']['model']);student.eval()
    step=int(state.get('trainer',{}).get('step',0))
    final_test=set(config.get('cross_validation',{}).get('final_test',[]))
    heldout=list(config.get('eval_extra_topologies') or [])
    if (not heldout or set(heldout)&final_test
            or set(heldout)&set(config.get('truss_topologies') or [])):
        raise ValueError('Diagnostics require development holdouts and must exclude final_test.')
    del state
    result=dict(checkpoint=str(path),step=step,seed=cfg.seed,topologies={},
                replay_sampling_seed=314159,reset_seeds=list(range(1000,1000+episodes)),
                rollout_protocol='Native MuJoCo; configured reset randomization; clean observations/actions (no wrapper noise).')
    for topology in heldout:
        teacher_path=Path(config['distillation']['teachers'][topology])
        teacher_state=torch.load(teacher_path,map_location='cpu',weights_only=False,mmap=True)
        teacher_cfg=SimpleNamespace(**teacher_state['config'])
        teacher=GNNActorCritic(teacher_cfg);teacher.load_state_dict(teacher_state['agent']['model']);teacher.eval()
        replay=teacher_state['buffer']
        if 'buffers' in replay:
            candidates=[v for k,v in replay['buffers'].items() if k.split(':')[-1]==topology]
            if not candidates and len(replay['buffers'])==1:candidates=list(replay['buffers'].values())
            if len(candidates)!=1:raise ValueError('Ambiguous held-out teacher replay.')
            replay=candidates[0]
        graphs=replay_samples(replay,samples,314159)
        base=replay_contract_observation(replay,graphs[0])
        metadata=reconstruct_control_metadata(teacher_cfg,cfg,base)
        enriched=[attach_control_metadata(g,metadata) for g in graphs]
        tb=Batch.from_data_list([prepared(teacher_cfg,g) for g in enriched])
        sb=Batch.from_data_list([prepared(cfg,g) for g in enriched])
        tm,tl=teacher.policy_distribution(tb);sm,sl=student.policy_distribution(sb)
        metrics=dict(teacher_checkpoint=str(teacher_path),teacher_replay_samples=samples,
                     teacher_kl=float(gaussian_forward_kl(tm,tl,sm,sl).mean()),
                     teacher_tanh_mean_mse=float((tm.tanh()-sm.tanh()).square().mean()))
        del teacher_state,replay,teacher,tb,sb,graphs,enriched
        env_cfg=deepcopy(config);env_cfg.update(truss_topology=topology,truss_topologies=None,
                                              mujoco_backend='mujoco',sim_backend='mujoco')
        env=MujocoPresetGraphEnv(env_cfg);rollouts=[]
        try:
            for reset_seed in range(1000,1000+episodes):
                raw,_=env.reset(seed=reset_seed);distance=0.;length=0;terminated=truncated=False
                while not (terminated or truncated):
                    graph=Data(**{k:torch.as_tensor(v) for k,v in raw.items()})
                    observation=prepared(cfg,graph)
                    active=student.pi_mean(observation)
                    actions=torch.zeros(graph.num_nodes,config['action_dim'])
                    actions[policy_action_mask(graph)]=active
                    raw,_,terminated,truncated,info=env.step(actions.numpy())
                    distance+=float(info.get('com_delta_x',0));length+=1
                rollouts.append(dict(seed=reset_seed,distance=distance,length=length,
                                     terminated=bool(terminated),truncated=bool(truncated)))
        finally:env.close()
        metrics.update(rollouts=rollouts,episode_distance=float(np.mean([x['distance'] for x in rollouts])),
                       episode_length=float(np.mean([x['length'] for x in rollouts])))
        result['topologies'][topology]=metrics
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--episodes',type=int,default=5)
    args=parser.parse_args();torch.set_num_threads(4)
    results={}
    for stage,path in select_checkpoints(args.checkpoint_dir):
        print('Evaluating',stage,path,flush=True)
        results[stage]=evaluate_checkpoint(path,args.samples,args.episodes)
        args.output.write_text(json.dumps(results,indent=2)+'\n')
    if not results:raise ValueError('No diagnostic checkpoints available.')


if __name__=='__main__':main()
