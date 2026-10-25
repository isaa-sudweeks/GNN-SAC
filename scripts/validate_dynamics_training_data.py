"""Audit source dataset roles, graph aliases and selected full-snapshot transitions."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from omegaconf import OmegaConf

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'sac')]
from dynamics.dataset import file_sha256,load_dataset
from dynamics.simulator import SimulatorBranch
from scripts.collect_dynamics_training_data import make_training_env,observable_state


def validate(root: Path, output: Path) -> dict:
    """Check every label file; replay two saved transitions per source topology."""
    started=time.perf_counter()
    manifest=json.loads((root/'manifest.json').read_text())
    episodes=load_dataset(root)
    settings=OmegaConf.create(json.loads((root/manifest['collection_protocol']).read_text())['settings'])
    seeds=[e['seed'] for e in episodes]
    if len(seeds)!=len(set(seeds)) or set(seeds)&{8100,8101,8102,8103}:
        raise ValueError('Duplicate or reserved seed')
    summaries={};snapshot_checks=[];initial_state_counts={}
    max_routed_command_error=0.0
    for e in episodes:
        routed=(np.asarray(e['routing_incidence']) @ (e['action'][:,:,0]*float(e['speed'])).T).T
        expected=np.clip(routed,np.asarray(e['ctrlrange'])[:,0],np.asarray(e['ctrlrange'])[:,1])
        error=float(np.abs(expected-e['executed_control']).max())
        max_routed_command_error=max(error,max_routed_command_error)
        if error>1e-7:
            raise ValueError(f'Executed command label mismatch {e["topology"]}: {error}')
    for topology in sorted({e['topology'] for e in episodes}):
        topo_episodes=[e for e in episodes if e['topology']==topology]
        initial_state_counts[topology]=len({hashlib.sha256(np.r_[e['state'][0].ravel(),e['global_xy'][0]].tobytes()).hexdigest() for e in topo_episodes})
        summaries[topology]={}
        for role in ['fitting','validation','evaluation']:
            part=[e for e in topo_episodes if e['role']==role]
            summaries[topology][role]=dict(episodes=len(part),transitions=sum(len(e['state']) for e in part),
                collapsed=sum(bool(e['terminated'][-1]) for e in part),
                shortened_dt_rows=sum(int(np.sum(e['dt']<0.2-1e-8)) for e in part),
                min_dt=min(float(e['dt'].min()) for e in part),
                net_forward_displacement=[float(e['progress'][:,0].sum()/e['initial_diagonal']) for e in part])
        record=next(r for r in manifest['episodes'] if r['topology']==topology and r['role']=='validation')
        e=next(e for e in topo_episodes if e['role']=='validation' and e['episode_id']==record['episode_id'])
        snapshots_path=root/record['snapshots']
        if file_sha256(snapshots_path)!=record['snapshots_sha256']:
            raise ValueError('Snapshot hash mismatch')
        with gzip.open(snapshots_path,'rb') as stream:
            snapshots=pickle.load(stream)  # trusted local artifacts only
        live,planning=make_training_env(topology,settings),make_training_env(topology,settings)
        try:
            live.reset();planning.reset();branch=SimulatorBranch(live,planning)
            for saved_index in sorted(set([0,len(snapshots)-1])):
                timestep=record['snapshot_timesteps'][saved_index]
                branch.restore(snapshots[saved_index])
                pre,xy=observable_state(planning.unwrapped,np.asarray(e['node_to_physical']))
                obs,reward,done,info=planning.step(e['action'][timestep])
                nxt,nxy=observable_state(planning.unwrapped,np.asarray(e['node_to_physical']))
                row=dict(topology=topology,episode_id=record['episode_id'],timestep=timestep,
                    pre_state_max_error=float(np.abs(pre-e['state'][timestep]).max()),
                    next_state_max_error=float(np.abs(nxt-e['next_state'][timestep]).max()),
                    progress_max_error=float(np.abs((nxy-xy)-e['progress'][timestep]).max()),
                    raw_reward_error=abs(float(reward)-float(e['reward'][timestep])),
                    termination_matches=bool(info['terminated'])==bool(e['terminated'][timestep]),
                    truncation_matches=bool(info['truncated'])==bool(e['truncated'][timestep]))
                if any(row[k]>1e-12 for k in ['pre_state_max_error','next_state_max_error','progress_max_error','raw_reward_error']) or not row['termination_matches'] or not row['truncation_matches']:
                    raise ValueError(f'Restored transition mismatch {row}')
                snapshot_checks.append(row)
        finally:
            live.close();planning.close()
    report=dict(dataset=str(root.resolve()),manifest_sha256=file_sha256(root/'manifest.json'),
        episodes=len(episodes),transitions=sum(len(e['state']) for e in episodes),
        all_episode_hashes_finite_adjacency_and_alias_checks=True,all_seeds_unique_and_disjoint=True,
        max_routed_command_label_error=max_routed_command_error,
        summary=summaries,distinct_initial_state_counts=initial_state_counts,snapshot_checks=snapshot_checks,wall_seconds=time.perf_counter()-started,
        limitations=['Snapshot checks are selected transitions, not exhaustive resimulation.',
                     'Smooth-random action distribution only; no teacher/planner coverage claim.',
                     'No held-out results; independent source episodes are still development evidence.',
                     'Early-collapse rows have shortened physical duration; a fixed-step predictor does not establish collapse timing.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'outputs/dynamics-pilot/source-training-2026-10-23')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/dynamics-pilot/source-training-2026-10-23/data-validation.json')
    args=parser.parse_args()
    print(json.dumps(validate(args.dataset,args.output),indent=2))
