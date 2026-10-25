"""Collect predeclared nominal-source fitting/validation/evaluation episodes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sac")]
from env.mujoco import make_env
from dynamics.dataset import SOURCE_TOPOLOGIES, file_sha256, validate_episode
from dynamics.planner import interpolate_knots
from dynamics.simulator import SimulatorBranch


def make_training_env(topology: str, settings=None):
    """Nominal abstract source env, physical-unit state, control-graph actions."""
    if topology not in SOURCE_TOPOLOGIES:
        raise ValueError(f"Refusing non-source topology {topology}")
    settings = settings or OmegaConf.load(ROOT / "config/dynamics_training_data.yaml")
    return make_env(OmegaConf.create({
        "task": f"truss-graph:{topology}", "obs": "state", "save_video": False,
        "visualize": False, "max_steps": 1000, "nsubsteps": int(settings.nsubsteps),
        "speed": float(settings.speed), "use_control_graph": True, "truss_realistic": False,
        "domain_randomization": False, "normalize_observations": False, "seed": 12000,
        "enable_wandb": False, "save_agent": False, "save_csv": False,
        "graph_features": {"edge_roles": True},
    }))


def graph_metadata(base, obs: dict) -> dict:
    """Export physical aliases and known route incidence, preserving action order."""
    physical_names = list(base.mj_model.node_names)
    graph_names = list(base.graph_node_names)
    controller = base.node_velocity_controller
    physical_by_graph = controller.control_graph.control_node_to_physical_node
    mapping = [physical_names.index(physical_by_graph[name]) for name in graph_names]
    seen, mask = set(), []
    for i in mapping:
        mask.append(i not in seen)
        seen.add(i)
    roles = {frozenset((edge.from_node, edge.to_node)): edge.type
             for edge in controller.control_graph.edges}
    edge_index = np.asarray(obs["edge_index"], dtype=int)
    edge_static = [[int(roles[frozenset((graph_names[a], graph_names[b]))] == "actuated"),
                    int(roles[frozenset((graph_names[a], graph_names[b]))] == "connector")]
                   for a, b in edge_index.T]
    positions = base.mj_model.get_node_position_matrix()
    graph_positions = positions[mapping]
    lengths = np.linalg.norm(graph_positions[edge_index[1]] - graph_positions[edge_index[0]], axis=-1)
    structural = []
    for a, b in base.mj_model.structural_edges:
        structural.append([physical_names.index(a) if isinstance(a, str) else int(a),
                           physical_names.index(b) if isinstance(b, str) else int(b)])
    return dict(
        topology=str(base.source_config.truss_topology), schema_version=1,
        state_units="metres, metres/second; xy centered on unique physical-node mean; absolute z",
        graph_node_names=graph_names, physical_node_names=physical_names,
        node_to_physical=mapping, physical_mask=mask,
        edge_index=edge_index.tolist(), edge_static=edge_static, initial_edge_length=lengths.tolist(),
        action_mask=obs["action_mask"].astype(bool).tolist(),
        action_units="normalized node command; multiply by speed, route via incidence, clip ctrlrange",
        routing_incidence=controller.incidence_matrix.tolist(),
        ctrlrange=base.mj_model.model.actuator_ctrlrange[controller.actuator_ids].tolist(),
        actuator_names=controller.edge_names, speed=float(base.config.speed),
        initial_bbox=base.mj_model.initial_bounding_box_dimensions.tolist(),
        initial_diagonal=float(base.mj_model.initial_bounding_box_diagonal),
        initial_rigidity_reference=float(base._initial_critical_eig),
        critical_eig_threshold=float(base.config.critical_eig_threshold),
        structural_edges=structural, axis_indices=list(base.mj_model.axis_indices),
        wcrm=bool(base.mj_model.wcrm), source_config=OmegaConf.to_container(base.source_config),
        simulator_xml_sha256=hashlib.sha256(str(base.mj_model.xml).encode()).hexdigest(),
    )


def observable_state(base, mapping: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read clean physical state, duplicate aliases only after physical centering."""
    positions = base.mj_model.get_node_position_matrix().copy().astype(np.float64)
    velocities = base.mj_model.get_node_linear_velocity_matrix().copy().astype(np.float64)
    xy = positions[:, :2].mean(0)
    positions[:, :2] -= xy
    return np.concatenate([positions, velocities], axis=1)[mapping], xy


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def collect(output: Path, settings, topologies: list[str], *, resume: bool = False) -> dict:
    """Collect fixed batches; checkpoint progress after each independent episode."""
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(settings=OmegaConf.to_container(settings), requested_topologies=topologies,
                    fitting_capacity=int(settings.capacity), source_balance="Equal episodes/steps requested per topology; early failures retained",
                    action_regimes="Smooth clipped normal node commands; amplitudes 0.15, 0.5, 1.0 cycling by episode; knots every5 steps",
                    stop_rule="Exactly declared batches; terminate episodes on true termination/truncation; do not extend based on outcomes",
                    fallback_reason="Legacy replay has noisy observations without clean counterparts and no trajectory provenance; native nominal labels and independent episodes required",
                    evaluation_exclusions=[8100,8101,8102,8103],
                    collection_cost_target_seconds=1800,
                    model_update_contract="Replay-based updates specified separately by model training protocol; no updates during collection",
                    timestamp_utc=datetime.now(timezone.utc).isoformat())
    protocol_path = output / "collection-protocol.json"
    if protocol_path.exists():
        saved_protocol = json.loads(protocol_path.read_text())
        if saved_protocol["settings"] != protocol["settings"]:
            raise ValueError("Cannot resume collection with changed settings")
    else:
        write_json(protocol_path, protocol)
    manifest_path = output / "manifest.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = dict(schema_version=1, role="source_training_validation_and_evaluation_separated",
                        collection_protocol="collection-protocol.json", episodes=[], topology_reports={},
                        revision=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
                        dependencies={k:importlib.metadata.version(k) for k in ['numpy','mujoco','mujoco-truss-gen']},
                        limitations=["Smooth random only; not teacher or actual planner-proposal coverage.",
                                     "No held-out access. Physical dynamics can remain partly observed.",
                                     "Fresh source evidence is development evidence; not a transfer result."])
    start = time.perf_counter()
    for topology in topologies:
        if topology not in SOURCE_TOPOLOGIES:
            raise ValueError("Source topology whitelist violated")
        topology_index = SOURCE_TOPOLOGIES.index(topology)
        env, planning = make_training_env(topology, settings), make_training_env(topology, settings)
        topo_start=time.perf_counter()
        previous_count=sum(r["topology"]==topology for r in manifest["episodes"])
        try:
            env.reset(); planning.reset()
            base=env.unwrapped
            branch=SimulatorBranch(env,planning)
            for role, role_settings in settings.roles.items():
                for episode_id in range(int(role_settings.episodes)):
                    key=(topology,role,episode_id)
                    if any((r['topology'],r['role'],r['episode_id']) == key for r in manifest['episodes']):
                        continue
                    seed=int(role_settings.seed_base)+topology_index*100+episode_id
                    if seed in (8100,8101,8102,8103):
                        raise ValueError('Reserved probe seed')
                    env.reset(); obs,_=base.reset(seed=seed)
                    metadata=graph_metadata(base,obs)
                    topo_dir=output/topology; topo_dir.mkdir(exist_ok=True)
                    metadata_name=f'{topology}/metadata.json'
                    metadata_path=output/metadata_name
                    if not metadata_path.exists():
                        write_json(metadata_path,metadata)
                        (topo_dir/'model.xml').write_text(str(base.mj_model.xml))
                    mapping=np.asarray(metadata['node_to_physical'],dtype=int)
                    mask=np.asarray(metadata['action_mask'],dtype=bool)[:,None]
                    rng=np.random.default_rng(seed+1000000)
                    amplitude=float(settings.amplitudes[episode_id%len(settings.amplitudes)])
                    sequence=interpolate_knots(rng.normal(0,amplitude,(max(2,int(settings.steps)//int(settings.knots_every)),len(mask))),int(settings.steps))
                    sequence=np.clip(sequence,-1,1)[:,:,None]*mask
                    rows=[]; snapshots=[]; snapshot_indices=[]
                    episode_start=time.perf_counter()
                    for timestep,action in enumerate(sequence):
                        if role in ('validation','evaluation') and timestep%int(settings.snapshot_every)==0:
                            snapshots.append(branch.capture());snapshot_indices.append(timestep)
                        state,xy=observable_state(base,mapping)
                        activation=base.mj_model.data.act.copy()
                        pre_rigidity=float(obs['rigidity'][0])
                        pre_time=float(base.mj_model.data.time)
                        obs,reward,done,info=env.step(action)
                        nxt,nxy=observable_state(base,mapping)
                        rows.append(dict(state=state,next_state=nxt,action=action.astype(np.float32),
                            progress=nxy-xy,global_xy=xy,next_global_xy=nxy,
                            dt=float(base.mj_model.data.time)-pre_time,terminated=bool(info['terminated']),
                            truncated=bool(info['truncated']),timestep=timestep,
                            rigidity=pre_rigidity,
                            next_rigidity=float(obs['rigidity'][0]),reward=float(reward),
                            activation=activation,next_activation=base.mj_model.data.act.copy(),
                            executed_control=base.node_velocity_controller.latest_edge_commands.copy(),
                            substeps=int(info['substeps_executed'])))
                        if done: break
                    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
                    # Current rigidity is read from the pre-state, rather than a future info cache.
                    # Saved next_rigidity is diagnostic only and never model input.
                    filename=f'{topology}/{role}-{episode_id:03d}.npz'
                    np.savez_compressed(output/filename,**arrays)
                    check=validate_episode({**metadata,**arrays})
                    record=dict(topology=topology,role=role,episode_id=episode_id,seed=seed,
                        file=filename,metadata=metadata_name,sha256=file_sha256(output/filename),
                        metadata_sha256=file_sha256(metadata_path),transitions=len(rows),
                        wall_seconds=time.perf_counter()-episode_start,amplitude=amplitude,
                        collapsed=bool(rows[-1]['terminated']),validation=check)
                    if snapshots:
                        snapshot_name=filename.replace('.npz','-snapshots.pkl.gz')
                        with gzip.open(output/snapshot_name,'wb') as stream:
                            pickle.dump(snapshots,stream,protocol=pickle.HIGHEST_PROTOCOL)
                        record.update(snapshots=snapshot_name,snapshot_timesteps=snapshot_indices,
                                      snapshots_sha256=file_sha256(output/snapshot_name))
                    manifest['episodes'].append(record)
                    manifest['wall_seconds_this_invocation']=time.perf_counter()-start
                    write_json(manifest_path,manifest)
                    print(json.dumps({k:record[k] for k in ['topology','role','episode_id','transitions','wall_seconds','collapsed']}),flush=True)
            current_count=sum(r['topology']==topology for r in manifest['episodes'])
            if current_count>previous_count:
                previous_seconds=manifest['topology_reports'].get(topology,{}).get('wall_seconds',0.0)
                manifest['topology_reports'][topology]=dict(wall_seconds=previous_seconds+time.perf_counter()-topo_start,
                    episodes=current_count,
                    transitions=sum(r['transitions'] for r in manifest['episodes'] if r['topology']==topology))
            write_json(manifest_path,manifest)
        finally:
            env.close();planning.close()
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/dynamics-pilot/source-training-2026-10-23')
    parser.add_argument('--config',type=Path,default=ROOT/'config/dynamics_training_data.yaml')
    parser.add_argument('--topologies',nargs='+',choices=SOURCE_TOPOLOGIES)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    settings=OmegaConf.load(args.config)
    collect(args.output,settings,args.topologies or list(settings.topologies),resume=args.resume)
