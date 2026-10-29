#!/usr/bin/env python3
"""Matched separate/shared distillation and fresh-data teacher rollout diagnostic."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

ROOT = Path(os.environ.get('GNN_SAC_REPO', Path(__file__).resolve().parents[1]))
for path in (ROOT, ROOT / 'sac', ROOT / 'scripts'):
    sys.path.insert(0, str(path))

from common.distillation import Distillation, attach_control_metadata, gaussian_forward_kl, reconstruct_control_metadata, replay_contract_observation
from common.gnn_actor_critic import GNNActorCritic
from common.gnn_buffer import GNNBuffer
from common.graph_transforms import graph_feature_flags, policy_action_mask, prepare_graph
from common.logger import Logger
from common.parser import cfg_to_dataclass, parse_cfg
from env import make_env
from env.mujoco_gen.topology_envs import MujocoPresetGraphEnv
from evaluate_routing_ablation import replay_samples
from gnn_sac import GNNSAC
from prepare_tube_replay import prepare_replay


def load(path: str | Path) -> dict:
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def prepared(cfg: object, graph: Data) -> Data:
    return prepare_graph(graph, use_virtual_node=bool(cfg.use_virtual_node), **graph_feature_flags(cfg))


def rollout(model: GNNActorCritic, policy_cfg: object, environment_cfg: dict,
            seeds: list[int]) -> list[dict]:
    """Identical teacher-derived environments and reset seeds for every policy."""
    settings = deepcopy(environment_cfg)
    settings.update(device='cpu', mujoco_backend='mujoco', sim_backend='mujoco', truss_topologies=None)
    settings['graph_features'] = dict(node_roles=True, edge_roles=True, edge_direction=True)
    env = MujocoPresetGraphEnv(settings)
    rows = []
    try:
        for seed in seeds:
            raw, _ = env.reset(seed=seed)
            distance, length, done = 0., 0, False
            while not done:
                graph = Data(**{key: torch.as_tensor(value) for key, value in raw.items()})
                with torch.no_grad():
                    active = model.pi_mean(prepared(policy_cfg, graph))
                action = torch.zeros(graph.num_nodes, policy_cfg.action_dim)
                action[policy_action_mask(graph)] = active
                raw, _, terminated, truncated, info = env.step(action.numpy())
                done = terminated or truncated
                distance += float(info.get('com_delta_x', 0))
                length += 1
            rows.append(dict(seed=seed, distance=distance, length=length,
                             terminated=bool(terminated), truncated=bool(truncated)))
    finally:
        env.close()
    return rows


def prepare_validation(manifest: dict, index: int) -> None:
    topology = manifest['topologies'][index]
    output = Path(manifest['validation_replay'][topology])
    prepare_replay(Path(manifest['training_replay'][topology]), output,
                   manifest['validation_samples'], manifest['validation_seed'])
    state = load(output)
    cfg = cfg_to_dataclass(OmegaConf.create(state['config']))
    teacher = GNNActorCritic(cfg).eval()
    teacher.load_state_dict(state['agent']['model'])
    rows = rollout(teacher, cfg, state['config'], manifest['reset_seeds'])
    output.with_suffix('.teacher.json').write_text(json.dumps(dict(topology=topology, rollouts=rows), indent=2))


def train_student(manifest: dict, index: int) -> None:
    job = manifest['jobs'][index]
    directory = Path(job['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'complete.json').exists():
        raise ValueError(f'Refusing to overwrite completed diagnostic {directory}')
    mapping = ','.join(f'{name}:{manifest["training_replay"][name]}' for name in job['topologies'])
    overrides = [
        'sac_backend=gnn', 'device=cpu', 'sim_backend=mujoco', 'num_envs=1',
        'cross_validation=disabled', 'distillation=kl', '+tube_ablation=signed',
        'truss_topologies=[' + ','.join(job['topologies']) + ']',
        f'seed={job["seed"]}', f'work_dir={directory}', 'steps=100000',
        '++distillation.teachers={' + mapping + '}', 'distillation.offline_only=true',
        'distillation.reconstruct_control_metadata=true',
        f'distillation.pretrain_updates={manifest["updates"]}',
        f'distillation.batch_size={manifest["batch_size"]}',
        f'distillation.cache_dir={manifest["target_cache"]}',
        'message_attention=true', 'use_virtual_node=true', 'save_video=false',
        'save_agent=false', 'save_csv=false', 'domain_randomization_params.observation_noise.enabled=false',
        'domain_randomization_params.action_noise.enabled=false',
        'domain_randomization_params.length_scale.enabled=false',
        f'enable_wandb={str(manifest.get("wandb", True)).lower()}', 'set_wandb_offline=true',
        f'wandb_dir={manifest["run_root"]}',
        f'tube_experiment_base=teacher-student-v1-{job["label"]}',
        f'wandb_name=teacher-student-{job["label"]}-seed-{job["seed"]}',
    ]
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        cfg = parse_cfg(compose(config_name='config', overrides=overrides + manifest.get('overrides', [])))
    # Only obtain the raw graph dimensions; no SAC collection or GPU simulator.
    env = make_env(cfg)
    env.close()
    random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
    cfg.device = manifest.get('device', 'cuda')
    agent = GNNSAC(cfg)
    digest = hashlib.sha256()
    for parameter in agent.model.actor_parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    identity = dict(initial_actor_sha256=digest.hexdigest(),
                    actor_parameters=sum(p.numel() for p in agent.model.actor_parameters()),
                    topology_count=len(job['topologies']), updates_per_topology=manifest['updates'],
                    observations_per_topology_per_update=manifest['batch_size'])
    (directory / 'identity.json').write_text(json.dumps(identity, indent=2))
    distillation = Distillation(cfg, GNNBuffer._task_names(cfg))
    logger = Logger(cfg)
    try:
        while distillation.completed_updates < manifest['updates']:
            metrics = distillation.offline_update(agent)
            update = distillation.completed_updates
            if update == 1 or update % 100 == 0 or update == manifest['updates']:
                logger.log(dict(step=0, **metrics), 'distillation')
            if update in manifest['snapshots']:
                path = directory / f'update_{update}.pt'
                torch.save(dict(config=vars(cfg), agent={'model': agent.model.state_dict()},
                                offline_updates=update, identity=identity), path)
        (directory / 'complete.json').write_text(json.dumps(dict(updates=update, final_training_metrics=metrics), indent=2))
    finally:
        logger.finish()


def evaluate_student(manifest: dict, index: int) -> None:
    job = manifest['jobs'][index]
    directory = Path(job['directory'])
    results = {}
    for update in manifest['snapshots']:
        state = load(directory / f'update_{update}.pt')
        state['config']['device'] = 'cpu'
        cfg = cfg_to_dataclass(OmegaConf.create(state['config']))
        student = GNNActorCritic(cfg).eval()
        student.load_state_dict(state['agent']['model'])
        gradients, metrics = {}, {}
        for topology in job['topologies']:
            teacher_state = load(manifest['validation_replay'][topology])
            teacher_cfg = cfg_to_dataclass(OmegaConf.create(teacher_state['config']))
            teacher = GNNActorCritic(teacher_cfg).eval()
            teacher.load_state_dict(teacher_state['agent']['model'])
            buffers = teacher_state['buffer']['buffers']
            if len(buffers) != 1:
                raise ValueError('Validation replay must contain one topology.')
            replay = next(iter(buffers.values()))
            graphs = replay_samples(replay, manifest['evaluation_samples'], 424242)
            metadata = reconstruct_control_metadata(teacher_cfg, cfg, replay_contract_observation(replay, graphs[0]))
            graphs = [attach_control_metadata(g, metadata) for g in graphs]
            tb = Batch.from_data_list([prepared(teacher_cfg, g) for g in graphs])
            sb = Batch.from_data_list([prepared(cfg, g) for g in graphs])
            with torch.no_grad():
                tm, tl = teacher.policy_distribution(tb)
            sm, sl = student.policy_distribution(sb)
            kl = gaussian_forward_kl(tm, tl, sm, sl).mean()
            grads = torch.autograd.grad(kl, tuple(student.actor_parameters()))
            gradients[topology] = torch.cat([g.detach().flatten() for g in grads])
            rows = rollout(student, cfg, teacher_state['config'], manifest['reset_seeds'])
            metrics[topology] = dict(fresh_teacher_kl=float(kl.detach()),
                action_mse=float((tm.tanh() - sm.tanh()).square().mean().detach()), rollouts=rows,
                episode_distance=float(np.mean([r['distance'] for r in rows])),
                episode_length=float(np.mean([r['length'] for r in rows])))
        cosine = {a: {b: float(torch.nn.functional.cosine_similarity(ga[None], gb[None]))
                      for b, gb in gradients.items()} for a, ga in gradients.items()}
        results[str(update)] = dict(topologies=metrics, actor_gradient_cosine=cosine)
        (directory / 'evaluation.json').write_text(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'train', 'evaluate'])
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    manifest = json.loads(args.manifest.read_text())
    {'prepare': prepare_validation, 'train': train_student, 'evaluate': evaluate_student}[args.action](manifest, args.index)


if __name__ == '__main__':
    main()
