#!/usr/bin/env python3
"""Collect matched, noise-free replay using a frozen teacher (CPU Slurm worker)."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'sac'):
    sys.path.insert(0, str(path))

from common.gnn_buffer import GNNBuffer
from common.parser import cfg_to_dataclass
from env import make_env
from gnn_sac import GNNSAC


def prepare_replay(source: Path, output: Path, samples: int, seed: int) -> Path:
    """Reuse only a provenance-matched cache; atomically publish clean replay."""
    if samples < 1:
        raise ValueError('Replay sample count must be positive.')
    source, output = source.resolve(), output.resolve()
    stat = source.stat()
    provenance = dict(version=1, source=str(source), size=stat.st_size,
                      mtime_ns=stat.st_mtime_ns, samples=samples, seed=seed)
    if output.exists():
        cached = torch.load(output, map_location='cpu', weights_only=False, mmap=True)
        if cached.get('tube_replay_provenance') == provenance:
            print(f'Reusing {output}', flush=True)
            return output
        raise ValueError(f'Clean replay cache provenance changed: {output}. Use a new cache directory.')
    state = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    config = deepcopy(state['config'])
    config.update(device='cpu', sim_backend='mujoco', mujoco_backend='mujoco',
                  num_envs=1, multitask=False, truss_topologies=None,
                  buffer_size=samples, steps=samples, batch_size=min(256, samples), seed=seed)
    parameters = config.setdefault('domain_randomization_params', {})
    for name in ('observation_noise', 'action_noise', 'length_scale'):
        parameters.setdefault(name, {})['enabled'] = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cfg = cfg_to_dataclass(OmegaConf.create(config))
    env = make_env(cfg)
    try:
        agent = GNNSAC(cfg)
        agent.model.load_state_dict(state['agent']['model'])
        agent.requires_grad_(False).eval()
        buffer = GNNBuffer(cfg)
        observation = env.reset()
        for index in range(samples):
            action = agent.act(observation)
            following, reward, done, info = env.step(action)
            buffer.add([
                dict(obs=observation, action=torch.zeros_like(action),
                     reward=torch.tensor(0.), terminated=torch.tensor(False)),
                dict(obs=following, action=action, reward=reward, terminated=info['terminated']),
            ], count_episode=bool(done))
            observation = env.reset() if bool(done) else following
            if (index + 1) % 1024 == 0:
                print(f'{index + 1}/{samples} clean transitions', flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f'.{os.getpid()}.tmp')
        try:
            torch.save(dict(config=vars(cfg), agent={'model': state['agent']['model'],
                       'graph_feature_schema': state['agent'].get('graph_feature_schema')},
                       buffer=buffer.state_dict(), tube_replay_provenance=provenance), temporary)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        env.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--index', type=int, default=int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')))
    args = parser.parse_args()
    task = json.loads(args.manifest.read_text())[args.index]
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    prepare_replay(Path(task['source']), Path(task['output']), task['samples'], task['seed'])


if __name__ == '__main__':
    main()
