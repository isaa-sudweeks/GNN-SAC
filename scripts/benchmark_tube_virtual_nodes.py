#!/usr/bin/env python3
"""Measure prepared graph size and actor inference overhead for tube membership."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from hydra import compose, initialize_config_dir
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'sac'):
    sys.path.insert(0, str(path))
from common.gnn_actor_critic import GNNActorCritic
from common.graph_transforms import graph_feature_flags, prepare_graph
from common.parser import parse_cfg
from env import make_env


def benchmark(topology: str, device: str, batch_size: int, iterations: int) -> dict:
    """Report resident tensor bytes and synchronized actor latency, excluding physics."""
    if batch_size < 1 or iterations < 1:
        raise ValueError('batch_size and iterations must be positive')
    results = {}
    for arm in ('signed', 'membership'):
        with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
            cfg = parse_cfg(compose(config_name='config', overrides=[
                'sac_backend=gnn', f'+tube_ablation={arm}', f'truss_topology={topology}',
                f'device={device}', 'domain_randomization=false', 'enable_wandb=false',
                'save_video=false', 'work_dir=/tmp/tube-benchmark',
                'use_virtual_node=true', 'message_attention=true']))
        env = make_env(cfg)
        try:
            raw = env.reset()
            prepared = prepare_graph(raw, use_virtual_node=True, **graph_feature_flags(cfg))
            graph = Batch.from_data_list([prepared] * batch_size).to(device)
            model = GNNActorCritic(cfg).to(device).eval()
            def synchronize():
                if torch.device(device).type == 'cuda':
                    torch.cuda.synchronize(device)
            with torch.no_grad():
                for _ in range(5):
                    model.pi_mean(graph)
                synchronize()
                start = time.perf_counter()
                for _ in range(iterations):
                    model.pi_mean(graph)
                synchronize()
            results[arm] = dict(
                nodes_per_graph=prepared.num_nodes,
                directed_edges_per_graph=prepared.edge_index.size(1),
                prepared_tensor_bytes=sum(v.numel() * v.element_size()
                                          for v in graph.to_dict().values() if isinstance(v, torch.Tensor)),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                actor_batch_ms=1000 * (time.perf_counter() - start) / iterations,
            )
        finally:
            env.close()
    return dict(topology=topology, device=device, batch_size=batch_size,
                iterations=iterations, scope='actor inference; excludes simulator and preprocessing', arms=results)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topology', default='octahedron')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = benchmark(args.topology, args.device, args.batch_size, args.iterations)
    rendered = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == '__main__':
    main()
