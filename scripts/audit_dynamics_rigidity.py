"""Check geometry-only rigidity against ordinary source simulator trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'sac')]
from dynamics.rigidity import critical_eigenvalue
from scripts.run_simulator_mpc_pilot import make_source_env


def audit(output: Path, steps: int = 40) -> dict:
    """Audit tetrahedron only; do not use these states for fitting or selection."""
    if output.exists():
        raise FileExistsError(output)
    env = make_source_env(OmegaConf.create({'topology': 'tetrahedron'}))
    started = time.perf_counter()
    results = []
    try:
        env.reset()
        base = env.unwrapped
        model = base.mj_model
        mapping = {name: i for i, name in enumerate(model.node_names)}
        edges = np.asarray([[mapping[a], mapping[b]] for a, b in model.structural_edges])
        for seed in (9200, 9201, 9202):
            env.reset()
            base.reset(seed=seed)
            rng = np.random.default_rng(seed + 1000000)
            for step in range(steps):
                positions = model.get_node_position_matrix()
                observed = model._critical_eig()
                reconstructed = critical_eigenvalue(positions, edges,
                    axis_indices=tuple(model.axis_indices), weighted=model.wcrm)
                results.append({'seed': seed, 'step': step,
                    'simulator': float(observed), 'geometry': reconstructed,
                    'absolute_error': abs(observed - reconstructed)})
                _, _, done, _ = env.step(rng.uniform(-.5, .5, env.action_space.shape))
                if done:
                    break
        report = {'topology': 'tetrahedron', 'role': 'source_geometry_audit_only',
                  'model_inputs': 'physical positions and fixed structural graph only',
                  'n_states': len(results), 'max_absolute_error': max(r['absolute_error'] for r in results),
                  'initial_reference': float(base._initial_critical_eig),
                  'axis_indices': list(model.axis_indices), 'weighted': bool(model.wcrm),
                  'structural_edges': edges.tolist(), 'wall_seconds': time.perf_counter() - started,
                  'limits': ['Only ordinary source tetrahedron states checked.',
                             'No near-collapse false-negative evidence.',
                             'Abstract trusses only; realistic connector graphs not audited.'],
                  'results': results}
    finally:
        env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/dynamics-pilot/rigidity-2026-10-23/report.json')
    parser.add_argument('--steps', type=int, default=40)
    args = parser.parse_args()
    report = audit(args.output, args.steps)
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))
