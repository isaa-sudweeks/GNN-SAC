"""Source tetrahedron branch restoration and native-MuJoCo throughput check."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sac"))

from dynamics.simulator import SimulatorBranch
from scripts.run_simulator_mpc_pilot import make_source_env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/dynamics-pilot/simulator-branch.json")
    parser.add_argument("--candidates", type=int, default=16)
    args = parser.parse_args()
    settings = OmegaConf.load(ROOT / "config/dynamics_pilot.yaml")
    live, planning = make_source_env(settings), make_source_env(settings)
    live.reset()
    planning.reset()
    branch = SimulatorBranch(live, planning)
    rng = np.random.default_rng(7022)
    for action in rng.uniform(-0.2, 0.2, (4, *live.action_space.shape)):
        live.step(action)
    snapshot = branch.capture()
    before = live.unwrapped.mj_model.data.qpos.copy()
    records = []
    for horizon in (10, 20, 40):
        sequences = rng.uniform(-0.5, 0.5, (args.candidates, horizon, *live.action_space.shape))
        results, latencies = [], []
        for sequence in sequences:
            start = time.perf_counter()
            results.append(branch.rollout(snapshot, sequence))
            latencies.append(time.perf_counter() - start)
        repeated = [branch.rollout(snapshot, sequence) for sequence in sequences[::-1]][::-1]
        errors = [float(np.max(np.abs(a.com_positions - b.com_positions)))
                  for a, b in zip(results, repeated)]
        records.append({
            "horizon": horizon,
            "requested_seconds": horizon * live.unwrapped.nsubsteps * live.unwrapped.mj_model.model.opt.timestep,
            "candidate_seconds": latencies,
            "candidate_median_seconds": float(np.median(latencies)),
            "candidate_p95_seconds": float(np.percentile(latencies, 95)),
            "simulated_control_steps": sum(len(result.rewards) for result in results),
            "physics_substeps": sum(result.simulator_steps for result in results),
            "control_steps_per_second": sum(len(result.rewards) for result in results) / sum(latencies),
            "max_order_com_error": max(errors),
            "order_rewards_bitwise_equal": all(np.array_equal(a.rewards, b.rewards) for a,b in zip(results,repeated)),
            "stopped_candidates": sum(result.terminated or result.truncated or result.nonfinite for result in results),
            "projected_256_by_3_seconds": float(np.mean(latencies) * 256 * 3),
        })
    result = {
        "topology": "tetrahedron", "seed": 7022, "reset_seed": 4101,
        "runtime": sys.version, "mujoco": importlib.metadata.version("mujoco"),
        "mujoco_truss_gen": importlib.metadata.version("mujoco-truss-gen"),
        "git_revision": subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT,text=True).strip(),
        "config": str(live.unwrapped.source_config),
        "snapshot_signature": snapshot.signature,
        "physics_timestep_seconds": live.unwrapped.mj_model.model.opt.timestep,
        "control_interval_seconds": live.unwrapped.nsubsteps * live.unwrapped.mj_model.model.opt.timestep,
        "live_qpos_unchanged": bool(np.array_equal(before, live.unwrapped.mj_model.data.qpos)),
        "live_steps_unchanged": live.unwrapped.steps == 4,
        "benchmarks": records,
        "timing_scope": "Branch restore and rollout only; repeats for order checks excluded from throughput. Projections are estimates, not measured CEM latency.",
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
    live.close()
    planning.close()


if __name__ == "__main__":
    main()
