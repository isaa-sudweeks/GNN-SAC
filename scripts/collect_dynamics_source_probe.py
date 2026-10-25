"""Collect indexed source evaluation episodes including omitted actuator state.

This small dataset verifies collection mechanics. It is not a fitting dataset,
does not establish action coverage, and is never eligible for model selection.
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sac")]
from dynamics.planner import interpolate_knots
from dynamics.simulator import SimulatorBranch
from scripts.run_simulator_mpc_pilot import make_source_env


def simulator_state(base):
    """Capture MuJoCo integration state, including activation and solver history."""
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(base.mj_model.model, spec))
    mujoco.mj_getState(base.mj_model.model, base.mj_model.data, state, spec)
    return state


def collect(output: Path, seeds: list[int], steps: int) -> dict:
    """Save whole episodes independently, with true terminal observations."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {output}")
    output.mkdir(parents=True)
    settings = OmegaConf.load(ROOT / "config/dynamics_pilot.yaml")
    env = make_source_env(settings)
    planning = make_source_env(settings)
    records = []
    try:
        base = env.unwrapped
        env.reset()
        planning.reset()
        branch = SimulatorBranch(env, planning)
        xml = str(base.mj_model.xml)
        (output / "model.xml").write_text(xml)
        dt = float(base.nsubsteps * base.mj_model.model.opt.timestep)
        for episode_id, seed in enumerate(seeds):
            env.reset()
            obs, _ = base.reset(seed=seed)
            mask = obs["action_mask"][:, None]
            rng = np.random.default_rng(seed + 1000000)
            sequence = interpolate_knots(rng.normal(0, 0.5, (max(2, steps // 5), len(mask))), steps)
            sequence = np.clip(sequence, -1, 1)[:, :, None] * mask
            rows = []
            raw_infos = []
            snapshots = []
            for timestep, action in enumerate(sequence):
                snapshots.append(branch.capture())
                pre = simulator_state(base)
                pre_obs = obs["x"].copy()
                pre_act = base.mj_model.data.act.copy()
                pre_ctrl = base.mj_model.get_external_ctrl()
                pre_node_positions = base.mj_model.get_node_position_matrix().copy()
                pre_time = float(base.mj_model.data.time)
                obs, reward, done, info = env.step(action)
                rows.append(dict(
                    episode_id=episode_id, env_id=0, timestep=timestep, reset_seed=seed,
                    pre_state=pre, next_state=simulator_state(base),
                    pre_observation=pre_obs, next_observation=obs["x"].copy(),
                    action=action.copy(), activation=pre_act,
                    next_activation=base.mj_model.data.act.copy(), external_control=pre_ctrl,
                    next_external_control=base.mj_model.get_external_ctrl(),
                    physical_positions=pre_node_positions,
                    next_physical_positions=base.mj_model.get_node_position_matrix().copy(),
                    terminated=bool(info["terminated"]), truncated=bool(info["truncated"]),
                    elapsed_seconds=float(base.mj_model.data.time) - pre_time,
                    reward=float(reward), substeps=int(info["substeps_executed"])))
                raw_infos.append(info)
                if done:
                    break
            arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
            filename = f"evaluation-episode-{episode_id:03d}.npz"
            np.savez_compressed(output / filename, **arrays)
            snapshot_name = filename.replace('.npz', '-full-snapshots.pkl.gz')
            with gzip.open(output / snapshot_name, 'wb') as stream:
                pickle.dump(snapshots, stream, protocol=pickle.HIGHEST_PROTOCOL)
            (output / filename.replace(".npz", "-reward-components.json")).write_text(
                json.dumps(raw_infos, indent=2, default=lambda value: np.asarray(value).tolist()))
            aligned = len(rows) == 1 or np.array_equal(arrays["next_state"][:-1], arrays["pre_state"][1:])
            if not aligned:
                raise RuntimeError("Adjacent indexed transitions are inconsistent")
            records.append(dict(file=filename, seed=seed, episode_id=episode_id,
                                full_snapshots=snapshot_name,
                                full_snapshots_sha256=hashlib.sha256((output / snapshot_name).read_bytes()).hexdigest(),
                                transitions=len(rows), alignment_exact=aligned,
                                sha256=hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                                collapsed=bool(rows[-1]["terminated"])))
        manifest = dict(
            schema_version=2, role="source_evaluation_only", topology="tetrahedron",
            provenance="smooth random node actions, clipped normal knots, linear interpolation",
            fitting_eligible=False, model_selection_eligible=False, control_interval_seconds=dt,
            state_spec="mjSTATE_INTEGRATION", state_spec_value=int(mujoco.mjtState.mjSTATE_INTEGRATION),
            normalization="Policy observations scaled by initial bbox; physical_positions in metres",
            physical_node_names=base.mj_model.node_names, graph_node_names=base.graph_node_names,
            action_mask=mask[:, 0].tolist(), edge_index=obs["edge_index"].tolist(),
            initial_bbox=base.mj_model.initial_bounding_box_dimensions.tolist(),
            initial_diagonal=float(base.mj_model.initial_bounding_box_diagonal),
            initial_rigidity_reference=float(base._initial_critical_eig),
            source_config=OmegaConf.to_container(base.source_config),
            model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
            revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            dependencies={name: importlib.metadata.version(name) for name in
                          ("mujoco", "mujoco-truss-gen", "numpy", "torch")},
            episodes=records,
            limitations=["Single source topology; no held-out access.",
                         "No proof of teacher or planner action coverage.",
                         "Full snapshots include derived caches and wrapper counters; only load trusted local pickle files.",
                         "Probe size 4 episodes x 40 steps; not the planned fitting collection."])
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))
        return manifest
    finally:
        env.close()
        planning.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dynamics-pilot/source-evaluation-probe-v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[8100, 8101, 8102, 8103])
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive steps and unique reset seeds are required")
    collect(args.output, args.seeds, args.steps)
