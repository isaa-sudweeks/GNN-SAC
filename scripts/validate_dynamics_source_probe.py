"""Check indexed source files and replay sampled actions from saved full states."""

import argparse
import gzip
import json
import pickle
from pathlib import Path
import sys

import mujoco
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sac")]
from scripts.run_simulator_mpc_pilot import make_source_env
from dynamics.simulator import SimulatorBranch


def validate(directory: Path) -> dict:
    """Restore explicit integration state without inferring replay row chronology."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["role"] != "source_evaluation_only" or manifest["topology"] != "tetrahedron":
        raise ValueError("Validator is restricted to the source evaluation probe")
    env = make_source_env(OmegaConf.load(ROOT / "config/dynamics_pilot.yaml"))
    placeholder = make_source_env(OmegaConf.load(ROOT / "config/dynamics_pilot.yaml"))
    checks = []
    try:
        env.reset()
        placeholder.reset()
        restore = SimulatorBranch(placeholder, env)
        base = env.unwrapped
        for episode in manifest["episodes"]:
            snapshots = None
            if episode.get("full_snapshots"):
                with gzip.open(directory / episode["full_snapshots"], 'rb') as stream:
                    snapshots = pickle.load(stream)  # Trusted local collector output only.
            with np.load(directory / episode["file"]) as data:
                n = episode["transitions"]
                assert np.array_equal(data["timestep"], np.arange(n))
                assert np.all(data["episode_id"] == episode["episode_id"])
                assert np.array_equal(data["pre_state"][1:], data["next_state"][:-1])
                assert not np.any(data["terminated"][:-1] | data["truncated"][:-1])
                for index in sorted(set([0, n // 2, n - 1])):
                    if snapshots is not None:
                        restore.restore(snapshots[index])
                    else:
                        env.reset()
                        base._initial_critical_eig = manifest["initial_rigidity_reference"]
                        base.steps = index
                        env._t = index
                        base._observation_rigidity = None
                        mujoco.mj_setState(base.mj_model.model, base.mj_model.data,
                                           data["pre_state"][index], manifest["state_spec_value"])
                        mujoco.mj_forward(base.mj_model.model, base.mj_model.data)
                    _, reward, _, info = env.step(data["action"][index])
                    error = float(np.max(np.abs(base.mj_model.get_node_position_matrix()
                                                - data["next_physical_positions"][index])))
                    activation_error = float(np.max(np.abs(base.mj_model.data.act
                                                           - data["next_activation"][index])))
                    checks.append(dict(episode_id=episode["episode_id"], timestep=index,
                                       max_position_error_m=error, max_activation_error=activation_error,
                                       raw_reward_error=abs(float(reward) - data["reward"][index]),
                                       terminated_matches=bool(info["terminated"]) == bool(data["terminated"][index]),
                                       truncated_matches=bool(info["truncated"]) == bool(data["truncated"][index])))
        report = dict(transitions=sum(x["transitions"] for x in manifest["episodes"]),
                      restored_checks=checks, max_position_error_m=max(x["max_position_error_m"] for x in checks),
                      max_activation_error=max(x["max_activation_error"] for x in checks),
                      alignment_checks_passed=True,
                      restore_tolerance_m=1e-8,
                      reward_restore_passed=all(x['raw_reward_error'] < 1e-10 for x in checks),
                      passed=all(x["max_position_error_m"] < 1e-8 and x["max_activation_error"] < 1e-8
                                 and x["raw_reward_error"] < 1e-10
                                 and x["terminated_matches"] and x["truncated_matches"] for x in checks))
        (directory / "validation.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise RuntimeError("Saved transition restoration failed")
        return report
    finally:
        env.close()
        placeholder.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=ROOT / "outputs/dynamics-pilot/source-evaluation-probe-v2")
    validate(parser.parse_args().directory)
