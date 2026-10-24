"""Measure physical-coordinate and omitted-control fields on source tetrahedron."""

import argparse
import copy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sac")]

from dynamics.state import center_xy, mass_weighted_com, observation_to_physical
from env.mujoco_gen.topology_envs import MujocoPresetGraphEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dynamics-pilot/state")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"topology": "tetrahedron", "seed": 723, "variants": []}
    for control_graph in (False, True):
        cfg = OmegaConf.merge(OmegaConf.load(ROOT / "config/environment.yaml"), {
            "truss_topology": "tetrahedron", "task": "truss-graph", "seed": 723,
            "truss_realistic": False, "domain_randomization": False,
            "normalize_observations": True, "use_control_graph": control_graph,
            "nsubsteps": 100, "max_steps": 100, "graph_features": {"edge_roles": True},
        })
        env = MujocoPresetGraphEnv(cfg)
        try:
            obs, _ = env.reset(seed=723)
            wrapper = env.mj_model
            physical_positions = wrapper.get_node_position_matrix()
            physical_centered, origin = center_xy(physical_positions)
            converted = observation_to_physical(obs["x"], wrapper.initial_bounding_box_dimensions,
                                               normalized=True)
            graph_body_ids = (wrapper.get_control_node_body_ids() if control_graph else
                              np.array([wrapper.node_body_ids[name] for name in wrapper.node_names]))
            expected = wrapper.data.xpos[graph_body_ids]
            expected_centered, _ = center_xy(expected)
            body_com = mass_weighted_com(wrapper.data.xipos, wrapper.model.body_mass)
            before = copy.copy(wrapper.data)
            trials = []
            next_positions = []
            for ctrl_value in (0., .05):
                mujoco.mj_copyData(wrapper.data, wrapper.model, before)
                env.steps = 0
                wrapper.set_external_ctrl(np.full(env.num_external_actuators, ctrl_value))
                same_obs = env._get_obs()
                next_obs, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
                next_positions.append(wrapper.get_node_position_matrix().copy())
                trials.append({"initial_external_control": ctrl_value,
                    "pre_observation_max_difference": float(np.max(np.abs(same_obs["x"] - obs["x"]))),
                    "next_external_control": wrapper.get_external_ctrl().tolist(),
                    "substeps_executed": info["substeps_executed"],
                    "elapsed_seconds": float(wrapper.data.time - before.time),
                    "terminated": bool(terminated), "truncated": bool(truncated)})
            item = {"use_control_graph": control_graph,
                "physical_node_count": len(physical_positions), "graph_node_count": len(graph_body_ids),
                "unique_graph_bodies": len(np.unique(graph_body_ids)),
                "graph_body_ids": graph_body_ids.tolist(), "physical_node_names": wrapper.node_names,
                "graph_node_names": env.graph_node_names,
                "coordinate_roundtrip_max_error_m": float(np.max(np.abs(converted[:, :3] - expected_centered))),
                "reward_center_minus_physical_node_mean_m": (env._center_of_mass() - wrapper.get_node_position_matrix().mean(0)).tolist(),
                "initial_true_mass_com_m": body_com.tolist(),
                "initial_node_mean_m": physical_positions.mean(0).tolist(),
                "initial_true_mass_com_minus_node_mean_m": (body_com - physical_positions.mean(0)).tolist(),
                "omitted_control_trials": trials,
                "omitted_control_next_position_max_difference_m": float(np.max(np.abs(next_positions[1] - next_positions[0]))),
                "physics_timestep_seconds": float(wrapper.model.opt.timestep),
                "nominal_control_interval_seconds": float(wrapper.model.opt.timestep * env.nsubsteps),
                "nq": wrapper.model.nq, "nv": wrapper.model.nv, "na": wrapper.model.na,
                "actuator_dyntype": wrapper.model.actuator_dyntype.tolist(),
                "angle_bisector_controller_enabled": wrapper.angle_bisector_controller.enabled}
            activation_trials = []
            activation_positions = []
            for offset in (0., .05):
                mujoco.mj_copyData(wrapper.data, wrapper.model, before)
                env.steps = 0
                wrapper.data.act[:] += offset
                # Act is not an observation field; avoid changing geometric state.
                same_obs = env._get_obs()
                env.step(np.zeros(env.action_space.shape))
                activation_positions.append(wrapper.get_node_position_matrix().copy())
                activation_trials.append({"activation_offset": offset,
                    "pre_observation_max_difference": float(np.max(np.abs(same_obs["x"] - obs["x"]))),
                    "final_activation": wrapper.data.act.tolist()})
            item["omitted_activation_trials"] = activation_trials
            item["omitted_activation_next_position_max_difference_m"] = float(
                np.max(np.abs(activation_positions[1] - activation_positions[0])))
            report["variants"].append(item)
        finally:
            env.close()
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = []
    for i, variant in enumerate(report["variants"]):
        label = "Control graph" if variant["use_control_graph"] else "Physical graph"
        rows.append(f'<text x="30" y="{130 + i * 110}" font-size="21">{label}</text>')
        for j, key in enumerate(("omitted_control_next_position_max_difference_m",
                                 "omitted_activation_next_position_max_difference_m")):
            value = variant[key] * 1000
            rows.append(f'<text x="{300+j*240}" y="{130+i*110}" font-size="24">{value:.6f} mm</text>')
        rows.append(f'<text x="30" y="{165+i*110}" font-size="17">{variant["graph_node_count"]} graph rows; {variant["unique_graph_bodies"]} physical bodies</text>')
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="830" height="370" viewBox="0 0 830 370">'
           '<rect width="830" height="370" fill="#f7f9fb"/><g font-family="Arial" fill="#183342">'
           '<text x="30" y="35" font-size="23">Same visible state and zero action: different hidden state</text>'
           '<text x="300" y="80" font-size="18">Saved command changed</text>'
           '<text x="540" y="80" font-size="18">Actuator target changed</text>'
           + ''.join(rows) + '<text x="30" y="340" font-size="16">Largest next-position difference after 0.2 seconds; source tetrahedron, one start.</text></g></svg>')
    (args.output / "state-audit.svg").write_text(svg)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
