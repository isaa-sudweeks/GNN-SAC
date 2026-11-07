"""Export nominal offset geometry and node-frame conventions; never open hardware."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sac")]

import hydra

from encoder_robot import build_encoder_geometry
from encoder_state import roller_frame


@hydra.main(config_name="inference/encoder_vive", config_path="../config", version_base=None)
def main(cfg):
    geometry = build_encoder_geometry(cfg)
    graph = geometry.graph
    mounts = {}
    for triangle, nodes in graph.triangle_nodes.items():
        order = geometry.triangle_indices[triangle]
        mounts[triangle] = {}
        for i, node in enumerate(nodes):
            pose = roller_frame(geometry.solver.initial_rollers[order[i:] + order[:i]])
            mounts[triangle][node] = {"origin_m": pose.position.tolist(), "rotation_matrix": pose.rotation.tolist()}
    payload = {"model_xml_sha256": geometry.model_xml_sha256,
               "reference_node_map": geometry.reference_node_map,
               "control_node_order": graph.control_node_names,
               "serial_node_order": graph.serial_node_order,
               "logical_node_order": graph.logical_node_names,
               "nominal_roller_positions_m": geometry.solver.initial_rollers.tolist(),
               "nominal_ball_positions_m": geometry.solver.initial_balls.tolist(),
               "nominal_edge_lengths_m": geometry.solver.reference_lengths.tolist(),
               "encoder_length_incidence": geometry.solver.incidence.tolist(),
               "tracker_mount_reference_frames": mounts,
               "note": "Nominal MuJoCo geometry only; measure and enter tracker mounts and encoder reference calibration."}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
