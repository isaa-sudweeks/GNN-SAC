#!/usr/bin/env python3
"""Print controller-ordered positions from a hand-authored tracker graph."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from real_robot_infer import (  # noqa: E402
    MissingTrackerFrame,
    SteamVRTrackerSource,
    TrackerLayout,
)


def _load_recorded_poses(path: Path) -> dict:
    with path.expanduser().open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Recorded pose file must contain a JSON object")
    poses = data.get("poses_by_serial", data)
    if not isinstance(poses, dict):
        raise ValueError("poses_by_serial must be a JSON object")
    return poses


def _undirected_connections(layout: TrackerLayout) -> list[list[object]]:
    connections = []
    seen = set()
    role_names = {0: "tube", 1: "connector"}
    for edge_number, (source, target) in enumerate(layout.edge_index.T):
        key = tuple(sorted((int(source), int(target))))
        if key in seen:
            continue
        seen.add(key)
        connection = [layout.node_names[key[0]], layout.node_names[key[1]]]
        if layout.edge_role is not None:
            role = int(layout.edge_role[edge_number])
            connection.append(role_names.get(role, role))
        connections.append(connection)
    return connections


def _configuration_record(layout: TrackerLayout) -> dict:
    return {
        "type": "configuration",
        "tracker_assignment": layout.assignment_mode,
        "node_order": list(layout.node_names),
        "actuated_nodes": [
            name for name, actuated in zip(layout.node_names, layout.action_mask) if actuated
        ],
        "serial_node_order": (
            list(layout.serial_node_order) if layout.serial_node_order is not None else None
        ),
        "connections": _undirected_connections(layout),
    }


def _position_record(layout: TrackerLayout, tracker_frame: dict) -> dict:
    positions = layout.ordered_positions(tracker_frame)
    return {
        "type": "controller_positions",
        "node_order": list(layout.node_names),
        "positions": np.asarray(positions, dtype=float).tolist(),
        "positions_by_node": {
            name: np.asarray(position, dtype=float).tolist()
            for name, position in zip(layout.node_names, positions)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a complete hand-authored graph, run the production tracker "
            "coordinate/reconstruction math, and print positions in controller node order."
        )
    )
    parser.add_argument("--serial-map", type=Path, required=True)
    parser.add_argument("--graph-definition", type=Path, required=True)
    parser.add_argument(
        "--poses",
        type=Path,
        help="Recorded pose JSON for an offline one-frame check; omit to read SteamVR live.",
    )
    parser.add_argument(
        "--frames", type=int, default=1, help="Number of complete live frames to print."
    )
    parser.add_argument("--frequency-hz", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if not np.isfinite(args.frequency_hz) or args.frequency_hz <= 0:
        raise ValueError("--frequency-hz must be positive and finite")

    layout = TrackerLayout.from_files(
        str(args.serial_map.expanduser()), str(args.graph_definition.expanduser())
    )
    print(json.dumps(_configuration_record(layout), sort_keys=True), flush=True)

    if args.poses is not None:
        print(
            json.dumps(
                _position_record(layout, _load_recorded_poses(args.poses)),
                sort_keys=True,
            ),
            flush=True,
        )
        return

    source = SteamVRTrackerSource()
    complete_frames = 0
    period = 1.0 / args.frequency_hz
    try:
        while complete_frames < args.frames:
            try:
                tracker_frame = (
                    source.poses_by_serial()
                    if layout.requires_orientations
                    else source.positions_by_serial()
                )
                print(
                    json.dumps(_position_record(layout, tracker_frame), sort_keys=True),
                    flush=True,
                )
                complete_frames += 1
            except MissingTrackerFrame as exc:
                print(f"tracker frame skipped: {exc}", file=sys.stderr, flush=True)
            if complete_frames < args.frames:
                time.sleep(period)
    finally:
        source.close()


if __name__ == "__main__":
    main()
