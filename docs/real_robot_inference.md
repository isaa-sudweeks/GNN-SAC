# Real-robot GNN inference

`sac/real_robot_infer.py` replaces the MuJoCo observation/step loop with SteamVR tracker input. It obtains the policy node order, control graph, action mask, and edge roles from either a selected `mujoco-truss-gen` preset or a hand-authored triangle definition. Preset mode constructs MuJoCo once for this metadata and then closes it; every closed-loop observation comes from SteamVR. The physical tracker positions in the first complete frame establish the observation bounding box and rigidity reference. Firmware-compatible commands are printed unless serial transport is explicitly enabled.

Real inference also records a JSONL diagnostic session by default. See
`docs/sim_to_real_diagnostics.md` for stationary tracker checks, exact graph-defined
MuJoCo command replay, paired comparison, and repeated-trial variance analysis.

Keep the permanent serial mapping in one JSON file:

```json
{
  "serial_to_tracker_id": {
    "LHR-AAAA": "tracker_1",
    "LHR-BBBB": "tracker_2"
  }
}
```

## Automatic assignment

Automatic mode is the default. No tracker-to-node mapping file is required:

```bash
python sac/real_robot_infer.py \
  model=/path/to/final.pt \
  truss_topology=octahedron \
  serial_map_file=/path/to/serial_map.json \
  tracker_assignment=automatic
```

On the first complete frame, the wrapper centers the transformed tracker cloud and nominal preset cloud, constructs every tracker/node distance, and finds the globally minimum-cost one-to-one assignment. The resulting assignment remains fixed for the process and is printed with its RMS matching error.

The physical robot should begin in the preset's nominal shape and orientation. Calibrate `steamvr_to_policy_matrix` if SteamVR and MuJoCo axes differ. Highly symmetric shapes can have multiple equally good assignments; those assignments may be graph symmetries, but use manual mode when the physical actuator identity makes a particular labeling important.

## Manual assignment

For an explicit placement-dependent assignment, provide a layout file:

```json
{
  "tracker_id_to_node": {"tracker_1": "node_0", "tracker_2": "node_1"},
  "steamvr_to_policy_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
}
```

Node names must match the selected preset. Graph structure and action metadata always come from that preset rather than the JSON file. Run manual mode with:

```bash
python sac/real_robot_infer.py \
  model=/path/to/final.pt \
  truss_topology=octahedron \
  serial_map_file=/path/to/serial_map.json \
  tracker_layout_file=/path/to/layout.json \
  tracker_assignment=manual \
  control_frequency_hz=10
```

Install the project dependencies, which pin the tested SteamVR binding to `openvr==2.5.102`, and start SteamVR before either command. This OpenVR release still imports `pkg_resources`, so the project also pins `setuptools==81.0.0`, the last compatible major release. For a targeted repair of an existing environment, run `python -m pip install --force-reinstall setuptools==81.0.0 openvr==2.5.102`. The configuration must match training, especially the topology, scale, realism/control-graph settings, normalization, and graph features. The current simulation observation centers x/y and retains absolute z, so `com_relative_axes` defaults to `[true, true, false]`; set all three true only when that matches training. The first velocity observation is zero; subsequent velocities use elapsed monotonic time. Set `velocity_filter_alpha` below 1 for exponential smoothing.

## Six-tracker triangle-plane reconstruction

Use `tracker_assignment=triangle_planes` when the physical system has one puck per ideal/abstract node while the policy uses duplicated control-graph nodes. This mode reads each tracker's full position and orientation, reconstructs each abstract spherical joint, and copies that physical joint position to every control node whose name has the same prefix before `_tri_`.

The layout explicitly records (1) which triangle carries each tracker, (2) which abstract joint the tracker is rigidly connected to, and (3) the two tracked triangle planes whose intersection contains that joint:

```json
{
  "tracker_mounts": {
    "B11": {
      "triangle": "triangle_0",
      "abstract_node": "node_0",
      "joint_triangles": ["triangle_0", "triangle_1"],
      "local_plane_normal": [0, 0, 1],
      "local_plane_point_offset": [0, 0, 0]
    },
    "B12": {
      "triangle": "triangle_1",
      "abstract_node": "node_1",
      "joint_triangles": ["triangle_1", "triangle_2"],
      "local_plane_normal": [0, 0, 1],
      "local_plane_point_offset": [0, 0, 0]
    }
  },
  "steamvr_to_policy_matrix": [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
  "plane_parallel_tolerance": 1e-6
}
```

Provide one entry for every ideal/abstract preset node. Multiple trackers may be mounted on the same triangle; their calibrated plane estimates are normal-aligned and averaged into one measured triangle plane. Every name in `joint_triangles` must correspond to a triangle carrying at least one tracker. The `local_plane_normal` is the triangle normal expressed in that tracker's local SteamVR frame. If the tracker origin is not actually on the triangle plane, measure the vector from the tracker origin to any point on the plane in tracker-local coordinates and set `local_plane_point_offset`; leaving it zero applies the derivation's tracker-origin-on-plane assumption.

Run the reconstructed mode with:

```bash
python sac/real_robot_infer.py \
  model=/path/to/final.pt \
  truss_topology=YOUR_PRESET \
  serial_map_file=config/tracker_maps/vive_serial_map.json \
  tracker_layout_file=/path/to/six_tracker_layout.json \
  tracker_assignment=triangle_planes
```

For each joint, the code intersects its two measured planes and orthogonally projects the rigidly connected tracker's position onto that line. A frame is skipped when the planes are parallel or nearly parallel because the joint is not observable from that plane pair. The mechanical perpendicularity assumption must hold for each configured tracker/joint pair; mounting a puck on a triangle alone does not guarantee it.

## Hand-authored graph and position checker

When the physical layout is not a registered `mujoco-truss-gen` preset, define the four logical triangles and six tracker locations directly. The loader generates the complete policy graph and plane-intersection metadata:

```json
{
  "triangles": [
    {
      "name": "triangle_1",
      "nodes": ["node_1", "node_2", "node_4"],
      "passive_node": "node_1",
      "trackers": {"node_1": "B11", "node_2": "B12"},
      "rollers": {"node_2": "02", "node_4": "04"}
    },
    {
      "name": "triangle_2",
      "nodes": ["node_1", "node_5", "node_3"],
      "passive_node": "node_1",
      "trackers": {"node_3": "B13", "node_5": "B15"},
      "rollers": {"node_5": "06", "node_3": "07"}
    },
    {
      "name": "triangle_3",
      "nodes": ["node_3", "node_6", "node_2"],
      "passive_node": "node_6",
      "trackers": {"node_6": "B16"},
      "rollers": {"node_3": "08", "node_2": "10"}
    },
    {
      "name": "triangle_4",
      "nodes": ["node_4", "node_6", "node_5"],
      "passive_node": "node_6",
      "trackers": {"node_4": "B14"},
      "rollers": {"node_4": "11", "node_5": "12"}
    }
  ],
  "tracker_calibration": {
    "B11": {
      "local_plane_normal": [0, 0, 1],
      "local_plane_point_offset": [0, 0, 0]
    },
    "B12": {
      "local_plane_normal": [0, 0, 1],
      "local_plane_point_offset": [0, 0, 0]
    }
  },
  "steamvr_to_policy_matrix": [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
  "plane_parallel_tolerance": 1e-6
}
```

The `trackers` object maps the logical node carrying a physical puck to that puck's stable ID from `serial_to_tracker_id`. Two entries on one triangle mean two physical pucks contribute independent estimates of the same rigid triangle plane. If the serial map uses logical node names as its tracker IDs, the shorter `"tracker_nodes": ["node_1", "node_2"]` form is equivalent. A single `"tracker": "node_1"` is also accepted. `tracker_calibration` is optional and defaults to local normal `[0,0,1]` with zero offset; add one entry per puck when its mount differs.

The `rollers` object maps each actuated node occurrence on that triangle to its physical roller number. Store numbers as decimal strings so identifiers such as `"02"` retain their leading zero. A triangle's passive node is omitted because it is masked. Once any `rollers` object is present, every actuated occurrence across the definition must have one, and roller numbers must be unique. The generated transmitter command order is ascending numeric roller number. Because the mapping is triangle-local, two copies of the same logical node may use different roller numbers.

The generator intentionally matches `mujoco-truss-gen`:

- Triangles and nodes are traversed in file order.
- The first occurrence of a logical node keeps its name. Later occurrences are named `<node>_tri_<triangle>`.
- Each triangle produces three `tube` edges.
- All control-node copies of the same logical node receive `connector` edges.
- The occurrence named by each triangle's `passive_node` is excluded from the action mask.
- The two triangles containing each logical node become that node's plane-intersection pair.

Every logical node must have exactly one puck, every triangle must carry at least one puck, and every logical node must occur in exactly two triangles. Those constraints describe the current four-triangle/six-node mechanism and prevent an ambiguous reconstruction from running.

Before loading a policy, validate the definition and print one live reconstructed frame:

```bash
python scripts/check_tracker_positions.py \
  --serial-map config/tracker_maps/vive_serial_map.json \
  --graph-definition /path/to/robot_graph.json
```

The first JSON line echoes the controller node order, action nodes, and connections. The second line contains the reconstructed positions both as an ordered array and keyed by node name. This checker calls the same `TrackerLayout.ordered_positions()` implementation as real inference; it neither loads a checkpoint nor creates motor commands.

For a repeatable offline math test, save a pose frame as JSON and pass `--poses`:

```json
{
  "poses_by_serial": {
    "LHR-AAAA": {
      "position": [0.1, 0.2, 0.3],
      "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    }
  }
}
```

To use the hand-authored graph for print-only policy inference, pass it instead of a preset tracker layout:

```bash
python sac/real_robot_infer.py \
  model=/path/to/final.pt \
  serial_map_file=config/tracker_maps/vive_serial_map.json \
  graph_definition_file=/path/to/robot_graph.json \
  num_policy_actions=NUMBER_OF_ACTUATED_NODES
```

Do not also set `tracker_layout_file`. The checkpoint architecture and enabled graph features must match the hand-authored definition. The triangle generator supplies `tube` and `connector` roles on every generated edge.

## Transmitter command output and emergency stop

Each control cycle prints the exact newline-delimited command expected by the Arduino USB transmitter:

```text
VEL_DUR:0,500,-300:0.2
```

Each normalized policy action is clipped to `[-1, 1]`, multiplied directly by the firmware limit of 1800 ticks/second, and rounded to an integer. The simulation-side `speed` conversion is not applied to serial commands. Passive tracked nodes remain policy context but are omitted from the transmitter fields. `serial_node_order` controls the transmitter channel order. When it is null, a triangle graph uses ascending `rollers` order; definitions without roller metadata and presets use actuated graph order. An explicit `serial_node_order` override still takes precedence.

`command_transport=print` remains the default. Pressing `Ctrl-C` prints one immediate zero-duration velocity command containing a zero for every configured transmitter channel, for example `VEL_DUR:0,0,0:0`, before the tracker source closes.

For physical actuation, set `command_transport=serial`, `serial_port`, and optionally `serial_baud_rate` (115200 by default). The pyserial transport appends `\n`, waits for the firmware's `VEL_DUR command completed in <ms> ms` response, and keeps at most one normal command in flight. It reports confirmed throughput, final per-node delivery failures, and cumulative delivery success after every command. `serial_ack_timeout_s` defaults to 5 seconds; a timeout or Ctrl-C stops normal control, makes one best-effort zero-command write, and closes the port. Complete `docs/real_robot_tracking_test_checklist.md` before enabling serial mode.

For `use_virtual_node=true`, the wrapper computes the first non-rigid eigenvalue from the configured bar graph and normalizes it by the first frame, matching the policy's rigidity input contract. Directly treating a reduced tracker set as the full graph cannot reproduce full-robot rigidity; triangle-plane mode first reconstructs and expands all configured control nodes before evaluating it.
