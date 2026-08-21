# Real-robot GNN inference

`sac/real_robot_infer.py` replaces the MuJoCo observation/step loop with SteamVR tracker input. It still constructs the selected `mujoco-truss-gen` preset once at startup to obtain the policy node order, control graph, action mask, edge roles, and nominal coordinates used for tracker matching. MuJoCo is then closed; every closed-loop observation comes from SteamVR. The physical tracker positions in the first complete frame establish the observation bounding box and rigidity reference. Commands are printed as JSON and are not sent to hardware.

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

Install the optional SteamVR binding (`pip install openvr`) and start SteamVR before either command. The configuration must match training, especially the topology, scale, realism/control-graph settings, normalization, and graph features. The current simulation observation centers x/y and retains absolute z, so `com_relative_axes` defaults to `[true, true, false]`; set all three true only when that matches training. The first velocity observation is zero; subsequent velocities use elapsed monotonic time. Set `velocity_filter_alpha` below 1 for exponential smoothing.

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

Provide one entry for every ideal/abstract preset node and one uniquely named triangle for every physical tracker. Every name in `joint_triangles` must correspond to one of those tracker-mounted triangles. The `local_plane_normal` is the triangle normal expressed in that tracker's local SteamVR frame. If the tracker origin is not actually on the triangle plane, measure the vector from the tracker origin to any point on the plane in tracker-local coordinates and set `local_plane_point_offset`; leaving it zero applies the derivation's tracker-origin-on-plane assumption.

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

## Transmitter command output and emergency stop

Each control cycle prints the exact newline-delimited command expected by the Arduino USB transmitter:

```text
VEL_DUR:0,500,-300:0.2
```

Each normalized policy action is clipped to `[-1, 1]`, multiplied directly by the firmware limit of 1800 ticks/second, and rounded to an integer. The simulation-side `speed` conversion is not applied to serial commands. Passive tracked nodes remain policy context but are omitted from the transmitter fields. `serial_node_order` controls the transmitter channel order; when null, it uses the actuated nodes in preset graph order. Set it explicitly when firmware receiver-address order differs from the preset.

Pressing `Ctrl-C` prints one immediate zero-duration velocity command containing a zero for every configured transmitter channel, for example `VEL_DUR:0,0,0:0`, before the tracker source closes. Serial I/O is intentionally not implemented yet; the future writer should send the printed command plus `\n` at `serial_baud_rate` (115200 by default).

For `use_virtual_node=true`, the wrapper computes the first non-rigid eigenvalue from the configured bar graph and normalizes it by the first frame, matching the policy's rigidity input contract. Directly treating a reduced tracker set as the full graph cannot reproduce full-robot rigidity; triangle-plane mode first reconstructs and expands all configured control nodes before evaluating it.
