# Real-Robot Tracking and Serial Inference Test Checklist

This document is the pre-test and test-day checklist for running the GNN policy from HTC Vive tracker measurements. `command_transport=print` is the safe default and prints the exact Arduino transmitter commands. `command_transport=serial` opens the configured USB port and actuates the physical robot; use it only after completing the print-only checks and approval gate below.

## 1. Current system boundary

The test uses two sources of information:

- `mujoco-truss-gen` provides the selected preset's node order, graph connectivity, action mask, edge roles, and nominal node coordinates used for automatic tracker matching.
- The physical Vive trackers provide the live poses used to obtain policy-node positions. In six-tracker mode, orientations define triangle planes and the reconstructed abstract-joint positions are expanded to the control graph. The first complete physical frame establishes the initial bounding box and rigidity reference.

After reading the preset metadata, MuJoCo is closed. It does not supply observations or advance the robot during the closed-loop test.

Each successful control cycle performs:

1. Read all valid tracker poses from SteamVR.
2. Convert SteamVR coordinates into the policy coordinate frame.
3. Either order directly tracked nodes or reconstruct abstract joints from triangle planes and expand them into preset node order.
4. Compute center-relative node positions.
5. Estimate node velocities from consecutive timestamped frames.
6. Apply the observation normalization used during training.
7. Run deterministic GNN inference.
8. Convert each normalized active-node action to firmware ticks per second using `round(clip(action, -1, 1) * 1800)`.
9. Print one firmware-ready `VEL_DUR` command, or send it and wait for transmitter confirmation.

Passive physical nodes remain part of the GNN observation but are omitted from the transmitter command fields.

## 2. Files used by the test

- Inference entry point: `sac/real_robot_infer.py`
- Hydra configuration: `config/inference/real_robot.yaml`
- Vive serial mapping: `config/tracker_maps/vive_serial_map.json`
- General wrapper documentation: `docs/real_robot_inference.md`

The serial mapping may contain trackers that are not used in a particular layout. Direct automatic/manual mode needs one visible tracker per preset graph node. Triangle-plane mode instead needs one visible tracker and one mount entry per ideal/abstract node.

## 3. Information to record before testing

Fill this out before starting SteamVR:

- Date: ____________________
- Operator: ____________________
- Computer: ____________________
- Git branch and commit: ____________________
- Checkpoint path or W&B reference: ____________________
- Checkpoint training run: ____________________
- `truss_topology`: ____________________
- `truss_realistic`: ____________________
- `use_control_graph`: ____________________
- Robot/preset `scale`: ____________________
- Expected tracked-node count: ____________________
- Expected actuated/transmitter-channel count: ____________________
- Tracker assignment mode: automatic / manual / triangle_planes
- Control frequency: ____________________ Hz
- Command duration: ____________________ seconds
- Coordinate transform verified: yes / no

The checkpoint and runtime configuration must agree on topology, realism, control-graph mode, scale, graph features, virtual-node use, and observation normalization.

## 4. Tracking-computer preparation

- [ ] Use a computer that can run SteamVR and access all Vive trackers.
- [ ] Check out the intended `sim-to-real` branch and confirm the desired commit.
- [ ] Create or activate the GNN-SAC Python environment.
- [ ] Install the project dependencies.
- [ ] Confirm the tested OpenVR Python binding is installed in that same environment:

```bash
python -m pip install --force-reinstall setuptools==81.0.0 openvr==2.5.102
python -m pip show openvr
```

- [ ] Start SteamVR.
- [ ] Pair and power on every required tracker.
- [ ] Confirm each tracker reports a valid, stable pose in SteamVR.
- [ ] Confirm every physical serial number appears in `config/tracker_maps/vive_serial_map.json`.
- [ ] Disable sleep, automatic updates, and other interruptions for the duration of the test.

## 5. Physical setup

- [ ] For direct mode, secure one tracker to each intended physical node. For triangle-plane mode, secure one tracker to each configured triangle and abstract-joint mount.
- [ ] Check that tracker mounts cannot shift relative to the nodes.
- [ ] Arrange the robot as closely as possible to the nominal `mujoco-truss-gen` preset pose.
- [ ] Use the same robot scale as the selected preset configuration.
- [ ] Keep the robot stationary during initial tracker assignment and first-frame calibration.
- [ ] Verify that the SteamVR tracking origin will remain fixed during the test.
- [ ] Keep the USB transmitter disconnected for the initial print-only test.

## 6. Coordinate-frame calibration

`steamvr_to_policy_matrix` converts raw SteamVR coordinates to the MuJoCo/policy world axes. The repository default maps policy `(x, y, z)` to SteamVR `(x, z, y)`; verify that this matches the actual tracking setup.

Before trusting automatic assignment or policy commands, verify:

- Which SteamVR axis points in the robot's trained forward direction.
- Which SteamVR axis points laterally.
- Which SteamVR axis points upward.
- Whether any axis requires a sign reversal.

Configure the transform in `config/inference/real_robot.yaml` or an optional tracker layout JSON:

```yaml
steamvr_to_policy_matrix:
  - [1, 0, 0]
  - [0, 0, 1]
  - [0, 1, 0]
```

Do not proceed to live actuation later until a known physical displacement produces the expected sign and axis in the policy observation.

## 7. Tracker-to-node assignment

### Automatic mode

Automatic mode is the default:

```text
tracker_assignment=automatic
```

On the first complete frame, the wrapper centers the transformed physical and nominal preset point clouds and finds the globally minimum-cost one-to-one assignment. That assignment remains fixed until the program exits.

Requirements:

- [ ] All required trackers are visible simultaneously.
- [ ] The robot begins in the nominal preset shape.
- [ ] The robot and preset have the same scale.
- [ ] The coordinate transform and initial orientation agree with the preset.
- [ ] The printed assignment is inspected before its output is trusted.

Expected output:

```text
Automatic tracker assignment: {"B11": "node_...", ...} (RMS error=0.012345 m)
```

Record the RMS assignment error: ____________________ m

A large error indicates a likely wrong preset, wrong scale, bad coordinate transform, incorrect initial orientation, missing tracker, shifted mount, or incomplete nominal pose.

Symmetric robots may have multiple assignments with equal or nearly equal error. Use manual assignment if actuator identity requires one particular symmetric labeling.

### Manual mode

Use manual assignment when the automatic result is ambiguous or physically incorrect:

```text
tracker_assignment=manual
tracker_layout_file=/absolute/path/to/layout.json
```

Example layout:

```json
{
  "tracker_id_to_node": {
    "B11": "node_0",
    "B12": "node_1"
  },
  "steamvr_to_policy_matrix": [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
  ]
}
```

Node names must exactly match the selected preset.

### Six-tracker triangle-plane mode

Use this mode when each puck is attached to one triangle/abstract-joint mount and the policy graph contains duplicated `_tri_` control nodes:

```text
tracker_assignment=triangle_planes
tracker_layout_file=/absolute/path/to/six_tracker_layout.json
```

Follow the `tracker_mounts` schema in `docs/real_robot_inference.md`. Before the bounded test, verify:

- [ ] There is exactly one mount for every abstract preset node.
- [ ] Every mount identifies its tracker-bearing triangle and both joint-intersection triangles.
- [ ] `local_plane_normal` matches the physical tracker orientation.
- [ ] `local_plane_point_offset` is measured if the tracker origin is not on the triangle plane.
- [ ] The tracker-to-joint direction is mechanically perpendicular to the configured plane-intersection line.
- [ ] The test pose does not make either configured plane pair nearly parallel.

## 8. First bounded tracking test

Start with a five-second, print-only test at 10 Hz:

```bash
python sac/real_robot_infer.py \
  model=/absolute/path/to/final.pt \
  truss_topology=YOUR_PRESET \
  serial_map_file=config/tracker_maps/vive_serial_map.json \
  tracker_assignment=triangle_planes \
  tracker_layout_file=/absolute/path/to/six_tracker_layout.json \
  control_frequency_hz=10 \
  control_steps=50
```

Add any checkpoint-specific overrides required to match training, such as `truss_realistic`, `use_control_graph`, `scale`, or graph-feature settings.

During the run, verify:

- [ ] The checkpoint loads without a graph-feature-schema mismatch.
- [ ] The expected preset and node count are used.
- [ ] The selected direct assignment or triangle-plane layout validates at startup.
- [ ] No configured tracker is reported missing.
- [ ] One command is printed per successful frame.
- [ ] The number of command fields equals the actuated-node/transmitter-channel count.
- [ ] Every velocity is within `[-1800, 1800]`.
- [ ] Passive graph nodes do not add transmitter fields.
- [ ] The program exits after 50 successful frames.

Example printed command:

```text
VEL_DUR:450,-900,0,1800:0.2
```

At 10 Hz, a new command is produced every 0.1 seconds. The default command duration is 0.2 seconds, allowing one missed update interval before the firmware command expires.

## 9. Motion-observation checks

Repeat bounded tests while moving one tracker/mount slowly by hand. Do not connect actuation yet.

- [ ] Move the robot in the trained forward direction and verify command changes are repeatable.
- [ ] Move it laterally and verify the response differs from forward motion as expected.
- [ ] Raise one node and verify the correct node is represented.
- [ ] Hold the robot still and inspect whether finite-difference velocity noise produces unstable commands.
- [ ] If necessary, reduce velocity noise using `velocity_filter_alpha` below `1.0`.
- [ ] Briefly occlude one tracker and confirm the frame is skipped rather than inferred from incomplete data.

Expected dropout message:

```text
tracker frame skipped: Missing valid SteamVR poses for nodes: [...]
```

The initial observation has zero velocity by design. Velocity is calculated from subsequent positions using monotonic elapsed time.

## 10. Ctrl-C emergency-stop test

Run an unbounded or longer print-only test:

```bash
python sac/real_robot_infer.py \
  model=/absolute/path/to/final.pt \
  truss_topology=YOUR_PRESET \
  serial_map_file=config/tracker_maps/vive_serial_map.json \
  tracker_assignment=automatic \
  control_frequency_hz=10
```

Press `Ctrl-C` while commands are printing.

- [ ] Command printing stops.
- [ ] The final standard-output command contains one zero for every transmitter channel.
- [ ] The stop command uses a zero-second duration.
- [ ] The tracker source closes cleanly.

Example:

```text
VEL_DUR:0,0,0,0:0
```

The number of zeros will depend on the selected preset's actuated-node count.

## 11. Acknowledgment-gated serial test

The serial transport sends the exact print-mode command followed by a newline:

```text
VEL_DUR:<v1>,<v2>,...:<duration>\n
```

It does not send another normal command until the transmitter prints:

```text
VEL_DUR command completed in <milliseconds> ms
```

The host also parses final delivery failures such as `FAILED to send to node 3 after 3 attempts`. Firmware node numbers are 1-based transmitter indices and are mapped through `serial_node_order` when printed. These are final failures after firmware retries; the current firmware does not expose individual RF retry losses.

Before connecting motor power, verify `serial_node_order` against the receiver-address array in the transmitter firmware. Do not infer transmitter order from Vive tracker IDs.

Run a bounded serial test with motors made mechanically safe:

```bash
python sac/real_robot_infer.py \
  model=/absolute/path/to/final.pt \
  truss_topology=YOUR_PRESET \
  serial_map_file=config/tracker_maps/vive_serial_map.json \
  command_transport=serial \
  serial_port=/dev/cu.usbserial-XXXX \
  serial_baud_rate=115200 \
  serial_ack_timeout_s=5 \
  control_frequency_hz=10 \
  control_steps=50
```

- [ ] The connection message names the intended port and baud rate.
- [ ] Every normal write is followed by a `VEL_DUR command completed` line before the next command.
- [ ] `effective_hz` never indicates catch-up bursts above the configured maximum rate.
- [ ] `current_failed_nodes` identifies the expected transmitter index and graph node when a receiver is intentionally unavailable.
- [ ] `node_delivery_success`, `node_drops`, and `cumulative_node_drops` update consistently.
- [ ] An acknowledgment timeout halts normal commands, writes one best-effort zero command, and closes the port.
- [ ] Ctrl-C writes one best-effort zero-duration command and closes the port.

Serial defaults are 115200 baud, a 5-second acknowledgment timeout, a 2-second Arduino startup delay, ±1800 ticks/second, and a 0.2-second regular command duration. `control_frequency_hz` is a maximum: slow confirmations skip missed slots rather than triggering catch-up sends.

## 12. Stop conditions

Stop the test immediately if any of the following occurs:

- Tracker-to-node assignment is missing, duplicated, or physically implausible.
- Assignment RMS error is unexpectedly large.
- The node or command-field count differs from expectation.
- A known physical movement appears on the wrong policy axis or with the wrong sign.
- Commands remain saturated near ±1800 while the robot is stationary.
- Tracker loss does not produce a skipped frame.
- `Ctrl-C` does not print the expected all-zero command.
- Serial mode sends a new normal command without a completion acknowledgment.
- Delivery failures rise unexpectedly or repeatedly identify the same node.
- Effective serial throughput is too low for the configured command duration.
- The runtime configuration does not match the checkpoint.

## 13. Test record

- Automatic/manual assignment result saved: yes / no
- Assignment RMS error: ____________________ m
- Bounded test passed: yes / no
- Tracker dropout test passed: yes / no
- Coordinate-axis checks passed: yes / no
- Command field count: ____________________
- Expected field count: ____________________
- Stationary command behavior acceptable: yes / no
- Ctrl-C emergency-stop test passed: yes / no
- Serial port and baud: ________________________________________________
- Confirmed command rate: ____________________ Hz
- Node delivery success: ____________________ %
- Nodes with final delivery failures: __________________________________
- Unexpected warnings/errors: ________________________________________________
- Notes: _________________________________________________________________

## 14. Approval gate before live serial actuation

Do not switch `command_transport` to `serial` solely because the print-only test runs. Before live actuation, confirm all of the following:

- [ ] Tracker assignment has been physically verified.
- [ ] Coordinate axes and signs have been verified.
- [ ] Transmitter channel order has been verified against firmware receiver addresses.
- [ ] Command count matches the firmware's configured node count.
- [ ] A hardware-level stop path exists independently of the Python process.
- [ ] Ctrl-C and process-failure behavior have been tested with the transmitter connected but motors made safe.
- [ ] The first powered test uses constrained speed, short duration, physical restraints, and a spotter.
