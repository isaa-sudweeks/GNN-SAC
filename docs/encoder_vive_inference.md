# Encoder shape estimation with two Vive anchors

Branch `sim-to-real-2` starts at lab commit `3a1275a`. This adds a sensor/estimation
path around the existing GNN, checkpoint loader, observation builder, node/edge
ordering, action mask, and firmware action scaling. It does not retrain or change
the learned models. Existing six-tracker inference remains available.

## Calibration and first use

Edit `config/tracker_maps/encoder_vive_calibration.json`. B13 and B16 are populated
as tracker IDs, with their existing serial map; their physical mounts are deliberately
unset. The program rejects an incomplete calibration before opening serial output.

1. Check the eight roller IDs in `config/tracker_maps/robot_def.json` and the actual
   flashed transmitter order. The user confirmed the flashed transmitter has eight
   slots, `01`–`08`; `firmware_receiver_order` is configurable. Do not use the upstream
   Arduino file's eleven-slot configuration as evidence of the flashed configuration.
2. Set `encoder_signs` to eight measured values, each `+1` or `-1`, in
   `serial_node_order`. Set `encoder_zero_ticks` to the encoder readings at the known
   reference shape. Do not reset encoders on startup or treat an arbitrary first pose
   as the reference. Confirm the real robot matches the configured nominal geometry
   before setting `reference_shape_confirmed=true`.
3. `ticks_per_inch=1125` comes from the firmware and kinematics examples. It can be
   a scalar or an eight-element array. Conversion is
   `displacement_m = sign * (ticks - zero_ticks) * 0.0254 / ticks_per_inch`.
   The reference hardware command example negates its generated values; that does
   not establish the sign of every mounted encoder on this robot. Signs therefore
   require explicit calibration. Encoder signs affect estimation, not the existing
   policy-to-command action convention.
4. For each tracker, choose its `triangle` and logical `node` in that triangle.
   A shared node belongs to two triangles, so both names are needed. The mounting
   model assumes a tracker rigidly attached to that triangle's roller/node housing,
   following the corner bisector and triangle normal as in the mechanism model;
   a freely rotating connector-ball mount needs its own orientation model.
5. Measure `position_m` and `rotation_matrix` for the tracker relative to the selected
   node's housing frame: origin at that physical roller, x along the interior corner
   bisector, z along the triangle's normal in cyclic file order, y=z×x. This frame
   follows the housing's changing corner angle rather than a single tube edge.
   `rotation_matrix` maps tracker-local axes into that frame. Identity and zero are
   valid only when physically measured; they are not supplied as mount defaults.
6. Check the proper right-handed `steamvr_to_policy_matrix` copied from the lab
   definition. Full orientation fusion rejects reflections (determinant −1).

Capture the complete tracker pose while holding the robot stationary in a known
reference alignment:

```bash
python scripts/vive_tracker_calibration_logger.py
```

The logger uses SteamVR only and never opens motor serial. Each `Full pose JSON:`
line contains the tracker serial, position in metres, and full-precision 3×3
`rotation_matrix` mapping tracker-local vectors into the SteamVR standing frame.
Only connected, valid, `Running_OK` tracker poses are printed. Copy one JSON line
for each mounted tracker, together with the known housing-to-world alignment.
The separate `Forward direction` readout is only the tracker's local −Z direction
expressed in that world frame; it is not a complete orientation or Euler angles.
Ctrl+C exits and closes OpenVR.

Inspect the exact nominal geometry, incidence matrix, serial order, and every
possible mounting frame without opening hardware or loading a policy:

```bash
python scripts/inspect_encoder_geometry.py > /tmp/encoder_geometry.json
```

It uses the existing MuJoCo preset, graph definition, scale, and physical parameters.
Review `connector_rod_length`, node sizes, and nominal tube lengths against measured
hardware. These simulation defaults are not a physical calibration.

After calibration, use the same checkpoint and architecture/feature overrides as
the previous inference run:

```bash
python sac/real_robot_infer.py --config-name inference/encoder_vive \
  model=/path/to/existing/final.pt \
  encoder_calibration_file=/path/to/measured_calibration.json \
  command_transport=serial serial_port=/dev/ttyUSB0 \
  record_output=/path/to/real_robot_session.jsonl
```

The default transport remains `print`. Print-only encoder mode requires
`encoder_input_file=/path/to/sensors.jsonl` and opens neither serial nor SteamVR.
Each input line contains:

```json
{"encoder_sample":{"ticks":[0,0,0,0,0,0,0,0],"sequence":1,"receive_age_s":0,"freshness_verified":false},"poses_by_serial":{"LHR-2BEE7775":{"position":[0,0,0],"rotation_matrix":[[1,0,0],[0,1,0],[0,0,1]]}}}
```

Supply both calibrated tracker poses for startup. Subsequent lines may omit one
or both. The example above illustrates the schema, not a usable calibration.
Use `state_preview=false operator_keyboard=false` for a headless replay.

## Shape and global pose

The shape solver uses the generated model's physical roller locations, logical
connector balls, tube incidence, and full reference offsets. For each triangle,
it represents each connector offset in the corner's bisector/tangent/normal frame.
Those components stay fixed as the triangle deforms. This preserves the generated
reference exactly, including its departures from ideal planar bisectors. It is an
offset-aware geometric estimator, not a copy of the reference repository's planar
solver or an exact reconstruction of compliant MuJoCo dynamics.

Signed encoder displacement changes the two independent segments through the same
directed incidence used by MuJoCo's node controller; the third follows tube-perimeter
conservation. A constrained spatial solve recovers the node positions. A six-coordinate
local gauge removes rigid motion; it does not assert fixed feet or a ground-contact
pattern. The solve is warm-started from its last accepted shape. Invalid lengths,
triangle inequalities, excessive residuals, singular configurations, encoder jumps,
and excessive state latency cause a fault and best-effort zero command. The local
geometry can have multiple solutions: a known initial shape and calibrated travel
limits are necessary. Solver residuals alone do not establish physical accuracy.

For each visible tracker, its position **and orientation**, measured mounting
transform, and current reconstructed triangle yield a complete world-from-shape
transform. Mounts change with deformation; they are not fixed points on a rigid
whole-robot body. OpenVR must report connected, valid, `Running_OK` poses.

When both trackers are visible, translation and orientation are fused using separate
inverse variances in m² and rad². Orientation averaging stays on SO(3), including
quaternion sign/±180° cases. Rolling second-difference jitter estimates remove
constant-velocity motion and use variance floors and initial priors. Window, floors,
priors, and reacquisition interval are configurable and logged. These are empirical
jitter weights, **not** calibrated absolute-error covariances; acceleration and
shared SteamVR errors can affect both. A stable biased tracker can still be wrong.

One visible tracker supplies the transform and raises a warning. With neither visible,
the last global transform is held, encoder shape updates continue, and control does
not stop because of tracker loss. The viewer/log mark the global pose as unobserved
and show anchor age. Global translation/rotation during that interval is not observable
from encoders. Initially there is no transform to hold: the default waits for both
trackers, sends only zero commands, and times out after 30 seconds. Startup count and
timeout are configurable. Changes in the visible anchor set reset the observation
velocity difference for one frame to avoid treating reference jumps as robot velocity.

## Pause, preview, and stop

- **Ctrl+P** in the focused terminal or preview toggles pause. The preview also has
  a Pause/resume button. Pausing commands zero velocity, continues sensor acquisition,
  shape/pose estimation, the preview, and JSONL logging. It does not command active
  position holding. Resume uses a newly sampled state rather than a queued old action.
- The existing `VEL_DUR` protocol is used for both movement and repeated paused
  zero-velocity polls. Each normal zero command has the configured command duration;
  Ctrl+C and emergency stop use the immediate zero-duration stop.
- **Ctrl+C** exits with a best-effort immediate zero command. **E** in the terminal
  or preview, or the preview's Emergency Stop button, latches a stop and ends the run.
  Ctrl+P cannot resume a latched stop. Starting a new run is required.
- During a blocking serial acknowledgment wait, operator input is polled and a pause
  zero can be queued immediately. The original and zero replies are consumed in order
  before any further normal command. Unknown/partial replies or missing encoder arrays
  time out; active receiver delivery failures invalidate feedback and stop control.
- Shutdown on completion, replay exhaustion, Ctrl+C, operator stop, or a loop error
  sends an immediate zero before closing serial. The software stop uses USB/radio;
  it does not replace an independent physical emergency stop. Firmware command expiry
  is the remaining limit when host/radio delivery fails. Measure actual stop latency.

The preview runs in a separate process with a bounded queue. It shows the exact
global node positions used by the existing observation builder (before centering and
normalization), node labels, shape-frame orientation axes at the centroid, tracker
status, anchor age, unverified encoder status, and pause state. Stale display samples
are marked. Closing the preview warns and leaves terminal controls active. Rendering
does not participate in the control timing loop.

## Firmware and analysis compatibility

Firmware is unchanged. Writing a newline-delimited `VEL_DUR` command makes the
transmitter send radio commands and print a completion line plus encoder array.
One serial reader consumes both; opening another reader on the same USB port would
steal replies. Initial sensing and paused sensing use zero-velocity commands.

The firmware's `lastKnownPositions` array can contain cached readings. An RF write
may succeed without a returned encoder ACK payload, and queued receiver ACKs can
precede the current command. The printout has no per-receiver timestamp, sequence,
or freshness flag. Consequently all live samples are explicitly marked
`encoder_freshness_verified=false`; `encoder_receive_age_s` is host receive age only.
The host's encoder `sequence` identifies received arrays, not hardware sample times.
This limitation is warned and retained by request; it cannot be repaired by Python
parsing. A known encoder reset/discontinuity stops the run and requires recalibration.

Schema version 1 and existing `session`, `tracker_frame`, `command_attempt`,
`command_result`, `emergency_command`, `node_positions`, `rigidity`, normalized actions,
active-channel command strings/ticks, ordering, and graph hashes remain usable by the
existing analysis/replay tools. Additive fields include calibration/model provenance,
raw encoders, fusion weights/variance estimates, pose age, solve residual/conditioning,
pause/operator events, `firmware_command`, and `reconstruction_diagnostics.encoder_vive`.
The configured eight-channel default sends the same active and wire command. Optional
extra firmware slots receive zero, and the complete wire string is retained separately.

Existing analysis commands in `docs/sim_to_real_diagnostics.md` still apply. Health
reports additionally expose `state_estimation` counts. A complete reconstructed-node
frame can have zero visible trackers: legacy `dropout_frames` counts missing node
frames; `held_global_transform_frames` counts unobserved global pose. These must not
be conflated in a paper. Pause intervals remain explicit zero commands in replay.

## Validation before paper measurements

The software tests cover synthetic geometry/fusion, protocol failures, controls,
logging, and replay compatibility. They do not validate physical geometry, encoder
freshness, global accuracy, GUI responsiveness on the lab OS, or emergency-stop latency.
Record stationary noise, slow calibrated single-channel moves in both directions,
one/both-tracker occlusion, reacquisition, pause/resume, radio loss, and stop response
before conducting motion trials. Calibrate all software thresholds on those captures.
Use independent tracked ground truth for node/pose error; the fused anchors are not
independent validation. Retain calibration files, checkpoint identity, logs, and
conditions alongside each trial. Test protocols/results are also recorded in
`My vault/Research/06 Experiment Notebook/2026-11-07-Encoder-Vive-State-Estimation.md`.

## Reference sources

- [Isoperimetric_Kinematics, pinned revision 9c202402](https://github.com/BYUCMR/Isoperimetric_Kinematics/tree/9c20240285e4d06c516c2aee37e7f01254f75180):
  `Python/truss_robot.py` forward kinematics; `Python/truss_robot_offset.py` offset
  constraints and tube incidence; `Python/HardwareControl/send_commands.py` scale
  and the example command sign. The spatial estimator here implements its own
  geometry adapter; no private library source is vendored.
- [TrussRobotFirmware, pinned revision aff95c58](https://github.com/BYUCMR/TrussRobotFirmware/tree/aff95c58f62140fed3fdc31a249ce22e5af9812d):
  `Arduino/src/transmitter.cpp` command/completion/encoder-array protocol and cached
  position handling; `Arduino/src/receiver.cpp` encoder updates, ACK payloads,
  velocity duration, and zero-duration stopping. The actual eight-slot flash was
  confirmed separately by the user and differs from that reference's slot count.
