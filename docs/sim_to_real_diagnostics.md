# Vive tracker diagnostics and exact MuJoCo replay

This workflow separates three questions: whether SteamVR tracking/reconstruction is
stable, whether firmware commands address the intended physical rollers, and whether
the configured MuJoCo dynamics reproduce the measured motion. A single moving trial
cannot identify those causes by itself, so run the stages below in order.

## 1. Record real inference

Real inference records a versioned JSONL session by default. The file contains every
raw Vive pose that OpenVR returns, reconstructed controller-node positions, skipped
frames, normalized policy actions, quantized ticks, exact firmware strings, transport
outcomes, and the graph definition/hash used for the run.

```bash
python sac/real_robot_infer.py \
  model=/path/to/final.pt \
  serial_map_file=config/tracker_maps/vive_serial_map.json \
  graph_definition_file=/path/to/robot_graph.json \
  record_output=/path/to/real_robot_session.jsonl
```

`command_transport=print` remains the default. Recording does not enable hardware.
Set `record_session=false` only when a capture is intentionally unnecessary.

## 2. Check stationary tracker health

Keep the robot stationary for the entire capture. Then run:

```bash
python scripts/analyze_sim_to_real.py health \
  --tracker-log /path/to/stationary_session.jsonl \
  --output-dir /path/to/stationary_health
```

The report measures frame dropouts and bursts, effective sampling rate and jitter,
per-tracker position/orientation noise, drift, jumps, reconstructed-node stability,
triangle-normal agreement, and plane-intersection conditioning. It also creates a 2D
diagnostic figure and `vive_nodes_3d.mp4`.

There is no universal Vive pass/fail threshold. Establish limits from a known-good
stationary capture, then optionally request warning counts with
`--velocity-warning` and `--acceleration-warning`.

## 3. Replay the exact graph-defined physical routing

Replay requires the same graph definition used by real inference. The selected preset
supplies nominal logical-node coordinates and Hydra physical parameters; the graph
definition replaces the preset's triangle order, passive endpoints, control-node
duplication, tube routing, and serial roller order.

```bash
python sac/replay_real_robot_commands.py \
  input_file=/path/to/real_robot_session.jsonl \
  graph_definition_file=/path/to/robot_graph.json \
  truss_topology=octahedron \
  truss_realistic=true \
  output_file=/path/to/mujoco_replay.jsonl \
  visualize=true
```

The replay aborts if the graph hash, controller nodes, passive mask, tube/connector
edges, routed actuator endpoints, or serial order differs. It applies the transmitted
integer ticks, not the unquantized policy action. A plain file containing one
`VEL_DUR:<ticks>:<duration>` line per command is also accepted.

For recorded JSONL sessions, the recorded velocity limit and transmitter node order
are authoritative. Each command is active only for its encoded duration; later
commands supersede earlier ones, and any interval with no active command is replayed
with zero action. Emergency-command records terminate replay at their recorded time.
Missing or timed-out acknowledgments remain marked as delivery-uncertain because a
timeout does not prove that the firmware failed to receive the command.

## 4. Compare Vive and MuJoCo

```bash
python scripts/analyze_sim_to_real.py compare \
  --tracker-log /path/to/real_robot_session.jsonl \
  --simulation-log /path/to/mujoco_replay.jsonl \
  --output-dir /path/to/comparison
```

The comparison synchronizes by command-relative time, linearly interpolates MuJoCo at
complete Vive timestamps, and computes one fixed proper rigid transform from the first
shared pose. The transform never fits scale: a real robot that is uniformly larger or
smaller than the nominal MuJoCo geometry therefore remains a measurable model mismatch.
Comparison also verifies that the replay was generated from the selected recording and
that its graph hash and command schedule match; unrelated logs are rejected.
The JSON summary reports the initial real-to-simulation characteristic-radius ratio as
a scale diagnostic without applying it. Per-node, COM-relative shape, and edge-length
errors show the consequence over time. The report also includes cross-node error
variance, COM motion, rigidity, initial-pose, and termination differences, with CSV
data, PNG figures, and `vive_vs_mujoco_3d.mp4`.

## 5. Measure repeatability

Run the same command sequence several times and analyze all captures together:

```bash
python scripts/analyze_sim_to_real.py repeat \
  --tracker-logs trial_1.jsonl trial_2.jsonl trial_3.jsonl \
  --output-dir /path/to/repeatability
```

The tool requires identical graph hashes, controller-node order, and command strings.
It reports genuine across-trial position variance separately from the variance across
nodes in one sim-to-real comparison. Stable tracking and repeatable physical motion,
combined with a repeatable simulation bias, are evidence for a model/parameter issue;
poor stationary or repeated-trial stability points to tracking, reconstruction, or
physical repeatability first.
