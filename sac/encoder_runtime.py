"""Encoder/Vive control loop using the existing policy and JSONL contracts."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

from robot_operator import OperatorEmergencyStop
from sim_to_real_io import numpy_value, parse_firmware_command, utc_now_iso


class EncoderReplaySource:
    """Read hardware-free sensor input; never opens a serial port or SteamVR."""

    def __init__(self, path):
        self.stream = Path(path).expanduser().open()

    def read_encoder_frame(self, now):
        from real_robot_infer import TrackerPose

        line = next(self.stream)
        frame = json.loads(line)
        raw = {serial: TrackerPose(np.asarray(pose["position"]), np.asarray(pose["rotation_matrix"]))
               for serial, pose in frame.get("poses_by_serial", {}).items()}
        sample = dict(frame["encoder_sample"])
        sample["received_at"] = now - float(sample.pop("receive_age_s", 0))
        return raw, sample

    def close(self):
        self.stream.close()


def run_encoder_loop(cfg, source, layout, builder, agent, formatter, transport,
                     estimator, controls, recorder=None, preview=None):
    """Keep policy inputs/analysis fields fixed while substituting estimated nodes."""
    frequency = float(cfg.control_frequency_hz)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("control_frequency_hz must be positive and finite")
    zeros = np.zeros((len(layout.node_names), 1), dtype=float)
    pause_command = formatter.velocity_command(zeros)
    stop_command = formatter.emergency_stop_command()
    transport.operator_controls = controls
    transport.pause_command = pause_command
    period = 1 / frequency
    step = 0
    started = time.monotonic()
    next_tick = started
    reason = "error"
    previous_tracking_status = None
    last_revision = controls.revision
    pending_pause = False
    policy_started = False
    previous_dispatch = None
    coverage_warnings = 0

    def record(kind, *, recorded_at=None, **payload):
        if recorder is not None:
            recorder.write(kind, step=step, relative_time_s=recorder.relative_time(recorded_at), **payload)

    def operator_events():
        nonlocal last_revision, pending_pause
        controls.poll()
        if controls.revision != last_revision:
            for state in controls.drain_events():
                record("operator_event", state=state)
                print(f"Operator state: {state}", file=sys.stderr, flush=True)
                if state == "paused":
                    pending_pause = True
            last_revision = controls.revision

    def send(command, action, *, sensor_poll=False, cycle_timing=None):
        nonlocal previous_dispatch, coverage_warnings
        attempt_time = time.monotonic()
        parsed = parse_firmware_command(command, limit=formatter.max_velocity_ticks_per_second)
        timing = dict(cycle_timing or {})
        if previous_dispatch is not None:
            interval = attempt_time - previous_dispatch["time"]
            uncovered = max(0., interval - previous_dispatch["duration_s"])
            risk = (not sensor_poll and not controls.paused and any(parsed.ticks)
                    and previous_dispatch["moving"] and uncovered > 1e-6)
            timing.update(dispatch_interval_s=interval, previous_duration_s=previous_dispatch["duration_s"],
                          host_uncovered_s=uncovered, host_lease_coverage_risk=bool(risk))
            if risk:
                coverage_warnings += 1
                if coverage_warnings == 1 or coverage_warnings % 10 == 0:
                    print(f"WARNING: host command interval {interval:.3f}s exceeds previous "
                          f"duration {previous_dispatch['duration_s']:.3f}s by {uncovered:.3f}s "
                          f"({coverage_warnings} coverage risks). Motors may expire between commands; "
                          "successful ACKs do not verify continuous motion.", file=sys.stderr, flush=True)
        previous_dispatch = dict(time=attempt_time, duration_s=parsed.duration_seconds,
                                 moving=not sensor_poll and not controls.paused and any(parsed.ticks))
        payload = dict(normalized_action=numpy_value(action), graph_node_order=list(formatter.graph_node_names),
                       serial_node_order=list(formatter.serial_node_order), ticks=list(parsed.ticks),
                       duration_s=parsed.duration_seconds, command=command,
                       transport="serial" if transport.acknowledgment_gated else "print",
                       control_state="paused" if controls.paused else "running", timing=timing)
        if hasattr(transport, "wire_command"):
            payload["firmware_command"] = transport.wire_command(command)
        record("sensor_poll" if sensor_poll else "command_attempt", recorded_at=attempt_time, **payload)
        try:
            transport.send(command)
            record("sensor_poll_result" if sensor_poll else "command_result", command=command,
                   status="acknowledged" if transport.acknowledgment_gated else "printed",
                   elapsed_s=time.monotonic() - attempt_time, delivery=getattr(transport, "last_delivery_result", None))
        except BaseException as exc:
            record("sensor_poll_result" if sensor_poll else "command_result", command=command,
                   status="interrupted_or_failed", error=repr(exc),
                   elapsed_s=time.monotonic() - attempt_time, delivery=getattr(transport, "last_delivery_result", None))
            raise
        finally:
            queued = getattr(transport, "last_pause_event", None)
            if queued is not None:
                # Preserve queued stop attempts even if waiting for either reply fails.
                if recorder is not None:
                    recorder.write("command_attempt", step=step,
                                   relative_time_s=recorder.relative_time(queued["sent_at"]),
                                   **dict(payload, command=pause_command, normalized_action=zeros.tolist(),
                                          ticks=[0] * len(formatter.serial_node_order), control_state="paused",
                                          timing={},
                                          firmware_command=transport.wire_command(pause_command)))
                    recorder.write("command_result", step=step, relative_time_s=recorder.relative_time(),
                                   command=pause_command, status="acknowledged" if queued["delivery"] is not None else "unconfirmed",
                                   delivery=queued["delivery"])
                transport.last_pause_event = None
                previous_dispatch = dict(time=queued["sent_at"], duration_s=parsed.duration_seconds, moving=False)

    try:
        controls.drain_events()
        if formatter.duration_seconds <= period:
            print(f"WARNING: command duration {formatter.duration_seconds:.3f}s has no overlap at "
                  f"{frequency:g} Hz; computation or radio jitter can cause motor expiry gaps.",
                  file=sys.stderr, flush=True)
        record("operator_event", state="paused" if controls.paused else "running", reason="initial")
        if transport.acknowledgment_gated:
            send(pause_command, zeros, sensor_poll=True)
        while cfg.control_steps is None or step < int(cfg.control_steps):
            while time.monotonic() < next_tick:
                operator_events()
                if pending_pause:
                    break
                time.sleep(min(.01, max(0, next_tick - time.monotonic())))
            operator_events()
            if pending_pause:
                send(pause_command, zeros)
                pending_pause = False
            now = time.monotonic()
            cycle_lateness = max(0., now - next_tick)
            if hasattr(source, "read_encoder_frame"):
                raw, encoder = source.read_encoder_frame(now)
            else:
                raw = source.poses_by_serial()
                encoder = transport.latest_encoder_sample
            raw = {serial: pose for serial, pose in raw.items() if serial in layout.serial_to_tracker_id}
            positions, health = estimator.sample(encoder, raw, layout.serial_to_tracker_id, now)
            if positions is None:
                record("tracker_frame", status="waiting_for_initial_pose", sampled_at_utc=utc_now_iso(),
                       raw_poses_by_serial=_serialize(raw), reconstruction_diagnostics={"encoder_vive": health})
                if now - started > float(cfg.tracker_startup_timeout_s):
                    raise TimeoutError("Timed out waiting for calibrated initial Vive anchors")
                if transport.acknowledgment_gated:
                    send(pause_command, zeros, sensor_poll=True)
                next_tick = max(now + period, time.monotonic())
                continue
            tracking_status = (health["status"], tuple(health["visible_trackers"]))
            if previous_tracking_status is not None and tracking_status != previous_tracking_status:
                # Reacquisition and changing anchor sets can change the pose
                # reference abruptly; never turn that jump into robot velocity.
                builder.previous_positions = None
                builder.previous_time = None
                builder.filtered_velocity = None
                health["velocity_reset_reason"] = "tracker_visibility_transition"
            previous_tracking_status = tracking_status
            observation = builder.build(positions, now)
            record("tracker_frame", status="complete", sampled_at_utc=utc_now_iso(),
                   raw_poses_by_serial=_serialize(raw), node_order=list(layout.node_names),
                   node_positions=numpy_value(positions),
                   rigidity=float(observation.rigidity.detach().cpu().numpy()[0]),
                   reconstruction_diagnostics={"encoder_vive": health}, paused=controls.paused)
            if preview is not None:
                preview.publish(positions, health, controls.paused)
            if controls.paused:
                action = zeros
            else:
                action = agent.act(observation, t0=not policy_started, eval_mode=bool(cfg.deterministic)).detach().cpu().numpy()
                policy_started = True
            operator_events()
            if controls.paused or pending_pause:
                action = zeros
                pending_pause = False
            if time.monotonic() - now > float(cfg.encoder_max_state_latency_s):
                raise TimeoutError("Estimated policy input exceeded encoder_max_state_latency_s")
            send_time = time.monotonic()
            send(formatter.velocity_command(action), action,
                 cycle_timing=dict(cycle_compute_s=send_time - now, cycle_lateness_s=cycle_lateness,
                                   target_period_s=period))
            step += 1
            # A late cycle never sends a burst of catch-up motor commands.
            # Include reconstruction, inference and ACK time in the period.
            # Starting the period at dispatch adds the next cycle's compute time.
            next_tick = max(now + period, time.monotonic())
        reason = "control_steps_complete"
    except StopIteration:
        reason = "sensor_replay_complete"
    except KeyboardInterrupt:
        reason = "ctrl_c"
    except OperatorEmergencyStop:
        reason = "operator_emergency_stop"
    except Exception as exc:
        record("fault", error=repr(exc))
        raise
    finally:
        transport.emergency_stop(stop_command, reason=reason)
        record("emergency_command", command=stop_command, reason=reason, delivery="unconfirmed_best_effort")
    return reason


def _serialize(raw):
    return {serial: {"position": np.asarray(pose.position).tolist(), "rotation_matrix": np.asarray(pose.rotation_matrix).tolist()}
            for serial, pose in raw.items()}
