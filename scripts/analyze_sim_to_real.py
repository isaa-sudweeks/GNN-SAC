#!/usr/bin/env python3
"""Analyze Vive tracker health, paired MuJoCo replays, and repeated real trials."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sim_to_real_io import read_jsonl  # noqa: E402


def _real_capture(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    session = next((record for record in records if record.get("type") == "session"), None)
    if session is None:
        raise ValueError(f"{path} has no session record")
    frames = [record for record in records if record.get("type") == "tracker_frame"]
    complete = [record for record in frames if record.get("status") == "complete"]
    if not complete:
        raise ValueError(f"{path} has no complete tracker frames")
    commands = [record for record in records if record.get("type") == "command_attempt"]
    command_origin = float(commands[0]["relative_time_s"]) if commands else float(complete[0]["relative_time_s"])
    node_order = tuple(session.get("node_order", complete[0].get("node_order", ())))
    positions = np.asarray([frame["node_positions"] for frame in complete], dtype=float)
    times = np.asarray([float(frame["relative_time_s"]) - command_origin for frame in complete])
    pre_command = np.flatnonzero(times <= 0.0)
    if pre_command.size:
        times[pre_command[-1]] = 0.0
    return {
        "path": path,
        "records": records,
        "session": session,
        "frames": frames,
        "complete": complete,
        "commands": commands,
        "command_origin": command_origin,
        "node_order": node_order,
        "positions": positions,
        "times": times,
        "rigidity": np.asarray([float(frame.get("rigidity", np.nan)) for frame in complete]),
        "end_reason": next(
            (record.get("reason") for record in reversed(records) if record.get("type") == "session_end"),
            None,
        ),
    }


def _simulation_capture(path: Path) -> dict[str, Any]:
    records = read_jsonl(path)
    session = next((record for record in records if record.get("type") == "replay_session"), None)
    if session is None:
        raise ValueError(f"{path} has no replay_session record")
    frames = [record for record in records if record.get("type") == "simulation_frame"]
    if not frames:
        raise ValueError(f"{path} has no simulation frames")
    return {
        "path": path,
        "session": session,
        "frames": frames,
        "node_order": tuple(session["node_order"]),
        "positions": np.asarray([frame["node_positions"] for frame in frames], dtype=float),
        "times": np.asarray([float(frame["simulation_time_s"]) for frame in frames]),
        "rigidity": np.asarray([float(frame.get("rigidity", np.nan)) for frame in frames]),
        "end_reason": next(
            (record.get("reason") for record in reversed(records) if record.get("type") == "replay_end"),
            None,
        ),
    }


def _orientation_angle_degrees(reference: np.ndarray, rotation: np.ndarray) -> float:
    relative = reference.T @ rotation
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def tracker_health_metrics(
    capture: dict[str, Any],
    *,
    velocity_warning: float | None = None,
    acceleration_warning: float | None = None,
) -> dict[str, Any]:
    """Compute stationary tracker and reconstruction metrics without universal grading."""
    frames = capture["frames"]
    complete = capture["complete"]
    all_times = np.asarray([float(frame["relative_time_s"]) for frame in frames], dtype=float)
    complete_times = np.asarray([float(frame["relative_time_s"]) for frame in complete], dtype=float)
    intervals = np.diff(all_times)
    skipped = [frame.get("status") != "complete" for frame in frames]
    longest_dropout = current = 0
    for missing in skipped:
        current = current + 1 if missing else 0
        longest_dropout = max(longest_dropout, current)

    raw_by_serial: dict[str, list[tuple[float, dict]]] = {}
    for frame in frames:
        timestamp = float(frame["relative_time_s"])
        for serial, pose in frame.get("raw_poses_by_serial", {}).items():
            if pose.get("position") is not None:
                raw_by_serial.setdefault(str(serial), []).append((timestamp, pose))

    tracker_metrics = {}
    for serial, samples in sorted(raw_by_serial.items()):
        times = np.asarray([sample[0] for sample in samples], dtype=float)
        positions = np.asarray([sample[1]["position"] for sample in samples], dtype=float)
        deltas = np.diff(positions, axis=0)
        dt = np.diff(times)
        valid_dt = dt > 0.0
        velocity_vectors = deltas[valid_dt] / dt[valid_dt, None] if np.any(valid_dt) else np.empty((0, 3))
        velocity = np.linalg.norm(velocity_vectors, axis=1)
        acceleration = np.asarray([])
        if velocity_vectors.shape[0] > 1 and np.all(valid_dt):
            acceleration_dt = 0.5 * (dt[:-1] + dt[1:])
            acceleration = np.linalg.norm(
                np.diff(velocity_vectors, axis=0) / acceleration_dt[:, None], axis=1
            )
        rotations = [sample[1].get("rotation_matrix") for sample in samples]
        valid_rotations = [np.asarray(value, dtype=float) for value in rotations if value is not None]
        angular = (
            [_orientation_angle_degrees(valid_rotations[0], value) for value in valid_rotations]
            if valid_rotations
            else []
        )
        tracker_metrics[serial] = {
            "samples": len(samples),
            "missing_samples": len(frames) - len(samples),
            "visible_percentage": 100.0 * len(samples) / max(len(frames), 1),
            "position_std_xyz_m": np.std(positions, axis=0, ddof=0).tolist(),
            "position_std_norm_m": float(np.linalg.norm(np.std(positions, axis=0, ddof=0))),
            "drift_from_initial_m": float(np.linalg.norm(positions[-1] - positions[0])),
            "max_step_jump_m": float(np.max(np.linalg.norm(deltas, axis=1))) if deltas.size else 0.0,
            "max_velocity_m_s": float(np.max(velocity)) if velocity.size else 0.0,
            "max_acceleration_m_s2": float(np.max(acceleration)) if acceleration.size else 0.0,
            "velocity_warning_count": (
                int(np.count_nonzero(velocity > velocity_warning))
                if velocity_warning is not None else None
            ),
            "acceleration_warning_count": (
                int(np.count_nonzero(acceleration > acceleration_warning))
                if acceleration_warning is not None else None
            ),
            "orientation_std_degrees": float(np.std(angular)) if angular else None,
            "orientation_max_deviation_degrees": float(np.max(angular)) if angular else None,
        }
    for serial in capture["session"].get("serial_to_tracker_id", {}):
        if str(serial) not in tracker_metrics:
            tracker_metrics[str(serial)] = {
                "samples": 0,
                "missing_samples": len(frames),
                "visible_percentage": 0.0,
                "position_std_xyz_m": None,
                "position_std_norm_m": None,
                "drift_from_initial_m": None,
                "max_step_jump_m": None,
                "max_velocity_m_s": None,
                "max_acceleration_m_s2": None,
                "velocity_warning_count": None,
                "acceleration_warning_count": None,
                "orientation_std_degrees": None,
                "orientation_max_deviation_degrees": None,
            }

    positions = capture["positions"]
    node_metrics = {}
    for index, name in enumerate(capture["node_order"]):
        values = positions[:, index, :]
        node_metrics[name] = {
            "position_std_xyz_m": np.std(values, axis=0, ddof=0).tolist(),
            "position_std_norm_m": float(np.linalg.norm(np.std(values, axis=0, ddof=0))),
            "drift_from_initial_m": float(np.linalg.norm(values[-1] - values[0])),
        }

    cross_normals, normal_disagreements, reconstruction_rejections = [], [], 0
    for frame in frames:
        diagnostics = frame.get("reconstruction_diagnostics", {})
        for triangle in diagnostics.get("triangles", {}).values():
            normal_disagreements.append(float(triangle.get("max_normal_disagreement_degrees", 0.0)))
        for joint in diagnostics.get("joints", {}).values():
            cross_normals.append(float(joint.get("cross_normal_magnitude", np.nan)))
        if frame.get("status") != "complete" and (
            "parallel" in str(frame.get("error", "")).lower()
            or "plane" in str(frame.get("error", "")).lower()
        ):
            reconstruction_rejections += 1

    duration = float(all_times[-1] - all_times[0]) if all_times.size > 1 else 0.0
    return {
        "capture": str(capture["path"]),
        "expected_frames": len(frames),
        "valid_frames": len(complete),
        "dropout_frames": len(frames) - len(complete),
        "dropout_percentage": 100.0 * (len(frames) - len(complete)) / max(len(frames), 1),
        "longest_dropout_burst_frames": longest_dropout,
        "duration_s": duration,
        "effective_sample_rate_hz": (len(frames) - 1) / duration if duration > 0.0 else None,
        "median_interval_s": float(np.median(intervals)) if intervals.size else None,
        "interval_jitter_std_s": float(np.std(intervals, ddof=0)) if intervals.size else None,
        "complete_interval_jitter_std_s": (
            float(np.std(np.diff(complete_times), ddof=0)) if complete_times.size > 1 else None
        ),
        "trackers": tracker_metrics,
        "reconstructed_nodes": node_metrics,
        "reconstruction": {
            "rejected_frames": reconstruction_rejections,
            "min_cross_normal_magnitude": float(np.nanmin(cross_normals)) if cross_normals else None,
            "max_normal_disagreement_degrees": max(normal_disagreements, default=None),
        },
        "warning_thresholds": {
            "velocity_m_s": velocity_warning,
            "acceleration_m_s2": acceleration_warning,
        },
    }


def kabsch_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return proper rotation and translation mapping row-vector source to target."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Kabsch inputs must have matching (nodes, 3) shape")
    if source.shape[0] < 3 or np.linalg.matrix_rank(source - source.mean(axis=0)) < 2:
        raise ValueError("Rigid alignment requires at least three non-collinear nodes")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _interpolate_positions(times: np.ndarray, positions: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    output = np.empty((len(target_times), positions.shape[1], 3), dtype=float)
    for node in range(positions.shape[1]):
        for axis in range(3):
            output[:, node, axis] = np.interp(target_times, times, positions[:, node, axis])
    return output


def paired_metrics(real: dict[str, Any], simulation: dict[str, Any]) -> dict[str, Any]:
    if real["node_order"] != simulation["node_order"]:
        raise ValueError("Real and simulation controller-node order must match exactly")
    start = max(float(np.min(real["times"])), float(np.min(simulation["times"])), 0.0)
    end = min(float(np.max(real["times"])), float(np.max(simulation["times"])))
    mask = (real["times"] >= start) & (real["times"] <= end)
    times = real["times"][mask]
    if times.size < 2:
        raise ValueError("Paired comparison requires at least two shared complete frames")
    real_positions = real["positions"][mask]
    simulation_positions = _interpolate_positions(
        simulation["times"], simulation["positions"], times
    )
    rotation, translation = kabsch_transform(simulation_positions[0], real_positions[0])
    aligned = simulation_positions @ rotation.T + translation
    initial_real_centered = real_positions[0] - np.mean(real_positions[0], axis=0)
    initial_sim_centered = simulation_positions[0] - np.mean(simulation_positions[0], axis=0)
    initial_real_radius = float(np.sqrt(np.mean(np.sum(initial_real_centered**2, axis=1))))
    initial_sim_radius = float(np.sqrt(np.mean(np.sum(initial_sim_centered**2, axis=1))))
    if initial_sim_radius <= np.finfo(float).eps:
        raise ValueError("Cannot diagnose scale from a zero-size initial simulation pose")
    vector_error = real_positions - aligned
    node_error = np.linalg.norm(vector_error, axis=2)
    mean_error = np.mean(node_error, axis=1)
    error_variance = np.var(node_error, axis=1, ddof=0)
    real_com = np.mean(real_positions, axis=1)
    sim_com = np.mean(aligned, axis=1)
    com_error = np.linalg.norm(real_com - sim_com, axis=1)
    real_shape = real_positions - real_com[:, None, :]
    sim_shape = aligned - sim_com[:, None, :]
    shape_error = np.sqrt(np.mean(np.sum((real_shape - sim_shape) ** 2, axis=2), axis=1))

    edge_pairs = []
    node_index = {name: index for index, name in enumerate(real["node_order"])}
    for edge in real["session"].get("edges", []):
        if edge[0] in node_index and edge[1] in node_index:
            edge_pairs.append((node_index[edge[0]], node_index[edge[1]]))
    edge_error = np.zeros(len(times))
    if edge_pairs:
        real_lengths = np.stack(
            [np.linalg.norm(real_positions[:, i] - real_positions[:, j], axis=1) for i, j in edge_pairs],
            axis=1,
        )
        sim_lengths = np.stack(
            [np.linalg.norm(aligned[:, i] - aligned[:, j], axis=1) for i, j in edge_pairs],
            axis=1,
        )
        edge_error = np.sqrt(np.mean((real_lengths - sim_lengths) ** 2, axis=1))

    real_rigidity = real["rigidity"][mask]
    sim_rigidity = np.interp(times, simulation["times"], simulation["rigidity"])
    rigidity_error = np.abs(real_rigidity - sim_rigidity)

    per_node = {
        name: {
            "rmse_m": float(np.sqrt(np.mean(node_error[:, index] ** 2))),
            "max_error_m": float(np.max(node_error[:, index])),
        }
        for index, name in enumerate(real["node_order"])
    }
    real_end = real.get("end_reason")
    simulation_end = simulation.get("end_reason")
    normalized_real_end = "complete" if real_end == "control_steps_complete" else real_end
    return {
        "times": times,
        "node_order": real["node_order"],
        "real_positions": real_positions,
        "simulation_positions_aligned": aligned,
        "node_error": node_error,
        "mean_error": mean_error,
        "error_variance": error_variance,
        "com_error": com_error,
        "shape_error": shape_error,
        "edge_length_rmse": edge_error,
        "rigidity_error": rigidity_error,
        "summary": {
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "alignment_uses_scale_fitting": False,
            "initial_characteristic_radius_real_m": initial_real_radius,
            "initial_characteristic_radius_simulation_m": initial_sim_radius,
            "initial_scale_ratio_real_to_simulation": initial_real_radius / initial_sim_radius,
            "initial_pose_rmse_before_alignment_m": float(
                np.sqrt(np.mean(np.sum((simulation_positions[0] - real_positions[0]) ** 2, axis=1)))
            ),
            "aggregate_rmse_m": float(np.sqrt(np.mean(node_error**2))),
            "mean_com_error_m": float(np.mean(com_error)),
            "mean_shape_rmse_m": float(np.mean(shape_error)),
            "initial_shape_rmse_after_rigid_alignment_m": float(shape_error[0]),
            "mean_edge_length_rmse_m": float(np.mean(edge_error)),
            "mean_rigidity_error": (
                float(np.nanmean(rigidity_error))
                if np.any(np.isfinite(rigidity_error)) else None
            ),
            "termination": {
                "real": real_end,
                "simulation": simulation_end,
                "different": normalized_real_end != simulation_end,
            },
            "per_node": per_node,
        },
    }


def repeated_trial_metrics(captures: list[dict[str, Any]]) -> dict[str, Any]:
    first = captures[0]
    first_hash = first["session"].get("graph_definition_sha256")
    first_commands = [record.get("command") for record in first["commands"]]
    for capture in captures[1:]:
        if capture["session"].get("graph_definition_sha256") != first_hash:
            raise ValueError("Repeated trials must have identical graph hashes")
        if capture["node_order"] != first["node_order"]:
            raise ValueError("Repeated trials must have identical node order")
        if [record.get("command") for record in capture["commands"]] != first_commands:
            raise ValueError("Repeated trials must have identical command sequences")
    start = max(max(float(np.min(capture["times"])), 0.0) for capture in captures)
    end = min(float(np.max(capture["times"])) for capture in captures)
    reference_times = first["times"][(first["times"] >= start) & (first["times"] <= end)]
    if reference_times.size < 2:
        raise ValueError("Repeated trials have no usable shared time interval")
    stacked = np.stack(
        [_interpolate_positions(capture["times"], capture["positions"], reference_times) for capture in captures]
    )
    mean_positions = np.mean(stacked, axis=0)
    deviations = np.linalg.norm(stacked - mean_positions[None, ...], axis=3)
    return {
        "times": reference_times,
        "node_order": first["node_order"],
        "mean_positions": mean_positions,
        "position_variance_xyz": np.var(stacked, axis=0, ddof=0),
        "deviation_mean": np.mean(deviations, axis=0),
        "deviation_variance": np.var(deviations, axis=0, ddof=0),
        "trial_count": len(captures),
    }


def _plot_health(capture: dict[str, Any], metrics: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    times = capture["times"]
    positions = capture["positions"]
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].plot(times, np.mean(positions, axis=1))
    axes[0, 0].set_title("Mean reconstructed node position")
    axes[0, 0].set_ylabel("m")
    axes[0, 0].legend(("x", "y", "z"))
    for index, name in enumerate(capture["node_order"]):
        drift = np.linalg.norm(positions[:, index] - positions[0, index], axis=1)
        axes[0, 1].plot(times, drift, label=name)
    axes[0, 1].set_title("Reconstructed-node drift from first frame")
    axes[0, 1].set_ylabel("m")
    axes[0, 1].legend(fontsize=7, ncol=2)
    frame_times = [float(frame["relative_time_s"]) for frame in capture["frames"]]
    valid = [1 if frame.get("status") == "complete" else 0 for frame in capture["frames"]]
    axes[1, 0].step(frame_times, valid, where="post")
    axes[1, 0].set_title("Complete tracker frames")
    axes[1, 0].set_yticks((0, 1), ("missing", "complete"))
    cross = []
    for frame in capture["frames"]:
        values = [
            float(joint.get("cross_normal_magnitude", np.nan))
            for joint in frame.get("reconstruction_diagnostics", {}).get("joints", {}).values()
        ]
        cross.append(min(values) if values else np.nan)
    axes[1, 1].plot(frame_times, cross)
    axes[1, 1].set_title("Minimum plane cross-normal magnitude")
    for axis in axes.flat:
        axis.set_xlabel("time (s)")
        axis.grid(alpha=0.25)
    figure.suptitle(
        f"Tracker health: {metrics['valid_frames']}/{metrics['expected_frames']} complete frames"
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_comparison(result: dict[str, Any], output: Path, com_output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for index, name in enumerate(result["node_order"]):
        axes[0].plot(result["times"], result["node_error"][:, index], label=name, alpha=0.7)
    axes[0].plot(result["times"], result["mean_error"], color="black", linewidth=2.5, label="mean")
    axes[0].set_ylabel("position error (m)")
    axes[0].legend(fontsize=7, ncol=3)
    axes[1].plot(result["times"], result["error_variance"], color="tab:purple")
    axes[1].set_ylabel("cross-node error variance (m²)")
    axes[1].set_xlabel("command-relative time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.savefig(output, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    axis.plot(result["times"], result["com_error"], label="COM trajectory error")
    axis.plot(result["times"], result["shape_error"], label="COM-relative shape RMSE")
    axis.plot(result["times"], result["edge_length_rmse"], label="edge-length RMSE")
    if np.any(np.isfinite(result["rigidity_error"])):
        axis.plot(result["times"], result["rigidity_error"], label="normalized rigidity error")
    axis.set(xlabel="command-relative time (s)", ylabel="error (m)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(com_output, dpi=160)
    plt.close(figure)


def _plot_repeated(result: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(12, 6), constrained_layout=True)
    for index, name in enumerate(result["node_order"]):
        mean = result["deviation_mean"][:, index]
        std = np.sqrt(result["deviation_variance"][:, index])
        axis.plot(result["times"], mean, label=name)
        axis.fill_between(result["times"], np.maximum(0.0, mean - std), mean + std, alpha=0.15)
    axis.set(
        xlabel="command-relative time (s)",
        ylabel="distance from across-trial mean (m)",
        title=f"Physical/tracker repeatability across {result['trial_count']} trials",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=3)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _edge_indices(session: dict, node_order: tuple[str, ...]) -> list[tuple[int, int]]:
    index = {name: i for i, name in enumerate(node_order)}
    seen = set()
    result = []
    for edge in session.get("edges", []):
        if edge[0] not in index or edge[1] not in index:
            continue
        pair = tuple(sorted((index[edge[0]], index[edge[1]])))
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def _write_3d_video(
    path: Path,
    times: np.ndarray,
    node_order: tuple[str, ...],
    primary: np.ndarray,
    edges: list[tuple[int, int]],
    *,
    secondary: np.ndarray | None = None,
    fps: int = 15,
    label_primary: str = "Vive",
    label_secondary: str = "MuJoCo",
    frame_status: list[str] | None = None,
    frame_metric: np.ndarray | None = None,
    metric_label: str = "metric",
) -> None:
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    all_positions = primary if secondary is None else np.concatenate((primary, secondary), axis=0)
    low = np.min(all_positions, axis=(0, 1))
    high = np.max(all_positions, axis=(0, 1))
    center = (low + high) / 2.0
    radius = max(float(np.max(high - low)) / 2.0, 1e-3) * 1.1
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(node_order)))
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7) as writer:
        for frame_index, timestamp in enumerate(times):
            figure = plt.figure(figsize=(8, 7))
            axis = figure.add_subplot(111, projection="3d")
            for i, j in edges:
                axis.plot(*primary[frame_index, [i, j]].T, color="tab:blue", alpha=0.55)
                if secondary is not None:
                    axis.plot(*secondary[frame_index, [i, j]].T, color="tab:orange", linestyle="--", alpha=0.55)
            for index, name in enumerate(node_order):
                point = primary[frame_index, index]
                axis.scatter(*point, color=colors[index], marker="o", s=45)
                axis.text(*point, name, fontsize=7)
                if secondary is not None:
                    axis.scatter(*secondary[frame_index, index], color=colors[index], marker="x", s=45)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_box_aspect((1, 1, 1))
            subtitle = label_primary if secondary is None else f"o={label_primary}, x={label_secondary}"
            status = frame_status[frame_index] if frame_status is not None else "complete"
            status_text = " | TRACKER FRAME SKIPPED" if status != "complete" else ""
            metric_text = ""
            if frame_metric is not None:
                metric_text = f" | {metric_label}={frame_metric[frame_index]:.4f} m"
            axis.set_title(f"{subtitle} | t={timestamp:.3f} s{metric_text}{status_text}")
            axis.set_xlabel("x (m)")
            axis.set_ylabel("y (m)")
            axis.set_zlabel("z (m)")
            figure.canvas.draw()
            frame = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
            writer.append_data(frame)
            plt.close(figure)


def _write_health_csv(metrics: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("kind", "name", "metric", "value"))
        for kind, key in (("tracker", "trackers"), ("node", "reconstructed_nodes")):
            for name, values in metrics[key].items():
                for metric, value in values.items():
                    writer.writerow((kind, name, metric, json.dumps(value)))


def _write_comparison_csv(result: dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "node", "real_x", "real_y", "real_z", "sim_x", "sim_y", "sim_z", "error_m"))
        for time_index, timestamp in enumerate(result["times"]):
            for node_index, name in enumerate(result["node_order"]):
                writer.writerow((
                    float(timestamp), name,
                    *result["real_positions"][time_index, node_index].tolist(),
                    *result["simulation_positions_aligned"][time_index, node_index].tolist(),
                    float(result["node_error"][time_index, node_index]),
                ))


def _ensure_output(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_health(args: argparse.Namespace) -> None:
    output = _ensure_output(args.output_dir)
    capture = _real_capture(args.tracker_log)
    metrics = tracker_health_metrics(
        capture,
        velocity_warning=args.velocity_warning,
        acceleration_warning=args.acceleration_warning,
    )
    (output / "tracker_health.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    _write_health_csv(metrics, output / "tracker_health.csv")
    _plot_health(capture, metrics, output / "tracker_health.png")
    video_times, video_positions, video_status = [], [], []
    last_positions = None
    for frame in capture["frames"]:
        if frame.get("status") == "complete":
            last_positions = np.asarray(frame["node_positions"], dtype=float)
        if last_positions is None:
            continue
        video_times.append(float(frame["relative_time_s"]))
        video_positions.append(last_positions.copy())
        video_status.append(str(frame.get("status", "skipped")))
    _write_3d_video(
        output / "vive_nodes_3d.mp4",
        np.asarray(video_times), capture["node_order"], np.asarray(video_positions),
        _edge_indices(capture["session"], capture["node_order"]), fps=args.fps,
        frame_status=video_status,
    )


def run_compare(args: argparse.Namespace) -> None:
    output = _ensure_output(args.output_dir)
    real = _real_capture(args.tracker_log)
    simulation = _simulation_capture(args.simulation_log)
    result = paired_metrics(real, simulation)
    (output / "comparison_summary.json").write_text(json.dumps(result["summary"], indent=2) + "\n", encoding="utf-8")
    _write_comparison_csv(result, output / "synchronized_positions.csv")
    _plot_comparison(result, output / "node_position_error.png", output / "com_shape_error.png")
    _write_3d_video(
        output / "vive_vs_mujoco_3d.mp4",
        result["times"], result["node_order"], result["real_positions"],
        _edge_indices(real["session"], result["node_order"]),
        secondary=result["simulation_positions_aligned"], fps=args.fps,
        frame_metric=result["mean_error"], metric_label="mean error",
    )


def run_repeat(args: argparse.Namespace) -> None:
    output = _ensure_output(args.output_dir)
    result = repeated_trial_metrics([_real_capture(path) for path in args.tracker_logs])
    summary = {
        "trial_count": result["trial_count"],
        "per_node_mean_variance_m2": {
            name: float(np.mean(np.sum(result["position_variance_xyz"][:, index], axis=1)))
            for index, name in enumerate(result["node_order"])
        },
    }
    (output / "repeatability_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot_repeated(result, output / "repeated_trial_variance.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    health = subparsers.add_parser("health", help="Analyze a stationary Vive capture")
    health.add_argument("--tracker-log", type=Path, required=True)
    health.add_argument("--output-dir", type=Path, required=True)
    health.add_argument("--velocity-warning", type=float)
    health.add_argument("--acceleration-warning", type=float)
    health.add_argument("--fps", type=int, default=15)
    health.set_defaults(function=run_health)

    compare = subparsers.add_parser("compare", help="Compare Vive and MuJoCo positions")
    compare.add_argument("--tracker-log", type=Path, required=True)
    compare.add_argument("--simulation-log", type=Path, required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--fps", type=int, default=15)
    compare.set_defaults(function=run_compare)

    repeat = subparsers.add_parser("repeat", help="Analyze repeated identical real trials")
    repeat.add_argument("--tracker-logs", type=Path, nargs="+", required=True)
    repeat.add_argument("--output-dir", type=Path, required=True)
    repeat.set_defaults(function=run_repeat)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
