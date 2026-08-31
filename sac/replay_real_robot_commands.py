"""Replay firmware-compatible real-robot commands in an exact routed MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import hydra
import numpy as np
from omegaconf import OmegaConf

from env.mujoco_gen.topology_envs import (
    MujocoPresetGraphEnv,
    _fixed_model_scale,
    _physical_parameters_from_config,
    resolve_truss_realistic,
    resolve_truss_topology,
)
from mujoco_truss_gen import get_mujoco_spec, get_node_features, get_preset_definition
from mujoco_truss_gen.mujoco_model.control_graph import control_graph_metadata_from_xml
from sim_to_real_io import (
    SCHEMA_VERSION,
    canonical_json,
    FirmwareCommand,
    JsonlWriter,
    load_triangle_graph_definition,
    parse_firmware_command,
    read_jsonl,
    utc_now_iso,
)


@dataclass(frozen=True)
class ScheduledCommand:
    start_time_s: float
    duration_s: float
    command: FirmwareCommand
    delivery_status: str | None = None
    delivery_uncertain: bool = False


def _latest_started_index(
    scheduled: list[ScheduledCommand], simulation_time_s: float, start_index: int = -1
) -> int:
    """Return the latest event begun by this time; equal timestamps use file order."""
    index = start_index
    while index + 1 < len(scheduled) and (
        scheduled[index + 1].start_time_s <= simulation_time_s + 1e-12
    ):
        index += 1
    return index


def _schedule_end_time(scheduled: list[ScheduledCommand]) -> float:
    """Return when the final effective command or emergency event ends."""
    if not scheduled:
        raise ValueError("Replay schedule must contain at least one command")
    final = scheduled[-1]
    return final.start_time_s + final.duration_s


def _recorded_limit(session: dict) -> int:
    if "max_velocity_ticks_per_second" not in session:
        raise ValueError("Recorded session has no max_velocity_ticks_per_second")
    raw_limit = session["max_velocity_ticks_per_second"]
    if isinstance(raw_limit, bool):
        raise ValueError("Recorded max_velocity_ticks_per_second must be a positive integer")
    try:
        limit = int(raw_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Recorded max_velocity_ticks_per_second must be a positive integer"
        ) from exc
    if limit <= 0 or float(raw_limit) != float(limit):
        raise ValueError("Recorded max_velocity_ticks_per_second must be a positive integer")
    return limit


def _load_commands(
    path: Path, *, limit: int
) -> tuple[list[ScheduledCommand], dict, int]:
    text = path.expanduser().read_text(encoding="utf-8")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if first.startswith("{"):
        records = read_jsonl(path)
        if not records or records[0].get("type") != "session":
            raise ValueError("Recorded session must begin with exactly one session record")
        sessions = [record for record in records if record.get("type") == "session"]
        if len(sessions) != 1:
            raise ValueError("Recorded session must begin with exactly one session record")
        if any(record.get("schema_version") != SCHEMA_VERSION for record in records):
            raise ValueError(
                f"Recorded session contains an unsupported schema version; expected {SCHEMA_VERSION}"
            )
        session = dict(sessions[0])
        session["_source_recording_sha256"] = hashlib.sha256(
            canonical_json(records).encode("utf-8")
        ).hexdigest()
        recorded_limit = _recorded_limit(session)
        results = [record for record in records if record.get("type") == "command_result"]
        result_by_attempt = {
            (record.get("step"), record.get("command")): record
            for record in results
        }
        event_records = [
            record
            for record in records
            if record.get("type") in {"command_attempt", "emergency_command"}
        ]
        scheduled = []
        previous_start_time = None
        for order, record in enumerate(event_records):
            try:
                start_time = float(record["relative_time_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Recorded command has no valid relative_time_s") from exc
            if not np.isfinite(start_time) or start_time < 0.0:
                raise ValueError("Recorded command relative_time_s must be finite and nonnegative")
            if previous_start_time is not None and start_time < previous_start_time:
                raise ValueError("Recorded command times must be nondecreasing")
            previous_start_time = start_time
            command = parse_firmware_command(str(record["command"]), limit=recorded_limit)
            if record.get("type") == "emergency_command":
                status = "emergency_stop"
                uncertain = False
            else:
                if "ticks" in record:
                    raw_ticks = record["ticks"]
                    ticks_are_integers = isinstance(raw_ticks, list) and all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in raw_ticks
                    )
                    if not ticks_are_integers or tuple(raw_ticks) != command.ticks:
                        raise ValueError("Recorded command ticks do not match command text")
                if "duration_s" in record:
                    if isinstance(record["duration_s"], bool):
                        raise ValueError(
                            "Recorded command duration_s does not match command text"
                        )
                    try:
                        recorded_duration = float(record["duration_s"])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(
                            "Recorded command duration_s does not match command text"
                        ) from exc
                    if (
                        not np.isfinite(recorded_duration)
                        or recorded_duration != command.duration_seconds
                    ):
                        raise ValueError(
                            "Recorded command duration_s does not match command text"
                        )
                if (
                    "serial_node_order" in record
                    and record["serial_node_order"] != session.get("serial_node_order")
                ):
                    raise ValueError(
                        "Recorded command serial_node_order does not match session"
                    )
                result = result_by_attempt.get((record.get("step"), record.get("command")))
                status = str(result.get("status")) if result is not None else "missing"
                uncertain = status not in {"acknowledged", "printed"}
            scheduled.append((start_time, order, command, status, uncertain))
        origin = scheduled[0][0] if scheduled else 0.0
        commands = [
            ScheduledCommand(
                start_time - origin,
                command.duration_seconds,
                command,
                status,
                uncertain,
            )
            for start_time, _, command, status, uncertain in scheduled
        ]
        emergency_index = next(
            (index for index, item in enumerate(commands) if item.command.emergency_stop),
            None,
        )
        if emergency_index is not None:
            commands = commands[: emergency_index + 1]
        if not commands:
            raise ValueError("Recorded session contains no command or emergency records")
        return commands, session, recorded_limit

    scheduled = []
    current_time = 0.0
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            command = parse_firmware_command(line, limit=limit)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        scheduled.append(ScheduledCommand(current_time, command.duration_seconds, command))
        if command.emergency_stop:
            break
        current_time += command.duration_seconds
    if not scheduled:
        raise ValueError("Command file contains no commands")
    return scheduled, {}, limit


def _assert_graph_parity(graph, metadata, session: dict) -> None:
    if tuple(metadata.control_node_names) != graph.control_node_names:
        raise ValueError("MuJoCo control-node order does not match graph definition")
    if tuple(metadata.passive_control_node_names) != graph.passive_control_node_names:
        raise ValueError("MuJoCo passive-node placement does not match graph definition")
    model_edges = {
        (*sorted((edge.from_node, edge.to_node)), edge.type)
        for edge in metadata.edges
    }
    expected_edges = {
        (*sorted((source, target)), "actuated" if role == "tube" else "connector")
        for source, target, role in graph.edges
    }
    if model_edges != expected_edges:
        raise ValueError("MuJoCo control edges do not match graph definition")
    model_actuators = tuple((edge.from_node, edge.to_node) for edge in metadata.actuator_edges)
    if model_actuators != graph.actuator_edges:
        raise ValueError("MuJoCo routed actuator endpoints do not match graph definition")
    if session and not session.get("graph_definition_sha256"):
        raise ValueError("Recorded session has no graph definition hash")
    recorded_hash = session.get("graph_definition_sha256")
    if recorded_hash and recorded_hash != graph.sha256:
        raise ValueError(
            "Recorded graph definition hash does not match graph_definition_file: "
            f"{recorded_hash} != {graph.sha256}"
        )
    recorded_nodes = session.get("node_order")
    if recorded_nodes and tuple(recorded_nodes) != graph.control_node_names:
        raise ValueError("Recorded node order does not match graph_definition_file")
    expected_mapping = [
        {"triangle": triangle, "logical_node": logical, "control_node": control}
        for triangle, logical, control in graph.control_node_occurrences
    ]
    if session and session.get("control_node_mapping") != expected_mapping:
        raise ValueError("Recorded logical/control-node mapping does not match graph definition")
    expected_action_mask = [
        name not in graph.passive_control_node_names for name in graph.control_node_names
    ]
    if session and session.get("action_mask") != expected_action_mask:
        raise ValueError("Recorded passive action mask does not match graph definition")
    if session and tuple(session.get("passive_nodes", ())) != graph.passive_control_node_names:
        raise ValueError("Recorded passive-node occurrences do not match graph definition")
    recorded_edges = tuple(tuple(edge) for edge in session.get("edges", ()))
    if session and recorded_edges != graph.edges:
        raise ValueError("Recorded tube/connector edges do not match graph definition")
    recorded_actuators = tuple(tuple(edge) for edge in session.get("routed_actuator_edges", ()))
    if session and recorded_actuators != graph.actuator_edges:
        raise ValueError("Recorded routed actuator endpoints do not match graph definition")
    recorded_serial = session.get("serial_node_order")
    expected_actuated = {
        name for name in graph.control_node_names if name not in graph.passive_control_node_names
    }
    serial_is_valid = isinstance(recorded_serial, (list, tuple)) and all(
        isinstance(name, str) for name in recorded_serial
    )
    if serial_is_valid:
        serial_is_valid = (
            len(recorded_serial) == len(expected_actuated)
            and set(recorded_serial) == expected_actuated
        )
    if session and not serial_is_valid:
        raise ValueError(
            "Recorded serial order must contain every actuated graph node exactly once"
        )


def _assert_recorded_model_config(cfg, session: dict) -> None:
    if not session:
        return
    recorded_topology = session.get("truss_topology")
    if recorded_topology and recorded_topology != resolve_truss_topology(cfg):
        raise ValueError(
            f"Recorded topology {recorded_topology!r} does not match configured "
            f"{resolve_truss_topology(cfg)!r}"
        )
    if "truss_realistic" in session and bool(session["truss_realistic"]) != resolve_truss_realistic(cfg):
        raise ValueError("Recorded truss_realistic does not match replay configuration")
    if "scale" in session and not np.isclose(float(session["scale"]), _fixed_model_scale(cfg)):
        raise ValueError("Recorded scale does not match replay configuration")
    recorded_physics = session.get("physical_parameters")
    configured_physics = getattr(cfg, "physical_parameters", None)
    if OmegaConf.is_config(configured_physics):
        configured_physics = OmegaConf.to_container(configured_physics, resolve=True)
    if recorded_physics is not None and canonical_json(recorded_physics) != canonical_json(configured_physics):
        raise ValueError("Recorded physical_parameters do not match replay configuration")


def _build_environment(cfg, graph):
    topology = resolve_truss_topology(cfg)
    node_dict, _ = get_preset_definition(topology, scale=_fixed_model_scale(cfg))
    if set(node_dict) != set(graph.logical_node_names):
        raise ValueError(
            "Graph logical nodes must exactly match preset coordinates; "
            f"graph={sorted(graph.logical_node_names)}, preset={sorted(node_dict)}"
        )
    spec = get_mujoco_spec(
        node_dict,
        graph.triangle_dict,
        realistic=resolve_truss_realistic(cfg),
        physical_params=_physical_parameters_from_config(cfg),
    )
    metadata = control_graph_metadata_from_xml(spec.to_xml())
    render_mode = "human" if bool(getattr(cfg, "visualize", False)) else None
    env = MujocoPresetGraphEnv(cfg, render_mode=render_mode, model_source=spec)
    return env, metadata


def _output_path(cfg) -> Path:
    configured = getattr(cfg, "output_file", None)
    if configured not in (None, "", "null"):
        return Path(str(configured)).expanduser()
    try:
        from hydra.core.hydra_config import HydraConfig

        if HydraConfig.initialized():
            return Path(HydraConfig.get().runtime.output_dir) / "mujoco_replay.jsonl"
    except (ImportError, AttributeError, ValueError):
        pass
    return Path(str(getattr(cfg, "work_dir", "outputs"))) / "mujoco_replay.jsonl"


def _positions(env) -> np.ndarray:
    return np.asarray(
        get_node_features(
            env.mj_model,
            graph_view=env._graph_view(),
            aggregation="connector_ball",
        )[:, :3],
        dtype=np.float64,
    )


def _command_action(
    command: FirmwareCommand, graph, limit: int, *, serial_order=None
) -> np.ndarray:
    if serial_order is None:
        serial_order = graph.serial_node_order
    if serial_order is None:
        raise ValueError("Exact command replay requires roller-derived serial ordering")
    if len(command.ticks) != len(serial_order):
        raise ValueError(
            f"Command has {len(command.ticks)} channels but graph expects {len(serial_order)}"
        )
    by_name = dict(zip(serial_order, command.ticks))
    return np.asarray(
        [[float(by_name.get(name, 0)) / float(limit)] for name in graph.control_node_names],
        dtype=np.float32,
    )


def run_replay(cfg) -> Path:
    cfg.domain_randomization = False
    cfg.use_graph_observations = True
    cfg.use_control_graph = True
    cfg.mujoco_backend = "mujoco"
    graph = load_triangle_graph_definition(str(cfg.graph_definition_file))
    limit = int(getattr(cfg, "max_velocity_ticks_per_second", 1800))
    scheduled, session, limit = _load_commands(Path(str(cfg.input_file)), limit=limit)
    _assert_recorded_model_config(cfg, session)
    env, metadata = _build_environment(cfg, graph)
    _assert_graph_parity(graph, metadata, session)
    serial_order = tuple(session["serial_node_order"]) if session else graph.serial_node_order
    output_path = _output_path(cfg)
    dt = float(env.mj_model.model.opt.timestep) * int(env.nsubsteps)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("MuJoCo environment step duration must be positive")

    end_time = _schedule_end_time(scheduled)
    active_index = -1
    try:
        observation, _ = env.reset(seed=int(getattr(cfg, "seed", 0)))
        initial_rigidity = float(np.asarray(observation["rigidity"]).reshape(-1)[0])
        physical_parameters = getattr(cfg, "physical_parameters", None)
        if OmegaConf.is_config(physical_parameters):
            physical_parameters = OmegaConf.to_container(physical_parameters, resolve=True)
        with JsonlWriter(output_path) as writer:
            writer.write(
                "replay_session",
                started_at_utc=utc_now_iso(),
                graph_definition=graph.raw,
                graph_definition_sha256=graph.sha256,
                truss_topology=resolve_truss_topology(cfg),
                truss_realistic=resolve_truss_realistic(cfg),
                scale=_fixed_model_scale(cfg),
                physical_parameters=physical_parameters,
                node_order=list(graph.control_node_names),
                control_node_mapping=[
                    {"triangle": triangle, "logical_node": logical, "control_node": control}
                    for triangle, logical, control in graph.control_node_occurrences
                ],
                action_mask=[
                    name not in graph.passive_control_node_names
                    for name in graph.control_node_names
                ],
                passive_nodes=list(graph.passive_control_node_names),
                edges=list(graph.edges),
                serial_node_order=list(serial_order or ()),
                graph_serial_node_order=list(graph.serial_node_order or ()),
                routed_actuator_edges=list(graph.actuator_edges),
                max_velocity_ticks_per_second=limit,
                source_commands=[
                    {
                        "command": item.command.text,
                        "start_time_s": item.start_time_s,
                        "delivery_status": item.delivery_status,
                        "delivery_uncertain": item.delivery_uncertain,
                    }
                    for item in scheduled
                ],
                has_uncertain_command_delivery=any(
                    item.delivery_uncertain for item in scheduled
                ),
                source_recording_sha256=session.get("_source_recording_sha256"),
                simulation_dt_s=dt,
            )
            writer.write(
                "simulation_frame",
                step=0,
                simulation_time_s=0.0,
                node_positions=_positions(env).tolist(),
                command=None,
                routed_actuator_ctrl=env.mj_model.get_external_ctrl().astype(float).tolist(),
                rigidity=initial_rigidity,
                terminated=False,
                truncated=False,
            )
            simulation_time = 0.0
            step = 0
            while simulation_time < end_time - 1e-12:
                active_index = _latest_started_index(
                    scheduled, simulation_time, active_index
                )
                item = scheduled[active_index] if active_index >= 0 else None
                if item is not None and item.command.emergency_stop:
                    writer.write(
                        "replay_end",
                        reason="emergency_stop",
                        simulation_time_s=simulation_time,
                        remaining_commands=0,
                    )
                    break
                command_is_active = item is not None and (
                    simulation_time < item.start_time_s + item.duration_s - 1e-12
                )
                if command_is_active:
                    action = _command_action(
                        item.command, graph, limit, serial_order=serial_order
                    )
                    command_text = item.command.text
                else:
                    action = np.zeros((len(graph.control_node_names), 1), dtype=np.float32)
                    command_text = None
                observation, reward, terminated, truncated, info = env.step(action)
                step += 1
                substeps = int(info.get("substeps_executed", env.nsubsteps))
                simulation_time += float(env.mj_model.model.opt.timestep) * substeps
                if bool(getattr(cfg, "visualize", False)):
                    env.render()
                writer.write(
                    "simulation_frame",
                    step=step,
                    simulation_time_s=simulation_time,
                    node_positions=_positions(env).tolist(),
                    command=command_text,
                    command_delivery_status=(
                        item.delivery_status if command_is_active else None
                    ),
                    command_delivery_uncertain=(
                        item.delivery_uncertain if command_is_active else False
                    ),
                    normalized_action=action[:, 0].astype(float).tolist(),
                    routed_actuator_ctrl=env.mj_model.get_external_ctrl().astype(float).tolist(),
                    reward=float(reward),
                    rigidity=float(np.asarray(observation["rigidity"]).reshape(-1)[0]),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
                if terminated or truncated:
                    writer.write(
                        "replay_end",
                        reason="terminated" if terminated else "truncated",
                        simulation_time_s=simulation_time,
                        remaining_commands=len(scheduled) - active_index,
                    )
                    break
            else:
                reason = (
                    "emergency_stop"
                    if scheduled[-1].command.emergency_stop
                    and scheduled[-1].start_time_s <= simulation_time + 1e-12
                    else "complete"
                )
                writer.write("replay_end", reason=reason, simulation_time_s=simulation_time)
    finally:
        env.close()
    print(f"Saved MuJoCo replay to {output_path}")
    return output_path


@hydra.main(config_name="inference/real_robot_replay", config_path="../config", version_base=None)
def main(cfg):
    run_replay(cfg)


if __name__ == "__main__":
    main()
