"""Shared graph, command, and JSONL contracts for sim-to-real diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import numpy as np


SCHEMA_VERSION = 1
_COMMAND_RE = re.compile(r"^VEL_DUR:([+-]?\d+(?:,[+-]?\d+)*):([+-]?(?:\d+(?:\.\d*)?|\.\d+))$")


def canonical_json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TriangleGraphDefinition:
    """Resolved controller and MuJoCo routing from a triangle graph JSON object."""

    raw: dict[str, Any]
    canonical: str
    sha256: str
    triangle_nodes: dict[str, tuple[str, str, str]]
    passive_nodes: dict[str, str]
    triangle_dict: dict[str, list[str]]
    logical_node_names: tuple[str, ...]
    control_node_names: tuple[str, ...]
    control_node_by_occurrence: dict[tuple[str, str], str]
    control_node_occurrences: tuple[tuple[str, str, str], ...]
    passive_control_node_names: tuple[str, ...]
    edges: tuple[tuple[str, str, str], ...]
    actuator_edges: tuple[tuple[str, str], ...]
    serial_node_order: tuple[str, ...] | None


def resolve_triangle_graph_definition(value: Mapping[str, object]) -> TriangleGraphDefinition:
    """Validate the shared triangle/passive/roller contract and resolve routing."""
    raw_triangles = value.get("triangles")
    if not isinstance(raw_triangles, list) or not raw_triangles:
        raise ValueError("triangles must be a nonempty list")

    triangle_nodes: dict[str, tuple[str, str, str]] = {}
    passive_nodes: dict[str, str] = {}
    roller_by_occurrence: dict[tuple[str, str], tuple[int, str]] = {}
    roller_numbers: set[int] = set()
    any_rollers = False
    logical_order: list[str] = []

    for index, raw_triangle in enumerate(raw_triangles):
        if not isinstance(raw_triangle, dict):
            raise ValueError(f"triangles[{index}] must be an object")
        missing = {"name", "nodes", "passive_node"} - set(raw_triangle)
        if missing:
            raise ValueError(f"triangles[{index}] is missing fields: {sorted(missing)}")
        name = str(raw_triangle["name"])
        if not name or name in triangle_nodes:
            raise ValueError(f"Triangle names must be unique and nonempty; got {name!r}")
        raw_nodes = raw_triangle["nodes"]
        if not isinstance(raw_nodes, list) or len(raw_nodes) != 3:
            raise ValueError(f"Triangle {name!r} must contain exactly three nodes")
        nodes = tuple(str(node) for node in raw_nodes)
        if any(not node for node in nodes) or len(set(nodes)) != 3:
            raise ValueError(f"Triangle {name!r} nodes must be unique nonempty names")
        passive = str(raw_triangle["passive_node"])
        if passive not in nodes:
            raise ValueError(f"Triangle {name!r} passive_node must be one of its nodes")
        triangle_nodes[name] = nodes
        passive_nodes[name] = passive
        for node in nodes:
            if node not in logical_order:
                logical_order.append(node)

        rollers = raw_triangle.get("rollers")
        if rollers is None:
            continue
        any_rollers = True
        if not isinstance(rollers, dict):
            raise ValueError(f"Triangle {name!r} rollers must be a node-to-roller object")
        unknown = {str(node) for node in rollers} - set(nodes)
        if unknown:
            raise ValueError(f"Triangle {name!r} rollers contains unknown nodes: {sorted(unknown)}")
        if passive in {str(node) for node in rollers}:
            raise ValueError(f"Triangle {name!r} passive node {passive!r} must not have a roller")
        for raw_node, raw_roller in rollers.items():
            node = str(raw_node)
            roller_id = str(raw_roller)
            if not roller_id.isdecimal():
                raise ValueError(f"Triangle {name!r} roller for {node!r} must be decimal")
            number = int(roller_id)
            if number in roller_numbers:
                raise ValueError(f"Roller numbers must be unique; duplicate: {roller_id!r}")
            roller_numbers.add(number)
            roller_by_occurrence[(name, node)] = (number, roller_id)

    if any_rollers:
        expected = {
            (triangle, node)
            for triangle, nodes in triangle_nodes.items()
            for node in nodes
            if node != passive_nodes[triangle]
        }
        missing = expected - set(roller_by_occurrence)
        if missing:
            details = [f"{triangle}.{node}" for triangle, node in sorted(missing)]
            raise ValueError(
                "When rollers are configured, every actuated triangle node needs a "
                f"roller number; missing: {details}"
            )

    control_names: list[str] = []
    by_occurrence: dict[tuple[str, str], str] = {}
    by_triangle: dict[str, list[str]] = {}
    by_logical: dict[str, list[str]] = {}
    owned: set[str] = set()
    for triangle, nodes in triangle_nodes.items():
        controls = []
        for node in nodes:
            control = node if node not in owned else f"{node}_tri_{triangle}"
            owned.add(node)
            controls.append(control)
            control_names.append(control)
            by_occurrence[(triangle, node)] = control
            by_logical.setdefault(node, []).append(control)
        by_triangle[triangle] = controls

    edges: list[tuple[str, str, str]] = []
    passive_controls: list[str] = []
    actuator_edges: list[tuple[str, str]] = []
    for triangle, nodes in triangle_nodes.items():
        controls = by_triangle[triangle]
        passive_index = nodes.index(passive_nodes[triangle])
        passive_control = controls[passive_index]
        passive_controls.append(passive_control)
        for index, source in enumerate(controls):
            target = controls[(index + 1) % 3]
            edges.append((source, target, "tube"))
            if source == passive_control or target == passive_control:
                actuator_edges.append((source, target))
    for controls in by_logical.values():
        for index, source in enumerate(controls):
            for target in controls[index + 1 :]:
                edges.append((source, target, "connector"))

    serial_order = None
    if roller_by_occurrence:
        serial_order = tuple(
            by_occurrence[occurrence]
            for occurrence, _ in sorted(
                roller_by_occurrence.items(), key=lambda item: item[1][0]
            )
        )
    raw = json.loads(json.dumps(value))
    return TriangleGraphDefinition(
        raw=raw,
        canonical=canonical_json(raw),
        sha256=sha256_json(raw),
        triangle_nodes=triangle_nodes,
        passive_nodes=passive_nodes,
        triangle_dict={
            name: [*nodes, passive_nodes[name]] for name, nodes in triangle_nodes.items()
        },
        logical_node_names=tuple(logical_order),
        control_node_names=tuple(control_names),
        control_node_by_occurrence=by_occurrence,
        control_node_occurrences=tuple(
            (triangle, node, by_occurrence[(triangle, node)])
            for triangle, nodes in triangle_nodes.items()
            for node in nodes
        ),
        passive_control_node_names=tuple(passive_controls),
        edges=tuple(edges),
        actuator_edges=tuple(actuator_edges),
        serial_node_order=serial_order,
    )


def load_triangle_graph_definition(path: str | Path) -> TriangleGraphDefinition:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Graph definition must contain a JSON object")
    return resolve_triangle_graph_definition(value)


@dataclass(frozen=True)
class FirmwareCommand:
    ticks: tuple[int, ...]
    duration_seconds: float
    text: str

    @property
    def emergency_stop(self) -> bool:
        return self.duration_seconds == 0.0 and all(value == 0 for value in self.ticks)


def parse_firmware_command(text: str, *, limit: int = 1800) -> FirmwareCommand:
    text = text.strip()
    match = _COMMAND_RE.fullmatch(text)
    if match is None:
        raise ValueError(f"Invalid firmware command: {text!r}")
    ticks = tuple(int(value) for value in match.group(1).split(","))
    duration = float(match.group(2))
    if any(abs(value) > int(limit) for value in ticks):
        raise ValueError(f"Firmware command exceeds ±{limit} ticks: {text!r}")
    if not np.isfinite(duration) or duration < 0.0:
        raise ValueError("Firmware command duration must be finite and nonnegative")
    if duration == 0.0 and any(value != 0 for value in ticks):
        raise ValueError("Only an all-zero emergency command may have zero duration")
    return FirmwareCommand(ticks=ticks, duration_seconds=duration, text=text)


class JsonlWriter:
    """Crash-tolerant JSON Lines writer that flushes every complete record."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("w", encoding="utf-8")

    def write(self, record_type: str, **payload: object) -> None:
        record = {"schema_version": SCHEMA_VERSION, "type": record_type, **payload}
        self._stream.write(canonical_json(record) + "\n")
        self._stream.flush()

    @property
    def closed(self) -> bool:
        return self._stream.closed

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def read_jsonl(path: str | Path, *, tolerate_partial_final_line: bool = True) -> list[dict[str, Any]]:
    lines = Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    records = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_partial_final_line and index == len(lines):
                break
            raise ValueError(f"Invalid JSONL record at line {index} of {path}") from None
        if not isinstance(record, dict):
            raise ValueError(f"JSONL record {index} must be an object")
        records.append(record)
    return records


def numpy_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): numpy_value(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [numpy_value(item) for item in value]
    return value
