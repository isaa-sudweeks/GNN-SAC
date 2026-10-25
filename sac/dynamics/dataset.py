"""Indexed nominal-source episodes with explicit graph alias semantics."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

SOURCE_TOPOLOGIES = (
    "tetrahedron", "henneberg_n8_2tube_187", "henneberg_n6_2tube_1",
    "usevitch_210272254_p1", "henneberg_n5_1tube_1", "henneberg_n6_1tube_2",
    "henneberg_n8_3tube_64", "usevitch_60243677150_p1",
)


def file_sha256(path: Path) -> str:
    """Hash an artifact without loading it all into memory."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_episode(episode: dict) -> dict:
    """Check finite aligned labels and independent physical-node representatives."""
    state = np.asarray(episode["state"])
    nxt = np.asarray(episode["next_state"])
    action = np.asarray(episode["action"])
    if state.ndim != 3 or state.shape[-1] != 6 or state.shape != nxt.shape:
        raise ValueError("state and next_state must share shape (T,N,6)")
    t, n, _ = state.shape
    if t == 0 or action.shape != (t, n, 1):
        raise ValueError("Nonempty episodes need actions shaped (T,N,1)")
    expected_shapes = {"progress": (t, 2), "global_xy": (t, 2), "next_global_xy": (t, 2),
                       "dt": (t,), "terminated": (t,), "truncated": (t,)}
    for key, shape in expected_shapes.items():
        if np.asarray(episode[key]).shape != shape:
            raise ValueError(f"Invalid {key} shape; expected {shape}")
    for key in ("state", "next_state", "action", "progress", "dt", "global_xy", "next_global_xy"):
        if not np.isfinite(episode[key]).all():
            raise ValueError(f"Nonfinite {key}")
    if np.any(np.asarray(episode["dt"]) <= 0):
        raise ValueError("Elapsed time must be positive")
    if np.any(np.abs(action) > 1.000001):
        raise ValueError("Actions exceed normalized bounds")
    mapping = np.asarray(episode["node_to_physical"], dtype=int)
    mask = np.asarray(episode["physical_mask"], dtype=bool)
    if mapping.shape != (n,) or mask.shape != (n,) or np.any(mapping < 0):
        raise ValueError("Invalid graph-to-physical mapping")
    unique = np.unique(mapping)
    if not np.array_equal(unique, np.arange(len(unique))) or len(np.unique(mapping[mask])) != len(unique):
        raise ValueError("Each physical node needs exactly one representative")
    if int(mask.sum()) != len(unique):
        raise ValueError("Duplicate physical-node representatives")
    representatives = np.asarray([np.flatnonzero(mask & (mapping == i))[0] for i in unique])
    if not np.allclose(state, state[:, representatives[mapping]], atol=1e-7):
        raise ValueError("Graph aliases disagree")
    if not np.allclose(nxt, nxt[:, representatives[mapping]], atol=1e-7):
        raise ValueError("Next graph aliases disagree")
    if not np.allclose(state[:, mask, :2].mean(1), 0, atol=1e-7):
        raise ValueError("States are not centered on physical-node mean")
    if not np.allclose(episode["progress"], episode["next_global_xy"] - episode["global_xy"], atol=1e-8):
        raise ValueError("Progress labels disagree")
    if t > 1:
        if not np.array_equal(nxt[:-1], state[1:]):
            raise ValueError("Episode state adjacency is broken")
        if not np.array_equal(np.asarray(episode["next_global_xy"])[:-1], np.asarray(episode["global_xy"])[1:]):
            raise ValueError("Episode global translation adjacency is broken")
        if np.any(episode["terminated"][:-1]) or np.any(episode["truncated"][:-1]):
            raise ValueError("Transition occurs after episode end")
    return {"transitions": t, "graph_nodes": n, "physical_nodes": len(unique), "finite": True,
            "adjacency_exact": True, "aliases_consistent": True}


def load_dataset(root: str | Path, roles: Iterable[str] | None = None) -> list[dict]:
    """Load verified episode NPZs; keep topology and episode boundaries explicit.

    Each dict merges per-topology static metadata with per-transition arrays. Hidden
    activation diagnostics are retained separately, never selected as model inputs.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    allowed = set(roles) if roles is not None else None
    records = manifest["episodes"]
    identities = [(r["topology"], r["role"], r["episode_id"]) for r in records]
    seeds = [r["seed"] for r in records]
    if len(set(identities)) != len(identities) or len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate episode identity or reset seed")
    if set(seeds) & {8100, 8101, 8102, 8103}:
        raise ValueError("Reserved evaluation-probe seed included")
    if any(r["role"] not in {"fitting", "validation", "evaluation"} for r in records):
        raise ValueError("Unknown dataset role")
    episodes = []
    static_keys = ("node_to_physical", "physical_mask", "edge_index", "edge_static", "action_mask",
                   "initial_bbox", "routing_incidence", "ctrlrange", "structural_edges")
    for record in manifest["episodes"]:
        if allowed is not None and record["role"] not in allowed:
            continue
        if record["topology"] not in SOURCE_TOPOLOGIES:
            raise ValueError("Dataset includes non-source topology")
        path = root / record["file"]
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Hash mismatch: {path}")
        metadata_path = root / record["metadata"]
        if file_sha256(metadata_path) != record["metadata_sha256"]:
            raise ValueError(f"Metadata hash mismatch: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        item = dict(metadata)
        for key in static_keys:
            item[key] = np.asarray(item[key])
        with np.load(path, allow_pickle=False) as arrays:
            item.update({key: arrays[key] for key in arrays.files})
        item.update(role=record["role"], topology=record["topology"], file=str(path),
                    episode_id=record["episode_id"], seed=record["seed"], amplitude=record.get("amplitude"))
        validate_episode(item)
        episodes.append(item)
    return episodes
