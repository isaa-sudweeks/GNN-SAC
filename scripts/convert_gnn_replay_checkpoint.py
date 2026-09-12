"""Non-destructively convert an object-based GNN replay checkpoint to tensor replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.tensor_gnn_buffer import TensorGNNBuffer


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def convert(source: Path, destination: Path, *, chunk_size: int = 4096) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source == destination:
        raise ValueError("Source and destination must differ.")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    started = time.perf_counter()
    state = torch.load(source, map_location="cpu", weights_only=False)
    load_seconds = time.perf_counter() - started
    replay = state.get("buffer", {})
    if int(replay.get("format_version", 0)) != 2:
        raise ValueError("Source is not an object-based GNN replay checkpoint.")
    config = dict(state.get("config", {}))
    config.update(device="cpu", replay_backend="torchrl_tensor", replay_storage="cpu_pinned")
    cfg = SimpleNamespace(**config)
    tensor = TensorGNNBuffer(cfg)
    conversion_started = time.perf_counter()
    tensor.load_legacy_state_dict(replay, chunk_size=chunk_size)
    conversion_seconds = time.perf_counter() - conversion_started
    original_sizes = {task: int(value["size"]) for task, value in replay["buffers"].items()}
    if tensor.sizes_by_task != original_sizes:
        raise RuntimeError("Converted replay sizes differ from the source checkpoint.")
    state["buffer"] = tensor.state_dict()
    if isinstance(state.get("config"), dict):
        state["config"]["replay_backend"] = "torchrl_tensor"
        state["config"]["replay_storage"] = "auto"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    save_started = time.perf_counter()
    torch.save(state, temporary)
    save_seconds = time.perf_counter() - save_started
    verify_started = time.perf_counter()
    verified = torch.load(temporary, map_location="cpu", weights_only=False)
    if verified["buffer"]["format_version"] != 3:
        raise RuntimeError("Converted checkpoint failed format verification.")
    temporary.replace(destination)
    verify_seconds = time.perf_counter() - verify_started
    report = {
        "format_version": 1,
        "source": str(source), "destination": str(destination),
        "source_sha256": digest(source), "destination_sha256": digest(destination),
        "source_bytes": source.stat().st_size, "destination_bytes": destination.stat().st_size,
        "sizes_by_task": original_sizes,
        "load_seconds": load_seconds, "conversion_seconds": conversion_seconds,
        "save_seconds": save_seconds, "verify_seconds": verify_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    destination.with_suffix(destination.suffix + ".conversion.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--chunk-size", type=int, default=4096)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    print(json.dumps(convert(args.source, args.destination, chunk_size=args.chunk_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
