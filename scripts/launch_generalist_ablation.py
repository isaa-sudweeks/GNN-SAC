#!/usr/bin/env python3
"""Prepare or submit seven matched graph ablations from a shared-run manifest."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.launch_paper_shared import PROJECT_ROOT, submit_arms


ARMS = {
    "no-routing": ("graph_features.edge_direction", False),
    "no-tube-hubs": ("graph_features.tube_nodes", False),
    "no-edge-roles": ("graph_features.mask_edge_roles", True),
    "no-node-roles": ("graph_features.mask_node_roles", True),
    "no-rigidity-input": ("graph_features.mask_rigidity", True),
    "uniform-attention": ("message_attention_uniform", True),
    "no-edge-distance": ("graph_features.edge_distance", False),
}
RUN_IDENTITY = {"exp_name", "run_root", "work_dir", "wandb_name"}
COMMAND_PREFIX = ["uv", "run", "python", "sac/train.py", "-m"]


def _override_key(override: str) -> str:
    return override.partition("=")[0].lstrip("+")


def _resolved(overrides: list[str]) -> dict:
    cfg = compose(config_name="config", overrides=overrides)
    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)


def _differences(before: dict, after: dict, prefix: str = "") -> set[str]:
    differences = set()
    for key in before.keys() | after.keys():
        path = f"{prefix}{key}"
        left, right = before.get(key), after.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            differences.update(_differences(left, right, prefix=f"{path}."))
        elif key not in before or key not in after or left != right:
            differences.add(path)
    return differences


def build_manifest(baseline: dict, baseline_id: str, run_root: Path) -> dict:
    """Resolve each seed against the recorded scratch baseline and verify its delta."""
    scratch = [entry for entry in baseline["arms"] if entry["arm"] == "gnn_scratch"]
    if len(scratch) != 1 or scratch[0]["command"][:5] != COMMAND_PREFIX:
        raise ValueError("Expected one recorded Hydra GNN scratch sweep.")
    baseline_overrides = scratch[0]["command"][5:]
    seeds = tuple(int(job["seed"]) for job in scratch[0]["jobs"])
    seed_sweep = ",".join(map(str, seeds))
    if len(seeds) != 3 or len(set(seeds)) != 3 or f"seed={seed_sweep}" not in baseline_overrides:
        raise ValueError("Expected three distinct seeds matching the recorded sweep.")
    root = str(run_root)
    if not run_root.is_absolute() or any(char in root for char in "',\\\n"):
        raise ValueError("Use an absolute run root without quotes, commas, or backslashes.")
    base_common = [arg for arg in baseline_overrides if _override_key(arg) != "seed"]
    ablation_common = [arg for arg in base_common if _override_key(arg) not in RUN_IDENTITY]
    ablation_common.append(f"run_root='{root}'")
    # Stable naming also lets the shared offline W&B monitor pick up these runs.
    campaign = "paper-shared-gnn-scratch-ablation-v1"
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "config"), version_base=None):
        references = {seed: _resolved([*base_common, f"seed={seed}"]) for seed in seeds}
        ref = references[seeds[0]]
        train, heldout = ref["truss_topologies"], ref["eval_extra_topologies"]
        budget = int(ref["steps"]) * len(train)
        features = ref["graph_features"]
        if (
            ref["sac_backend"] != "gnn" or ref["distillation"]["enabled"]
            or ref["cross_validation"]["enabled"] or ref.get("init_from_agent")
            or ref["resume_from_checkpoint"] != "latest"
            or Path(ref["checkpoint_dir"]).is_absolute()
            or len(train) != 10 or len(set(train)) != 10
            or len(heldout) != 3 or len(set(heldout)) != 3 or set(train) & set(heldout)
            or budget != scratch[0]["total_online_transitions"]
            or list(train) != scratch[0]["training_topologies"]
            or list(heldout) != scratch[0]["heldout_topologies"]
        ):
            raise ValueError("The scratch baseline does not match the recorded split, budget, or initialization.")
        if (
            not ref["use_virtual_node"] or not ref["message_attention"] or ref["message_attention_uniform"]
            or features["tube_physics"]
            or not all(features[key] for key in ("edge_direction", "tube_nodes", "edge_roles", "node_roles", "edge_distance"))
            or any(features[key] for key in ("mask_node_roles", "mask_edge_roles", "mask_rigidity"))
        ):
            raise ValueError("The baseline must use all seven studied features and have tube physics off.")

        entries = []
        for arm, (key, value) in ARMS.items():
            common = [arg for arg in ablation_common if _override_key(arg) != key]
            change = f"{key}={str(value).lower()}"
            exp_name = f"{campaign}-{arm}"
            common.extend((f"exp_name={exp_name}", change))
            jobs = []
            for seed, reference in references.items():
                resolved = _resolved([*common, f"seed={seed}"])
                actual = _differences(reference, resolved) - RUN_IDENTITY
                if actual != {key} or not Path(resolved["work_dir"]).is_relative_to(run_root):
                    raise ValueError(f"{arm}, seed {seed}: expected only {key}, found {sorted(actual)}.")
                jobs.append({"seed": seed, "resolved_config": resolved})
            command = [*COMMAND_PREFIX, *common, f"seed={seed_sweep}"]
            entries.append({
                "arm": arm, "override": change, "jobs": jobs,
                "command": command, "command_display": shlex.join(command),
            })
    return {
        "baseline_id": baseline_id, "baseline_source_commit": baseline["source_commit"],
        "baseline_command": scratch[0]["command"],
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "dependency_lock_sha256": hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest(),
        "total_online_transitions_per_job": budget, "seeds": list(seeds),
        "run_root": root, "job_count": len(entries) * len(seeds), "arms": entries,
    }


def relaunch(manifest_path: Path) -> int:
    """Replay the original sweeps for arms with no active Slurm jobs."""
    manifest = json.loads(manifest_path.read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    if head != manifest["source_commit"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
    ).strip():
        raise ValueError("Relaunch from the clean source commit recorded in the launch manifest.")
    active_names = subprocess.check_output(
        ["squeue", "--noheader", "--user", getpass.getuser(), "--format", "%j"], text=True,
    ).splitlines()
    entries = []
    for entry in manifest["arms"]:
        exp_name = entry["jobs"][0]["resolved_config"]["exp_name"]
        if any(name.strip().startswith(f"{exp_name}-") for name in active_names):
            print(f"Skipping {entry['arm']}: Slurm still has active jobs for this arm.")
        else:
            entries.append(entry)
    if not entries:
        return 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    environment = dict(os.environ)
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return submit_arms(entries, environment, manifest_path.parent / "relaunch_logs" / timestamp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--baseline-id", help="Reference scratch Slurm array ID.")
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true", help="Submit all arms together after preflight.")
    mode.add_argument("--relaunch-from", type=Path, help="Replay the exact commands from a submitted launch manifest.")
    args = parser.parse_args()
    new_run_args = (args.baseline_manifest, args.baseline_id, args.run_root, args.manifest)
    if args.relaunch_from:
        if any(new_run_args):
            parser.error("--relaunch-from uses the original manifest without new run options.")
        try:
            return relaunch(args.relaunch_from)
        except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as error:
            parser.error(str(error))
    if not all(new_run_args):
        parser.error("A fresh campaign requires --baseline-manifest, --baseline-id, --run-root and --manifest.")
    baseline = json.loads(args.baseline_manifest.read_text())
    try:
        manifest = build_manifest(baseline, args.baseline_id, args.run_root)
        if args.submit:
            if subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT).strip():
                raise ValueError("Commit the ablation source before submitting.")
            if subprocess.run(
                ["git", "merge-base", "--is-ancestor", baseline["source_commit"], "HEAD"],
                cwd=PROJECT_ROOT, check=False,
            ).returncode:
                raise ValueError("The checkout must descend from the recorded baseline source.")
            if manifest["source_commit"] == baseline["source_commit"]:
                raise ValueError("The ablation source must have its own commit.")
            if manifest["dependency_lock_sha256"] != baseline["dependency_lock_sha256"]:
                raise ValueError("The dependency lock must match the baseline.")
            if args.run_root.exists():
                raise ValueError("Use a fresh run root for the first submission.")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest.open("x") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
    except (KeyError, ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"{manifest['job_count']} jobs at {manifest['total_online_transitions_per_job']:,} transitions each.")
    for entry in manifest["arms"]:
        print(entry["command_display"])
    if args.submit:
        environment = dict(os.environ)
        environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        return submit_arms(manifest["arms"], environment, args.manifest.parent / "launch_logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
