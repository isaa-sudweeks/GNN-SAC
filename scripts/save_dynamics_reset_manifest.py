"""Recover source pilot initial states and verify against hashes recorded at run time."""

import argparse
from dataclasses import fields
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sac")]
from scripts.collect_dynamics_source_probe import simulator_state
from scripts.run_simulator_mpc_pilot import make_source_env


def save(result_path: Path):
    result = json.loads(result_path.read_text())
    env = make_source_env(OmegaConf.create(result["settings"]))
    rows, arrays = [], {}
    try:
        for seed in result["settings"]["reset_seeds"]:
            env.reset()
            base = env.unwrapped
            base.reset(seed=seed)
            data = base.mj_model.data
            actual = hashlib.sha256(np.concatenate([data.qpos, data.qvel, data.act]).tobytes()).hexdigest()
            recorded = {r["initial_state_sha256"] for r in result["results"] if r["seed"] == seed}
            if recorded != {actual}:
                raise RuntimeError(f"Reconstructed state differs from recorded initial state for seed {seed}")
            arrays[f"seed_{seed}_integration_state"] = simulator_state(base)
            arrays[f"seed_{seed}_qpos"] = data.qpos.copy()
            arrays[f"seed_{seed}_qvel"] = data.qvel.copy()
            arrays[f"seed_{seed}_activation"] = data.act.copy()
            arrays[f"seed_{seed}_node_positions"] = base.mj_model.get_node_position_matrix().copy()
            rows.append(dict(seed=seed, matches_all_three_controller_hashes=True, sha256=actual))
        output = result_path.with_name("source-reset-states.npz")
        np.savez_compressed(output, **arrays)
        resolved_config = {field.name: getattr(base.config, field.name)
                           for field in fields(base.config) if field.name != "model_source"}
        xml = str(base.mj_model.xml)
        (result_path.parent / "source-model.xml").write_text(xml)
        report = dict(
            resets=rows, npz_file=output.name,
            initial_state_variation_qpos_peak_to_peak=np.ptp(
                np.stack([arrays[f"seed_{s}_qpos"] for s in result["settings"]["reset_seeds"]]), axis=0).tolist(),
            resolved_environment_config=resolved_config,
            model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
            state_origin="Reconstructed with identical nominal source settings and checked against every recorded initial-state hash",
            dependencies={name: importlib.metadata.version(name) for name in ("mujoco", "mujoco-truss-gen", "numpy", "torch")},
            code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                         for pattern in ("sac/dynamics/*.py", "scripts/*dynamics*.py", "scripts/run_simulator_mpc_pilot.py")
                         for path in ROOT.glob(pattern)})
        result_path.with_name("source-reset-manifest.json").write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(rows, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=ROOT / "outputs/dynamics-pilot/simulator-mpc-source.json")
    save(parser.parse_args().results)
