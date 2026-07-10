"""Run the three-seed domain-randomization family training smoke matrix."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
OUTPUT = ROOT / "outputs" / "domain_randomization_validation" / "training_smoke"

FAMILIES = {
    "nominal": [],
    "body": ["body_mass_multiplier", "body_inertia_multiplier"],
    "dof": ["dof_damping_multiplier", "dof_armature", "dof_frictionloss"],
    "actuator": ["actuator_gain_multiplier", "actuator_bias_multiplier", "actuator_dynprm_multiplier"],
    "geom": ["geom_friction_slide", "geom_friction_torsional", "geom_friction_rolling"],
    "tendon": ["tendon_stiffness", "tendon_damping", "tendon_armature", "tendon_frictionloss"],
    "gravity": ["gravity_z"],
}
FAMILIES["combined"] = [name for family in FAMILIES.values() for name in family]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    common = [
        "device=cpu", "steps=40", "max_steps=10", "seed_steps=4", "batch_size=4", "replay_ratio=1",
        "pretrain_steps=0", "num_envs=1", "enable_wandb=false", "save_video=false",
        "save_agent=false", "save_csv=true", "checkpoint_freq=0", "eval_freq=1000",
        "eval_episodes=1", "progress_freq=40", "eval_at_end=true",
    ]
    for family, fields in FAMILIES.items():
        for seed in (1, 2, 3):
            run_dir = OUTPUT / f"{family}_seed{seed}"
            overrides = list(common) + [f"seed={seed}", f"hydra.run.dir={run_dir}"]
            if family == "nominal":
                overrides.append("domain_randomization=false")
            else:
                overrides.extend(["domain_randomization=true", "domain_randomization_params.length_scale.enabled=false"])
                overrides.extend(f"domain_randomization_params.{field}.enabled=true" for field in fields)
            started = time.perf_counter()
            completed = subprocess.run(
                [str(PYTHON), "sac/gnn_train.py", *overrides],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "validation_driver.log"
            log_path.write_text(completed.stdout)
            rows.append({
                "family": family,
                "seed": seed,
                "returncode": completed.returncode,
                "elapsed_seconds": time.perf_counter() - started,
                "run_dir": str(run_dir),
                "log": str(log_path),
            })
            (OUTPUT / "training_results.json").write_text(json.dumps(rows, indent=2))
            print(f"{family} seed={seed}: rc={completed.returncode}", flush=True)
    path = OUTPUT / "training_results.json"
    path.write_text(json.dumps(rows, indent=2))
    print(path)


if __name__ == "__main__":
    main()
