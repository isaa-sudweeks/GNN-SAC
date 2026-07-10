"""Execute the domain-randomization validation plan and write machine-readable results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
for path in (ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from env.mujoco_gen.topology_envs import (  # noqa: E402
    _RUNTIME_DOMAIN_RANDOMIZATION_FIELDS,
    _domain_randomization,
    _enabled_range,
    make_truss_env_config,
)
from mujoco_truss_gen import MujocoRelativeObsEnv  # noqa: E402
from tests.test_gnn_mujoco_truss_gen_smoke import graph_test_cfg  # noqa: E402


FAMILIES = {
    "body": ("body_mass_multiplier", "body_inertia_multiplier"),
    "dof": ("dof_damping_multiplier", "dof_armature", "dof_frictionloss"),
    "actuator": (
        "actuator_gain_multiplier",
        "actuator_bias_multiplier",
        "actuator_dynprm_multiplier",
    ),
    "geom": ("geom_friction_slide", "geom_friction_torsional", "geom_friction_rolling"),
    "tendon": ("tendon_stiffness", "tendon_damping", "tendon_armature", "tendon_frictionloss"),
    "gravity": ("gravity_z",),
}

MODEL_TARGETS = {
    "body_mass_multiplier": "body_mass",
    "body_inertia_multiplier": "body_inertia",
    "dof_damping_multiplier": "dof_damping",
    "dof_armature": "dof_armature",
    "dof_frictionloss": "dof_frictionloss",
    "actuator_gain_multiplier": "actuator_gainprm",
    "actuator_bias_multiplier": "actuator_biasprm",
    "actuator_dynprm_multiplier": "actuator_dynprm",
    "geom_friction_slide": "geom_friction",
    "geom_friction_torsional": "geom_friction",
    "geom_friction_rolling": "geom_friction",
    "tendon_stiffness": "tendon_stiffness",
    "tendon_damping": "tendon_damping",
    "tendon_armature": "tendon_armature",
    "tendon_frictionloss": "tendon_frictionloss",
    "gravity_z": "gravity",
}


def ranges_from_yaml() -> dict[str, tuple[float, float]]:
    cfg = OmegaConf.load(ROOT / "config" / "physics" / "domain_randomization.yaml")
    params = cfg.domain_randomization_params
    return {name: _enabled_range({name: {**params[name], "enabled": True}}, name) for name in _RUNTIME_DOMAIN_RANDOMIZATION_FIELDS}


def randomization_params(ranges: dict[str, tuple[float, float]], *, model: bool = False) -> dict:
    params = {"length_scale": {"enabled": model, "min": 0.95, "max": 1.05}}
    params.update(
        {name: {"enabled": True, "min": low, "max": high} for name, (low, high) in ranges.items()}
    )
    if model:
        params["physical_parameters"] = {
            "node_radius": {"enabled": True, "min": 0.095, "max": 0.105}
        }
    return params


def make_native(ranges: dict[str, tuple[float, float]], *, realistic: bool = False, model: bool = False):
    cfg = graph_test_cfg(
        truss_topology="octahedron",
        truss_realistic=realistic,
        max_steps=110,
        domain_randomization=bool(ranges or model),
        domain_randomization_params=randomization_params(ranges, model=model),
    )
    return MujocoRelativeObsEnv(make_truss_env_config(cfg))


def finite_env(env, obs, reward: float | None = None) -> bool:
    data = env.mj_model.data
    arrays = (obs, data.qpos, data.qvel, data.qacc, data.ctrl)
    return all(np.isfinite(np.asarray(value)).all() for value in arrays) and (
        reward is None or np.isfinite(reward)
    )


def contract_validation(results: dict) -> None:
    failures = []
    for bad, message in [
        ({"enabled": True, "min": 1.0}, "missing"),
        ({"enabled": True, "min": np.nan, "max": 1.0}, "non-finite"),
        ({"enabled": True, "min": 2.0, "max": 1.0}, "reversed"),
    ]:
        try:
            _enabled_range({"body_mass_multiplier": bad}, "body_mass_multiplier")
            failures.append(f"{message} range accepted")
        except ValueError:
            pass
    cfg = graph_test_cfg(
        domain_randomization=True,
        domain_randomization_params={
            "length_scale": {"enabled": False},
            "dof_armature": {"enabled": True, "min": 0.01, "max": 0.01},
        },
    )
    dr = _domain_randomization(cfg, "octahedron", False)
    if dr.dof_armature_range != (0.01, 0.01):
        failures.append("exact range propagation failed")
    results["contract"] = {"passed": not failures, "failures": failures}


def native_target_validation(bounds: dict, results: dict) -> None:
    failures = []
    for name, (low, high) in bounds.items():
        # Use a boundary rather than the midpoint: multiplier midpoints and sliding
        # friction intentionally coincide with the nominal value.
        value = high
        env = make_native({name: (value, value)})
        try:
            before = {key: array.copy() for key, array in env._runtime_nominals.items()}
            _, info = env.reset(seed=7)
            after = {
                "body_mass": env.mj_model.model.body_mass,
                "body_inertia": env.mj_model.model.body_inertia,
                "dof_damping": env.mj_model.model.dof_damping,
                "dof_armature": env.mj_model.model.dof_armature,
                "dof_frictionloss": env.mj_model.model.dof_frictionloss,
                "actuator_gainprm": env.mj_model.model.actuator_gainprm,
                "actuator_biasprm": env.mj_model.model.actuator_biasprm,
                "actuator_dynprm": env.mj_model.model.actuator_dynprm,
                "geom_friction": env.mj_model.model.geom_friction,
                "tendon_stiffness": env.mj_model.model.tendon_stiffness,
                "tendon_damping": env.mj_model.model.tendon_damping,
                "tendon_armature": env.mj_model.model.tendon_armature,
                "tendon_frictionloss": env.mj_model.model.tendon_frictionloss,
                "gravity": env.mj_model.model.opt.gravity,
            }
            if info["domain_randomization"].get(name) != value:
                failures.append(f"{name}: wrong reported sample")
            target = MODEL_TARGETS[name]
            changed = {key for key in before if not np.array_equal(before[key], after[key])}
            if target not in changed:
                failures.append(f"{name}: target {target} did not change")
            if changed - {target}:
                failures.append(f"{name}: also changed {sorted(changed - {target})}")
        finally:
            env.close()
    results["native_targets"] = {"passed": not failures, "failures": failures}


def sampling_validation(bounds: dict, results: dict, output: Path) -> None:
    samples = []
    env = make_native(bounds)
    try:
        for seed in range(64):
            _, info = env.reset(seed=seed)
            samples.append(info["domain_randomization"])
        same_a = env.reset(seed=123)[1]["domain_randomization"]
        same_b = env.reset(seed=123)[1]["domain_randomization"]
        different = env.reset(seed=124)[1]["domain_randomization"]
    finally:
        env.close()
    failures = []
    for name, (low, high) in bounds.items():
        values = np.asarray([row[name] for row in samples])
        if not ((values >= low).all() and (values <= high).all()):
            failures.append(f"{name}: out-of-bounds sample")
        if low != high and np.unique(values).size < 2:
            failures.append(f"{name}: samples did not vary")
    if same_a != same_b:
        failures.append("identical seed was not reproducible")
    if same_a == different:
        failures.append("different seed produced identical complete sample")

    csv_path = output / "sampled_ranges.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bounds))
        writer.writeheader()
        writer.writerows(samples)
    results["sampling"] = {
        "passed": not failures,
        "failures": failures,
        "sample_count": len(samples),
        "csv": str(csv_path),
        "summary": {
            name: {"min": min(row[name] for row in samples), "max": max(row[name] for row in samples)}
            for name in bounds
        },
    }


def run_case(name: str, ranges: dict, realistic: bool, model: bool = False, iterations: int = 100) -> dict:
    env = make_native(ranges, realistic=realistic, model=model)
    failures = []
    warning_messages = []
    start = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for seed in range(iterations):
                obs, _ = env.reset(seed=seed)
                if not finite_env(env, obs):
                    failures.append(f"reset {seed}: non-finite state")
                    break
                action = env.action_space.sample()
                obs, reward, _, _, _ = env.step(action)
                if not finite_env(env, obs, reward):
                    failures.append(f"step {seed}: non-finite state/reward")
                    break
            warning_messages = sorted({str(item.message) for item in caught})
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        env.close()
    elapsed = time.perf_counter() - start
    return {
        "case": name,
        "realistic": realistic,
        "passed": not failures and not warning_messages,
        "failures": failures,
        "warnings": warning_messages,
        "resets": iterations,
        "steps": iterations,
        "elapsed_seconds": elapsed,
        "steps_per_second": iterations / elapsed,
    }


def simulation_matrix(bounds: dict, results: dict) -> None:
    cases = [("nominal", {}, False)]
    cases.extend((family, {name: bounds[name] for name in names}, False) for family, names in FAMILIES.items())
    cases.extend([
        ("all_fixed", bounds, False),
        ("model_level", {}, True),
        ("all_native", bounds, True),
    ])
    rows = []
    for realistic in (False, True):
        for name, ranges, model in cases:
            print(f"simulation case: realistic={realistic} {name}", flush=True)
            rows.append(run_case(name, ranges, realistic, model))
    nominal = {row["realistic"]: row["steps_per_second"] for row in rows if row["case"] == "nominal"}
    for row in rows:
        row["throughput_vs_nominal"] = row["steps_per_second"] / nominal[row["realistic"]]
    results["simulation_matrix"] = rows


def sensitivity(bounds: dict, results: dict) -> None:
    rows = []
    for realistic in (False, True):
        for name, (low, high) in bounds.items():
            print(f"sensitivity: realistic={realistic} {name}", flush=True)
            nominal = (low + high) / 2.0
            for point, value in (("min", low), ("nominal", nominal), ("max", high)):
                row = run_case(f"{name}:{point}", {name: (value, value)}, realistic, iterations=1)
                rows.append(row)
        coupled = {
            "mass_inertia": {name: bounds[name] for name in FAMILIES["body"]},
            "tendon_stiffness_damping": {name: bounds[name] for name in ("tendon_stiffness", "tendon_damping")},
            "all_friction_axes": {name: bounds[name] for name in FAMILIES["geom"]},
        }
        for name, ranges in coupled.items():
            rows.append(run_case(f"coupled:{name}", ranges, realistic, iterations=10))
    results["sensitivity"] = rows


def plot_sampling(results: dict, output: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plot_sampling_svg(results, output)
        return

    data = np.genfromtxt(results["sampling"]["csv"], delimiter=",", names=True)
    names = list(data.dtype.names or ())
    fig, axes = plt.subplots(4, 4, figsize=(14, 10), constrained_layout=True)
    for axis, name in zip(axes.flat, names):
        axis.hist(data[name], bins=12, color="#4472C4", edgecolor="#23395B")
        axis.set_title(name.replace("_", " "), fontsize=8)
        axis.tick_params(labelsize=7)
    fig.suptitle("Domain-randomization samples (64 seeded native MuJoCo resets)", fontsize=14)
    path = output / "sampled_range_distributions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    results["sampling"]["chart"] = str(path)


def plot_sampling_svg(results: dict, output: Path) -> None:
    """Dependency-free small-multiple histograms for minimal environments."""
    data = np.genfromtxt(results["sampling"]["csv"], delimiter=",", names=True)
    names = list(data.dtype.names or ())
    width, height = 1400, 1000
    panel_w, panel_h = 340, 220
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="700" y="35" text-anchor="middle" font-family="sans-serif" font-size="22" fill="#222">Domain-randomization samples (64 seeded native MuJoCo resets)</text>',
    ]
    for index, name in enumerate(names):
        col, row = index % 4, index // 4
        x0, y0 = 25 + col * panel_w, 60 + row * panel_h
        values = np.asarray(data[name], dtype=float)
        counts, _ = np.histogram(values, bins=12)
        peak = max(int(counts.max()), 1)
        parts.append(f'<text x="{x0 + 150}" y="{y0 + 18}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#222">{name.replace("_", " ")}</text>')
        parts.append(f'<line x1="{x0}" y1="{y0 + 180}" x2="{x0 + 300}" y2="{y0 + 180}" stroke="#555"/>')
        for bin_index, count in enumerate(counts):
            bar_h = 145 * count / peak
            x = x0 + bin_index * 25
            y = y0 + 180 - bar_h
            parts.append(f'<rect x="{x + 1}" y="{y:.1f}" width="23" height="{bar_h:.1f}" fill="#4472C4" stroke="#23395B"/>')
        parts.append(f'<text x="{x0}" y="{y0 + 200}" font-family="monospace" font-size="10" fill="#555">{values.min():.4g}</text>')
        parts.append(f'<text x="{x0 + 300}" y="{y0 + 200}" text-anchor="end" font-family="monospace" font-size="10" fill="#555">{values.max():.4g}</text>')
    parts.append("</svg>")
    path = output / "sampled_range_distributions.svg"
    path.write_text("\n".join(parts))
    results["sampling"]["chart"] = str(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "domain_randomization_validation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    bounds = ranges_from_yaml()
    contract_validation(results)
    native_target_validation(bounds, results)
    sampling_validation(bounds, results, args.output)
    simulation_matrix(bounds, results)
    sensitivity(bounds, results)
    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    path = args.output / "validation_results.json"
    path.write_text(json.dumps(results, indent=2))
    plot_sampling(results, args.output)
    path.write_text(json.dumps(results, indent=2))
    print(path)


if __name__ == "__main__":
    main()
