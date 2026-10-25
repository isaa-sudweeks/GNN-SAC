"""Source-only paired simulator-MPC diagnostic; never fits a dynamics model."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sac"))
from env.mujoco import make_env
from dynamics.planner import CEMConfig, cem_plan, interpolate_knots, planning_score
from dynamics.simulator import SimulatorBranch


def make_source_env(settings):
    """Construct nominal source envs independently with the same explicit settings."""
    if settings.topology != "tetrahedron":
        raise ValueError("First diagnostic is restricted to source tetrahedron")
    return make_env(OmegaConf.create({
        "task": f"truss-graph:{settings.topology}", "obs": "state",
        "save_video": False, "visualize": False, "max_steps": 1000,
        "nsubsteps": 100, "speed": 0.05, "use_control_graph": True,
        "truss_realistic": False, "domain_randomization": False,
        "normalize_observations": True, "seed": 4101,
        "enable_wandb": False, "save_agent": False, "save_csv": False,
    }))


def run(settings, output):
    """Run all controllers from exact copied initial states and retain individuals."""
    cfg = CEMConfig(**OmegaConf.to_container(settings.cem))
    live, planning = make_source_env(settings), make_source_env(settings)
    rows = []
    total_start = time.perf_counter()
    try:
        live.reset()
        planning.reset()
        branch = SimulatorBranch(live, planning)
        # A separate restore adapter restores the execution env only between
        # paired controller trials. Candidate evaluation exclusively uses branch.
        reset_adapter = SimulatorBranch(planning, live)
        base = live.unwrapped
        shape = live.action_space.shape
        action_size = int(np.prod(shape))
        dt = float(base.nsubsteps * base.mj_model.model.opt.timestep)
        diagonal = float(base.mj_model.initial_bounding_box_diagonal)
        n_actuators = base.num_external_actuators
        for seed in settings.reset_seeds:
            # Gym wrapper's reset doesn't forward seed, so reset native explicitly.
            live.reset()
            initial_obs, _ = base.reset(seed=int(seed))
            initial = branch.capture()
            mask = np.repeat(initial_obs["action_mask"], shape[-1]).astype(float)
            snapshot_hash = hashlib.sha256(
                np.concatenate([initial.data.qpos, initial.data.qvel, initial.data.act]).tobytes()
            ).hexdigest()
            for controller in ("zero", "smooth_random", "simulator_mpc"):
                reset_adapter.restore(initial)
                start_com = base._center_of_mass().copy()
                trace = [start_com.tolist()]
                raw_return = 0.0
                latencies = []
                control_actions = []
                sim_steps = 0
                previous_plan = None
                collapsed = False
                elapsed = 0.0
                planner_rng = np.random.default_rng(int(settings.planner_seed) + int(seed))
                random_rng = np.random.default_rng(int(settings.planner_seed) + int(seed) + 100000)
                random_sequence = interpolate_knots(
                    random_rng.normal(0, cfg.initial_std, (max(2, int(settings.episode_steps) // 5), action_size)),
                    int(settings.episode_steps))
                random_sequence = np.clip(random_sequence, -1, 1) * mask
                for step in range(int(settings.episode_steps)):
                    if controller == "simulator_mpc":
                        snapshot = branch.capture()
                        def score(result, horizon):
                            nonlocal sim_steps
                            sim_steps += result.simulator_steps
                            times = np.diff(result.elapsed_times)
                            effort = sum(float(info["energy_penalty_raw"]) * t / n_actuators
                                         for info, t in zip(result.infos, times))
                            return planning_score(
                                float(result.com_positions[-1, 0] - result.com_positions[0, 0]),
                                diagonal, effort,
                                effort_weight=float(settings.objective.effort_weight),
                                collapsed=result.terminated or result.nonfinite,
                                remaining_seconds=max(0, horizon * dt - result.elapsed_times[-1]),
                                failure_cost=float(settings.objective.failure_cost),
                                failure_cost_per_second=float(settings.objective.failure_cost_per_second))
                        def evaluate(sequences):
                            return branch.evaluate(snapshot, sequences.reshape(-1, cfg.horizon, *shape), score)
                        before = time.perf_counter()
                        result = cem_plan(evaluate, action_size, cfg, planner_rng,
                                          action_mask=mask, previous_plan=previous_plan)
                        latencies.append(time.perf_counter() - before)
                        previous_plan = result.actions
                        action = result.actions[0].reshape(shape)
                    elif controller == "zero":
                        action = np.zeros(shape)
                    else:
                        action = random_sequence[step].reshape(shape)
                    control_actions.append(action.tolist())
                    _, reward, done, info = live.step(action)
                    raw_return += float(reward)
                    elapsed += int(info["substeps_executed"]) * float(base.mj_model.model.opt.timestep)
                    trace.append(base._center_of_mass().tolist())
                    collapsed = bool(info.get("terminated", False))
                    if done:
                        break
                displacement = (trace[-1][0] - trace[0][0]) / diagonal
                row = dict(seed=int(seed), controller=controller, initial_state_sha256=snapshot_hash,
                           displacement=float(displacement), raw_return=raw_return, collapsed=collapsed,
                           executed_steps=len(control_actions), elapsed_seconds=elapsed,
                           planning_seconds=latencies, simulator_physics_steps=sim_steps,
                           node_mean_position=trace, actions=control_actions)
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("seed", "controller", "displacement", "collapsed", "executed_steps")}), flush=True)
                save(output, settings, cfg, rows, total_start, dt)
    finally:
        live.close()
        planning.close()


def save(output, settings, cfg, rows, total_start, dt):
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(settings=OmegaConf.to_container(settings), cem=asdict(cfg),
                   revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   working_tree_changes=True, control_interval_seconds=dt,
                   wall_seconds=time.perf_counter() - total_start, results=rows,
                   claim_boundary="Source development diagnostic; no held-out or learned-model results.")
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/dynamics_pilot.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/dynamics-pilot/simulator-mpc.json")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--candidates", type=int)
    args = parser.parse_args()
    settings = OmegaConf.load(args.config)
    if args.steps is not None:
        settings.episode_steps = args.steps
    if args.seeds is not None:
        settings.reset_seeds = args.seeds
    if args.candidates is not None:
        settings.cem.candidates = args.candidates
    run(settings, args.output)
