# Dynamics and MPC prerequisite pilot

The first implementation provides a source checkpoint auditor, physical-unit
coordinate helpers, a native MuJoCo branch evaluator, smooth bounded CEM action
proposals, and indexed evaluation collection. It does **not** yet fit a dynamics
model or evaluate held-out transfer. Settings are in `config/dynamics_pilot.yaml`.

Use the project's uv environment. A reusable external uv environment can be
selected with `UV_PROJECT_ENVIRONMENT`; `--no-sync` uses that existing environment.

```bash
uv run python -m unittest tests.test_dynamics_planner tests.test_dynamics_simulator tests.test_dynamics_state tests.test_dynamics_replay_audit -v
uv run python scripts/audit_dynamics_replay.py /trusted/source/checkpoint.pt --output outputs/dynamics-pilot/replay.json
uv run python scripts/probe_dynamics_state.py
uv run python scripts/benchmark_dynamics_simulator.py
uv run python scripts/run_simulator_mpc_pilot.py
uv run python scripts/collect_dynamics_source_probe.py
uv run python scripts/validate_dynamics_source_probe.py
uv run python scripts/save_dynamics_reset_manifest.py
uv run python scripts/plot_dynamics_pilot.py
```

The checkpoint auditor opens only an explicitly selected trusted checkpoint.
It filters replay tasks against the eight `random_5fold/fold_4` source topologies
before inspecting rows. Legacy generic specialist task names are resolved only
from an unambiguous saved topology config. It samples 128 rows with seed zero,
records valid circular-buffer indices and a checkpoint hash, and never assumes
that adjacent replay rows form trajectories. Its one-step candidate status is
not state-sufficiency approval. Only sampled rows receive finite-value checks.

All live simulator diagnostics are restricted to source tetrahedron, nominal
abstract physics, control-graph actions, and no rendering. The branch evaluator
uses a separate environment, copies complete `MjData`, and restores explicitly
audited environment/wrapper counters, caches and RNG. New wrappers, realistic
controllers, or randomized physics require another restoration audit.

The planner proposes bounded node commands with three linearly interpolated
knots. It scores actual simulator rollouts, executes one action, and replans.
The previous plan is shifted by one step for the next proposal mean. It retains
the best actually evaluated candidate across iterations. The diagnostic config
uses 32 candidates, two iterations and ten control steps; this is a measured-cost
source development setting, not the original 256-by-3 proposal or frozen holdout
configuration. The original matrix is expensive on native CPU simulation.

Its objective is forward **physical-node mean** displacement divided by initial
bounding-box diagonal, minus 0.1 times the time integral of mean squared routed,
clipped actuator commands. A collapsed/nonfinite candidate pays 1 plus its
remaining horizon in seconds. No additional terminal reward is used. These
weights are initial source settings, not established optimal values. Full raw
environment return is reported separately. Neither command effort nor planning
score is SAC return. The horizon spans 2 seconds at a 0.2-second control interval;
the three knots occur at 0, 0.9 and 1.8 seconds (sampled on the control grid).
The diagnostic episode spans 8 simulated seconds.

`run_simulator_mpc_pilot.py` pairs zero, smooth random and simulator MPC using exact
copied initial states for five reset seeds. It retains individual actions,
position traces, collapse status, raw returns and per-decision latency. Reset
variation is the nominal generator's seeded noise; it is not a broad distribution
of orientations or near-failure conditions. The random control uses clipped
normal knots every approximately five steps, while MPC uses its configured
planning-horizon knots. Both use the same bounds, initial scale and action mask.
This is an initial smooth-random control, not a matched search-compute comparator.

The source evaluation collector saves four independent 40-step episodes with
episode/environment/timestep IDs, separate termination/truncation, true next
observations, raw reward components, elapsed time, physical positions, actuator
activation, external controls and MuJoCo integration snapshots. Each episode is
a separate compressed file with a SHA-256 manifest. Schema v2 additionally saves
full `MjData` and wrapper snapshots because integration state alone did not
reproduce the raw reward exactly. These trusted local pickle files preserve
derived caches; the validator checks saved labels against restored transitions.
Existing datasets are never
overwritten. These 160 probe transitions are evaluation-only and excluded from
fitting and checkpoint selection; they demonstrate logging, not action coverage.

## Replay eligibility and observable-input baseline

The audited teacher checkpoint has one million transition pairs but no episode,
environment or timestep IDs, truncation field, or actuator activation state.
A hidden-activation intervention showed that visible node features and commands
do not uniquely determine exact simulator transitions. It did not establish that
approximate prediction on normal trajectories is inadequate for MPC.

Begin with physical-node positions and velocities, available structural/routing
metadata and known commands; initially use three message-passing layers of width
128. Hidden actuator targets and full simulator snapshots are diagnostic ground
truth, not required deployment-facing model features. Check units, centered-xy
translation bookkeeping, physical-node masks, balanced graph losses and recursive
geometry refresh before fitting. Never refresh imagined state from future truth.

Before reusing each source replay, trace clean/noisy observation and executed-action
labels and verify backend/settings compatibility. The saved configuration enables
observation/action noise and uses MJX. Missing actuator targets alone do not reject
one-step fitting. Missing episode provenance prevents sequence validation. Reuse
compatible transitions for one-step fitting; otherwise collect nominal source
rollouts with independent fitting and validation episodes. Keep the evaluation-only
probe excluded from fitting, normalization and checkpoint selection. Diagnose source
errors and action coverage before adding history or changing model capacity.

See `docs/plans/gnn_dynamics_mpc_pilot.md` for the active execution plan and gates.

Results under `outputs/dynamics-pilot/` are generated artifacts and must not be
committed. Experiment notes and figures live in the user's Obsidian notebook.
