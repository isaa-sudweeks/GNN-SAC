# Domain Randomization Test Plan

## Goal

Verify that each configured range reaches `mujoco-truss-gen`, is sampled at reset,
changes only the intended model quantity, remains numerically stable, and improves
robustness without hiding regressions in the nominal environment.

## 1. Fast contract tests (every change and CI)

- Compare this repository's runtime-range mapping with every `*_range` field in
  `DomainRandomizationConfig`. This detects upstream API additions and removals.
- Load `config/physics/domain_randomization.yaml` and require an entry for every
  mapped runtime range.
- Pass distinct deterministic ranges (`min == max`) through `_domain_randomization`
  and assert that the resulting upstream config contains every exact range.
- Reject missing, non-finite, reversed, and unsupported enabled ranges with a clear
  configuration error.

## 2. Backend reset tests (CI where MJX is available)

- Enable all fixed-shape ranges with distinct deterministic values, reset a batch,
  and assert every value in `MjxDomainRandomizationState` for every environment.
- For `abstract_node_mass_multiplier`, assert the state shape is
  `[batch_size, node_count]`, node samples are independent, and selective resets
  replace only masked batch rows.
- Reset with non-degenerate ranges and assert samples are in bounds and that at
  least two batch members differ for each field.
- Repeat the deterministic-value test with native MuJoCo and inspect the compiled
  model arrays after reset. Keep model-factory tests (length scale and physical
  parameters) native-only because they rebuild model structure.
- Check seeded reproducibility: identical seeds produce identical samples; changed
  seeds produce at least one different sample.

## 3. Simulation smoke matrix (nightly or before training)

Run 100 resets and 100 steps per case for `octahedron` and one realistic topology:

1. nominal/no randomization;
2. each randomization family alone (global/per-node body, DOF, actuator, geom,
   tendon, gravity);
3. all fixed-shape fields together at the intended training ranges;
4. native model-level geometry and physical-parameter randomization;
5. all native randomization together.

For every run require finite observations, rewards, controls, positions, velocities,
and accelerations; valid observation/action shapes; and no MuJoCo warnings. Record
reset/step throughput relative to nominal.

## 4. Range calibration

- Start with narrow physically plausible ranges and visualize sampled distributions.
- Run one-at-a-time sensitivity sweeps at minimum, nominal, and maximum values.
- Flag values that cause immediate collapse, solver instability, actuator saturation,
  or qualitatively different task definitions. Narrow or use curricula for those.
- Verify coupled constraints explicitly (for example inertia with mass, tendon
  stiffness with damping, and friction axes together).

## 5. Learning and robustness experiment

Train at least three seeds for: nominal baseline, randomization by family, and the
combined candidate distribution. Evaluate every checkpoint without training noise on:

- nominal parameters;
- in-distribution held-out samples;
- boundary values for every enabled range;
- modest out-of-distribution values beyond each boundary;
- both native MuJoCo and MJX when the topology supports both.

Compare return, failure/collapse rate, forward velocity, energy, rigidity, seed
variance, and worst-decile return. Adopt the combined distribution only if it improves
held-out and boundary robustness without an unacceptable nominal-performance or
training-throughput regression.
