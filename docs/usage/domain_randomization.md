# Physical parameters and domain randomization

## Nominal physical parameters

`config/physics/physical_parameters.yaml` holds the nominal
`mujoco-truss-gen` physical values (node radius and mass, tendon ranges,
actuator gains, and so on). Training and inference both use them. Set
`physical_parameters_enabled: false` to use the package defaults instead.

## Domain randomization

`config/physics/domain_randomization.yaml` contains all randomization settings.
`domain_randomization` is the master switch; each family under
`domain_randomization_params` has its own `enabled` flag and a `min`/`max` range.

| Family | Keys | Backends |
|---|---|---|
| Fixed-shape runtime ranges | `body_mass_multiplier`, `abstract_node_mass_multiplier`, `body_inertia_multiplier`, `dof_*`, `actuator_*`, `geom_friction_{slide,torsional,rolling}`, `tendon_*`, `gravity_z`, `hinge_position_kp` | Native MuJoCo and MJX |
| Initial pose | `initial_translation_x`, `initial_translation_y`, `initial_yaw` | Native MuJoCo and MJX |
| Model-rebuilding | `length_scale`, `physical_parameters.*` | Native MuJoCo only |
| Rollout noise | `action_noise`, `observation_noise` | Training rollouts only |

Notes:

- `abstract_node_mass_multiplier` draws a separate multiplier for every node.
  It is valid only with `truss_realistic=false` and multiplies with
  `body_mass_multiplier`.
- `hinge_position_kp` sets the absolute servo gain for the internal connector
  hinges in realistic models. `physical_parameters.hinge_position_kp` is a
  deprecated alias for it.
- Model-rebuilding randomization recompiles the MuJoCo model at every reset. The
  MJX backend rejects it.
- Observation noise is applied only to graph node features (`x`). Executed noisy
  actions are clipped and stored in replay.

Example:

```yaml
domain_randomization: true
domain_randomization_params:
  body_mass_multiplier:
    enabled: true
    min: 0.8
    max: 1.2
  abstract_node_mass_multiplier:
    enabled: true
    min: 0.8
    max: 1.2
```

## Validation

- `scripts/validate_domain_randomization.py` runs the checks in
  [plans/domain_randomization_test_plan.md](../plans/domain_randomization_test_plan.md)
  and writes machine-readable results.
- `scripts/run_domain_randomization_training_smoke.py` runs a three-seed
  training smoke test for each randomization family.
