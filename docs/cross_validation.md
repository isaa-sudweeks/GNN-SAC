# Leave-one-configuration-group-out cross-validation

Cross-validation definitions live in `config/cross_validation/`. Each enabled
definition contains at least two named development groups and may record a final
test set that is excluded from every generated job:

```yaml
# @package _global_
cross_validation:
  enabled: true
  name: node_count_loso
  groups:
    node_4:
      - tetrahedron
    node_6:
      - octahedron
      - henneberg_n6_1tube_2
  final_test:
    - henneberg_n7_1tube_1
  held_out_group: null
```

Launch all folds and seeds locally with:

```bash
python scripts/launch_cross_validation.py cross_validation=node_count_loso \
  --seeds 1,2,3,4,5 --shuffle-seed 17 platform=local
```

Use `platform=supercomputer` to submit the same matrix through Submitit. Add
`--dry-run` to write and inspect the launch manifest without starting jobs. A
custom manifest path can be selected with `--manifest PATH`.

For each fold, every group except the selected holdout becomes
`truss_topologies`; the holdout becomes `eval_extra_topologies`. The latter is
evaluated periodically in native MuJoCo but never enters environment collection,
replay, or normalization. `episode_*` remains the training-topology aggregate,
`heldout_episode_*` is the held-out aggregate, and `all_episode_*` combines both.

Group membership must be disjoint. Exact repeated identifiers are rejected, but
the launcher cannot infer that differently named routing, partition, scale,
realistic, or randomized variants came from one underlying topology. Keep all
such related variants in the same group to avoid leakage.

The optional `final_test` list is recorded and checked for overlap, but the
launcher never trains on or evaluates it. Cross-validation is development
evidence; run final-test evaluation separately only after freezing the method,
hyperparameters, checkpoint-selection rule, and metrics.
