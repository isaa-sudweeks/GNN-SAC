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
uv run python scripts/launch_cross_validation.py cross_validation=node_count_loso \
  --seeds 1,2,3,4,5 --shuffle-seed 17 platform=local
```

The launcher starts each Hydra multirun with `uv run python`, so jobs use the
locked project environment. Use `platform=supercomputer` to submit the same matrix through Submitit. Add
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

## Random-configuration folds

`node_count_loso` holds out an entire node count, so every fold measures
extrapolation to an unseen size. Its development pool contains 19 topologies
with completed paper-v4 teachers: one four-node, one five-node, five six-node,
eleven eight-node, and one nine-node topology. The three seven-node topologies
remain exclusively in `final_test`.

To compare against random configuration holdouts, `random_5fold` pools the same
19 development topologies and partitions them randomly into five folds of
three or four, ignoring node count. The `final_test` set is copied from the
source and stays excluded.

The split is generated once and committed so every seed and relaunch uses the
same folds:

```bash
uv run python scripts/make_random_cross_validation.py \
  --source node_count_loso --num-folds 5 --split-seed 0 --name random_5fold
```

The generated YAML records `cross_validation.split` (source, fold count, seed),
which reaches the W&B config; `tests/test_cross_validation.py` fails if the
committed file drifts from what the generator produces. Launch it like any other
definition, e.g. `uv run python scripts/launch_cross_validation.py cross_validation=random_5fold`.

Some node counts have a single development topology (4, 5, and 9 nodes), so a
random fold that holds one of them out is still an unseen-size fold. Compare
per-topology `eval/<topology>_episode_reward` against the matching
`node_count_loso` run rather than only the fold aggregate.

## Farthest-point morphology clusters

`farthest_point_5fold` uses the same 19 development topologies and final-test
reservation. It selects five distant morphology prototypes by farthest-point
sampling, then assigns each other topology to its nearest prototype. Each CV
job holds out one complete cluster. Similar morphologies are therefore held
out together, whereas random folds can spread similar morphologies across
training and evaluation. Cluster sizes may differ; compare per-topology
outcomes as well as fold aggregates.

The distance is Euclidean in 12 descriptor coordinates, standardized by the
development pool's population mean and standard deviation. Constant features
contribute zero. Descriptors come from the physical preset definitions, not
the policy's graph transforms or training results:

- Physical node, unique undirected edge, and tube counts.
- Degree standard deviation, mean clustering coefficient, mean shortest-path
  distance, and graph diameter.
- Active-edge fraction and coefficient of variation of tube route lengths.
- Edge-length coefficient of variation and the two smaller-to-largest
  eigenvalue ratios of centered position covariance.

Distances ignore translation, rotation, uniform scale, and node labels. They
describe graph, routing, and initial geometry; they do not prove separation in
learned policy behavior. All coordinates receive equal weight after
standardization, so correlated descriptors can reinforce each other. The first
prototype is farthest from the development centroid; subsequent prototypes
maximize distance to their nearest selected prototype. Sorted topology names
break ties. Each prototype stays in its own cluster, including when descriptors
are identical. No reward metrics, teacher selection scores, or final-test
presets are used to construct the split.

Regenerate and launch with:

```bash
uv run python scripts/make_farthest_point_cross_validation.py \
  --source node_count_loso --num-folds 5 --name farthest_point_5fold

uv run python scripts/launch_cross_validation.py cross_validation=farthest_point_5fold \
  --seeds 1,2,3 platform=supercomputer distillation=kl_paper_v4
```

`distillation=kl_paper_v4` loads the saved mapping in
`config/distillation/kl_paper_v4.yaml`; it also works with `node_count_loso` and
`random_5fold`. Each fold loads only its training teachers.
Use the same student settings, teacher map, and training budget across split
methods. Add `--dry-run` to either command to inspect without writing the split
or launching training, respectively.

The frozen YAML records the descriptors, their names, selected prototypes,
distance rule, algorithm version, and `mujoco-truss-gen` version. Training reads
the saved groups directly and never reruns sampling. Tests reproduce the groups
from the saved descriptors and verify source-pool coverage and final-test
isolation. Regeneration evaluates the procedural presets and can take several
minutes; after a preset/library update, review the new descriptor values and
group membership before using a changed split.
