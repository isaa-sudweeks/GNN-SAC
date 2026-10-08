# Tube-specific virtual nodes: feasibility and experiment plan

Status: grouping-only model, replay/distillation integration, regression tests,
and offline/online launch protocol implemented on `codex/implement-tube-virtual-nodes`.
Numerical research experiments, GPU canary, physical-length features, and optional
shuffled-group control pending. See [usage](../usage/tube_virtual_nodes.md).
Branch: `codex/tube-virtual-nodes`, based on `codex/signed-routing-ablation`
at `ec1e6fe`. Research date: 2026-10-07.

## Assessment

One non-actuated aggregation node per tube is a physically motivated extension.
It exposes a group relation and lets the network maintain a separate summary for
each conserved tube. It is worth testing, especially on long routed topologies.
It is not yet evidence that tube grouping is the missing cause of transfer failure.

Importantly, tube membership is already recoverable from the signed arm's typed
control graph: remove connector edges and take connected components of structural
`actuated` edges. The upstream label includes tube edges with no direct actuator;
using only `actuator_edges` would omit required members. The improvement is an
explicit representation and useful aggregation path, rather than necessarily new
information. A triangle already has pairwise links between all three members.
Adding a hub does not shorten communication within that triangle. A long routed
tube benefits more: any two members can exchange a tube summary in two layers.
The existing global virtual node also connects all nodes within two layers, but
does not maintain a separate summary for each tube.

## Verified simulator contract

Inspected the installed `mujoco-truss-gen==0.12.5`, matching `pyproject.toml` and
`uv.lock`, including `presets.py`, `builders.py`, `control_graph.py`,
`constraints.py`, `geometry.py`, and `controllers.py`.

- Usevitch definitions select an edge-disjoint triangle partition. Each selected
  triangle, rather than every geometric triangle found in the logical graph,
  corresponds to a tube.
- `_triangle_control_graph_metadata` creates three control-node instances per
  triangle. Instances sharing a logical vertex are joined by connector edges.
- `_shape_control_graph_metadata` creates an instance per route occurrence.
  Repeated logical vertices remain distinct control nodes, even within one route.
  Routes are open paths in these presets; do not automatically close them.
- `add_perimeter_constraint` creates one tendon equality per triangle.
  `add_route_length_constraints` creates one equality per enabled shape route.
  These are MuJoCo soft constraints: conservation is a target, not a promise of
  exact zero numerical residual at every step. Custom shapes can disable route
  length constraints; the control metadata alone does not record that flag.
- `ControlGraphMetadata` stores connectivity and actuator routing but has no
  explicit tube IDs or conserved-length fields. Tube grouping can be reconstructed
  for the inspected builders; authoritative source route/equality metadata is
  preferable for a general adapter and custom XML validation.

Native CPU environment construction/reset with domain randomization disabled
verified the following. Source tube sizes and reconstructed component sizes
matched; the compiled model had one expected perimeter/route equality per tube.
Counts refer to raw control graph instances, before architectural virtual nodes.

| Preset | Logical/abstract physical nodes | Control nodes | Tubes | Members per tube |
|---|---:|---:|---:|---|
| octahedron | 6 | 12 | 4 | 3, 3, 3, 3 |
| tetrahedron | 4 | 8 | 2 | 4, 4 |
| usevitch_212365307 | 8 | 18 | 6 | 3, 3, 3, 3, 3, 3 |
| henneberg_n8_2tube_127 | 8 | 20 | 2 | 10, 10 |
| henneberg_n8_2tube_134 | 8 | 20 | 2 | 10, 10 |

An additional source-level audit passed for all 19 development presets in
`config/cross_validation/random_5fold.yaml`. It called each preset definition and
the corresponding upstream control-metadata builder, removed connector edges,
and compared each component's multiset of logical nodes against the source
triangle/route. Exact multisets matched, including repeated route occurrences.
This audit did not construct 19 simulator environments or run policies. The three
N7 final-test presets were excluded. Both audits used the existing shared Python
environment at `/Volumes/External_Drive/Research/Robotics_Research/GNN-SAC/.venv/bin/python`.

## Proposed implementation

Retain signed physical edges, connector edges, and the existing global node.
Add one tube node connected bidirectionally only to its own control-node members.
Do not attach it to connector neighbors, other tubes, or the global node directly.
For a shared logical vertex, attach each control instance to its owning tube;
existing connector edges transmit the coupling between tubes.

Start with a grouping-only ablation: a tube-node type flag, a distinct membership
edge type, false action/physical masks, and zero incidence signs on membership
edges. Keep global rigidity exclusively on the global node. Physical observation
normalization must exclude architectural nodes. Do not interpret the tube node's
zero xyz channels as a physical point or calculate Euclidean membership distance.

Then test physical tube features separately. The idealized tube relation is
`sum(segment_lengths) = L_t`, and its time derivative is
`sum(segment_length_rates) = 0`. A bare membership node tells the network which
members share a constraint; it neither supplies this equation nor enforces it.
Useful features include member/segment count, the conserved reference length,
current total-length/reference-length ratio, and signed length residual.
Retrieve the reference from the simulator's actual equality/initialization
contract; never replace it with a newly computed current length each step.
Use actual tendon lengths where possible. Per-axis bounding-box normalization
of xyz is anisotropic, so lengths computed from normalized xyz are not generally
physical lengths scaled by a single constant. Normalize physical lengths with
one documented scalar scale. Preserve route order and repeated occurrences in
metadata for future segment-level features; a group hub alone loses that order.

The current attention weights are softmax-normalized over incoming neighbors.
A tube hub therefore produces a weighted summary, not an exact total. Explicit
counts/length features or a sum-based tube aggregation are needed if a conserved
sum is to be represented directly. Test these separately from membership alone.

## Code integration points and correctness risks

| Location | Required extension |
|---|---|
| `env/mujoco_gen/topology_envs.py`, `mjx_vector_env.py` | Validate and expose static tube membership in raw control-node order; physical features require matching native/MJX calculations. |
| `env/wrappers/tensor.py`, `multitask.py` | Preserve and validate membership metadata through observation conversion. |
| `sac/common/graph_transforms.py` | Add typed tube nodes/edges, masks, feature-schema dimensions, and structure signatures. |
| `sac/common/gnn_layers.py` | Select the global node explicitly for critic readout. Current `x[~physical_mask]` assumes exactly one architectural node per graph and would also select all tube nodes. |
| `sac/common/gnn_actor_critic.py` | Replace the fallback physical count `total_nodes - num_graphs` with an explicit count/mask. |
| `sac/common/tensor_gnn_buffer.py` | Extend supported raw fields, static templates, checkpoint signatures, and dense feature rebuilding. Current rigidity update targets the last node; preserve an explicit global index/mask. |
| `sac/common/distillation.py` | Reconstruct membership for legacy replay, enrich student observations, and preserve teacher feature schema/predictions. |
| `env/__init__.py`, configuration, inference | Resolve dimensions and feature flags consistently across actor, critics, resume, and inference. |

If raw membership uses a two-row node/tube index tensor, explicitly handle PyG
batch offsets: physical-node IDs and tube IDs have different increments.
An unqualified field ending in `index` uses PyG defaults that can offset the tube
row incorrectly. Per-topology replay templates can avoid this ambiguity.

Adding T tube nodes with membership sizes k_t adds T nodes and
`2 * sum(k_t)` directed message edges. For the inspected Usevitch preset this is
6 nodes and 36 edges; for either inspected Henneberg preset it is 2 nodes and
40 edges. Existing graph layers support variable node counts, but the masks,
readout, and replay assumptions above must be updated first.

## Experiment and acceptance criteria

Compare the same signed baseline against signed plus tube membership, then
signed plus membership and conserved-length features. Keep teachers, attention,
global node, width/depth, seeds, split, budgets, and KL schedule fixed. Match
trainable parameter counts where practical and report changes. A size-matched
shuffled-group hub control can distinguish physically correct grouping from
generic extra capacity or shortcut edges.

First screen held-out teacher Gaussian KL/action MSE after offline distillation,
then deterministic distance and survival across reset seeds. Report per-topology
results and all three training seeds. Preserve N7 final-test isolation and do not
select checkpoints by holdout performance. Measure runtime/memory overhead.
An offline prediction gain alone is insufficient: check that rollout behavior
improves and that any benefit survives online learning.

Before training, verify exact membership against source definitions, node/tube
relabeling equivariance, reverse-routing sign consistency, no actions on tube
nodes, batched mixed-topology critic output shape, global readout selection,
object/dense replay parity and resume, legacy teacher prediction preservation,
and native CPU plus MJX inference parity. Numerical experiments are pending.

## Research sources

- [Usevitch et al., An untethered isoperimetric soft robot (2020)](https://msl.stanford.edu/papers/usevitch_untethered_2020.pdf): physical motivation for tube-wide conserved length.
- [Heydari and Livi, Message Passing Neural Networks for Hypergraphs (2022)](https://arxiv.org/html/2203.16995v2): star expansion and vertex-to-hyperedge-to-vertex message passing; typed partitions avoid ambiguity.

These sources motivate the representation. They do not demonstrate improved
SAC transfer for this repository.
