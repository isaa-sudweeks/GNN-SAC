# Architecture

This page describes the graph SAC pipeline as it is currently implemented. The
early design proposals it replaces (edge-action decoders, edge features) are in
[`archive/`](../archive/).

```text
mujoco-truss-gen model
   │  env/mujoco_gen/topology_envs.py  (native)   or   mjx_vector_env.py (MJX)
   ▼
graph observation dict {x, edge_index, action_mask, rigidity[, edge_role]}
   │  env/wrappers → PyG Data;  sac/common/graph_transforms.prepare_graph
   ▼
GNN actor ──► one action per actuated control node ──► NodeVelocityController ──► tendon actuator ctrl
GNN critics ◄── node features ⊕ node actions  ──► pooled graph readout ──► Q(s, a)
```

## Graph observation

The environment builds each observation from the `mujoco-truss-gen` model:

| Field | Shape | Contents |
|---|---|---|
| `x` | `[num_nodes, 6]` | Node position relative to the xy center of mass, and node velocity. Both are divided by the initial bounding-box dimensions when `normalize_observations=true`. |
| `edge_index` | `[2, num_directed_edges]` | Directed message edges of the selected graph view. |
| `action_mask` | `[num_nodes]` | `True` for nodes that receive a policy action (not passive). |
| `rigidity` | `[1]` | Current rigidity normalized by its initial value. |
| `edge_role` | `[num_directed_edges]` | Only present with `graph_features.edge_roles`. |

**Which nodes appear** depends on the graph view:

- **`control`**: used when `use_control_graph=true`, the default for the GNN
  and padded-MLP backends, and always used by MJX. Nodes are the control nodes
  that `mujoco-truss-gen` defines for routed node-velocity control.
- **`physical`**: every physical node of the model.
- **`logical`**: realistic models with cloned nodes regrouped into logical nodes.

See [configuration.md](../usage/configuration.md#realistic-models-and-graph-views)
for how the view is chosen.

## Graph preparation

`prepare_graph` (`sac/common/graph_transforms.py`) optionally extends the raw
graph before it reaches the network:

- **Virtual node** (`use_virtual_node`, on for the GNN). Adds one global node
  connected to every physical node. Its features are an `is_virtual` flag and
  the normalized rigidity. Physical nodes get two zero context channels so every
  node has the same feature width.
- **Node roles** (`graph_features.node_roles`). Appends an `[actuated, passive]`
  one-hot to each node, taken from `action_mask`.
- **Edge features** (`graph_features.edge_roles`, `graph_features.edge_distance`).
  Builds `edge_attr` from edge-role one-hots and endpoint distance.

## Networks (`sac/common/gnn_actor_critic.py`, `gnn_layers.py`)

**Message passing.** Each layer computes a message
`phi([x_i, x_j, edge_attr])`, sums messages over incoming edges, and updates the
node with `gamma([x_i, aggregate])`.

- `message_attention` optionally weights each incoming message with a softmax
  over the edges arriving at a node.
- `mpl_dims` sets depth and widths. `mpl_skip_connections` adds residual
  connections between layers.

**Actor.** A GNN produces node embeddings. An MLP head maps each embedding at an
`action_mask` node to a tanh-squashed Gaussian, giving one scalar action per
actuated node (`node_action_dim=1`).

**Entropy normalization.** Per-node log-probabilities are *averaged* within each
graph. This keeps the entropy scale independent of robot size, and it is why
`target_entropy=-1`.

**Critics.** An ensemble of `num_q` critics, each with its own GNN and Polyak
target copy.

- Input: the node features concatenated with the node actions (zeros at nodes
  without actions).
- Readout (`critic_readout`): mean over physical nodes, the virtual-node
  embedding, or both concatenated (the GNN default).

## From node actions to actuators

**Control graph (default).** Node actions are clipped to `[-1, 1]` and scaled by
`speed` to give node velocity commands.
`mujoco-truss-gen`'s `NodeVelocityController` then turns them into clipped
tendon actuator commands using the control-graph incidence and routing. The
number of actuators is independent of the number of nodes. For example, the
octahedron has 6 policy actions and 8 actuators.

**Broken nodes.** With `domain_randomization_params.broken_nodes` enabled, an
episode can disable some originally active control nodes. They are removed from
`action_mask`, and their commands are zeroed before routing. The graph topology
and the order of action rows stay the same.

**Without the control graph (legacy).** Each actuated tendon's command is the
sum of its two endpoint node actions, clipped to `[-1, 1]`.

## Reward and termination

The reward terms are set in `config/environment.yaml`:

- forward center-of-mass velocity (`forward_weight`)
- squared actuator command penalty (`energy_weight`)
- alive bonus (`alive_bonus`)
- normalized rigidity (`rigidity_weight`)
- ground-contact slip penalty (`slip_weight`, `slip_height`)

An episode terminates when normalized rigidity falls below
`critical_eig_threshold`, and truncates at `max_steps`. Rewards are normalized
per task by the variance of the discounted return (`normalize_rewards`).

## Replay and updates

**Replay** is task-balanced, with one buffer per topology. Every batch draws
`batch_size / num_topologies` samples from each topology and collates them into
a PyG `Batch`.

- The default backend, `replay_backend=torchrl_tensor`
  (`sac/common/tensor_gnn_buffer.py`), gives each topology a TorchRL
  `TensorDictReplayBuffer`. Only the changing tensor fields are stored; graph
  structure, action masks, and edge roles are stored once per topology.
- `replay_storage` selects where replay lives (`auto`, `cpu_pinned`, or `cuda`).
- `replay_backend=legacy` (`GNNBuffer` in `sac/common/gnn_buffer.py`) stores
  PyG objects in Python lists and is kept for reproduction and rollback.

See [replay.md](../usage/replay.md) for details.

**Updates** (`GNNSAC` in `sac/gnn_sac.py`) follow standard SAC with automatic
temperature tuning.

- With `pcgrad=true`, critic and actor/temperature gradients are computed per
  topology and projected with PCGrad before the optimizer step.
- `gradient_diagnostics` logs cosine similarity between per-task gradients
  without changing the update.
- With `distillation=kl`, the actor loss also includes a decaying
  KL(teacher ‖ student) term against frozen per-topology teacher policies. See
  [distillation.md](../usage/distillation.md).

## Padded MLP baseline

`sac_backend=padded_mlp` uses the same environment, observations, action
routing, replay, and entropy averaging. It swaps the GNN for an MLP over a fixed
21-slot flattened node vector with existence and action masks, and it never
reads `edge_index`. See [padded_mlp_baseline.md](../usage/padded_mlp_baseline.md).
