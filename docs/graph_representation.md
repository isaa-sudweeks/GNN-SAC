# Graph Representation

This document defines the planned graph data contract for GNN-SAC. The first implementation should keep the representation explicit and testable.

## Recommended Data Structure

Use a PyTorch Geometric-style graph object for each observation.

Expected fields:

```python
Data(
    x=node_features,
    edge_index=edge_index,
    edge_attr=edge_features,
    actuated_edge_mask=actuated_edge_mask,
    global_attr=global_features,
)
```

Additional transition fields should live outside the observation graph:

```python
GraphTransition(
    obs=graph_t,
    action=edge_actions_t,
    reward=reward_t,
    done=done_t,
    next_obs=graph_t_plus_1,
)
```

## Node Features

`x` should have shape:

```text
[num_nodes, node_feature_dim]
```

Candidate node feature layout:

```text
[
  position_x,
  position_y,
  position_z,
  velocity_x,
  velocity_y,
  velocity_z,
  node_type_one_hot...,
  optional_physical_parameters...
]
```

Recommended first version:

- positions expressed relative to robot center of mass
- velocities expressed in a consistent frame
- explicit node type features only if topologies include heterogeneous node roles

Tests should verify:

- node feature shape is stable for one topology
- node ordering changes do not change graph-level critic output beyond numerical tolerance, once permutation tests are implemented
- no NaN or infinite values are produced by the environment wrapper

## Edge Index

`edge_index` should have shape:

```text
[2, num_edges]
```

Recommended first version:

- Store each physical undirected edge as two directed message-passing edges.
- Store actions only for canonical actuated physical edges, not both directed copies.

This creates two related edge concepts:

- message edges: directed edges used by the GNN
- actuator edges: physical actuated tendons that receive actions

The implementation should make this distinction explicit. Do not infer actuator edges by slicing message edges unless the ordering contract is documented and tested.

## Edge Features

`edge_attr` should have shape:

```text
[num_message_edges, edge_feature_dim]
```

Candidate edge feature layout:

```text
[
  edge_type_one_hot...,
  rest_length,
  current_length,
  length_error,
  relative_speed_along_edge,
  stiffness,
  damping,
  is_actuated
]
```

Recommended first version:

- include `is_actuated`
- include static physical attributes that differ across topologies
- include dynamic length features if they are not easily reconstructable from node positions

## Actuated Edge Metadata

The actor needs a stable way to map graph embeddings to actuator commands.

Recommended fields:

```python
actuator_edge_index      # shape [2, num_actuators]
actuator_edge_attr       # shape [num_actuators, actuator_feature_dim]
actuator_to_message_edge # optional mapping into edge_index
```

For each actuated tendon, the edge-action decoder should receive:

- source node embedding
- target node embedding
- actuator edge attributes

Then it should output:

```text
mean:    [num_actuators, action_dim_per_actuator]
log_std: [num_actuators, action_dim_per_actuator]
```

Recommended first version:

Use `action_dim_per_actuator = 1`.

## Actions

For one graph:

```text
edge_actions: [num_actuators, action_dim_per_actuator]
```

For a batched graph:

```text
edge_actions: [total_num_actuators_in_batch, action_dim_per_actuator]
```

The replay buffer must store actions in actuator-edge order, not message-edge order.

Tests should verify:

- action count equals number of actuated tendons
- action bounds are respected after squashing and scaling
- actions can be mapped back to simulator actuators without ambiguity

## Critic Input

The critic should consume graph state plus edge actions and output one scalar per graph.

Recommended approach:

1. Encode node and edge state with a GNN.
2. Attach actuator actions to the corresponding actuator edge representations.
3. Pool graph information to one graph embedding.
4. Output `Q(s, a)` as a scalar.

For a batch:

```text
q_values: [batch_size, 1]
```

Open design choices:

- Whether action information enters before, during, or after message passing
- Whether to pool node embeddings, edge embeddings, or both
- Whether to use sum, mean, or attention pooling

Initial recommendation:

Use mean pooling for normalized graph-level embeddings, and explicitly test whether larger graphs get biased Q magnitudes.

## Replay Buffer Contract

The replay buffer must store graph transitions, not flattened observations.

Minimum stored fields:

- graph observation
- edge action tensor
- scalar reward
- done flag
- next graph observation
- optional timeout flag

Batching requirements:

- Preserve per-graph batch indices.
- Preserve per-actuator graph ownership.
- Allow different numbers of nodes, edges, and actuators per sample.

Tests should verify:

- a mixed-topology batch can be sampled
- graph boundaries are preserved
- rewards and done flags align with the right graphs
- edge actions align with the right actuator edges

## Permutation Robustness Checks

The graph policy should be permutation equivariant with respect to node and edge ordering.

Required checks:

- Permuting node order should permute per-actuator outputs consistently.
- Permuting non-actuator message edge order should not change actuator outputs except for numerical tolerance.
- The critic output should remain invariant under equivalent graph permutations.

These tests should become part of the core test suite before relying on cross-topology results.
