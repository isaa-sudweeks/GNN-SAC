# Node to Edge Action Ideas

The current GNN produces actions for each node. The control problem may need actions on edges instead, so this note summarizes a few ways to convert node-level outputs into edge-level actions.

## 1. Sum Endpoint Node Actions

For an edge connecting nodes `i` and `j`:

```text
edge_action(i, j) = node_action(i) + node_action(j)
```

This is the simplest baseline.

Pros:
- Easy to implement.
- Symmetric for undirected edges.
- May work if an edge action should represent the combined influence of its two endpoint nodes.

Cons:
- Assumes both endpoint actions contribute additively.
- Cannot distinguish direction, so `(i, j)` and `(j, i)` produce the same edge action.
- Cannot express interactions like disagreement, cancellation, or edge-specific behavior.

Use this as a quick baseline, but it is probably too restrictive as a final design unless the physics clearly supports it.

## 2. Learn Edge Actions From Endpoint Node Actions

Instead of hard-coding the sum, feed the two endpoint node actions into a small MLP:

```text
edge_action(i, j) = MLP([node_action(i), node_action(j)])
```

For undirected edges, use symmetric features:

```text
edge_action(i, j) = MLP([
    node_action(i) + node_action(j),
    abs(node_action(i) - node_action(j))
])
```

Pros:
- More expressive than a raw sum.
- Can learn sum-like behavior if that is useful.
- The difference term lets the policy react when endpoint nodes disagree.

Cons:
- Still depends only on already-compressed node actions.
- May throw away useful information from the GNN hidden state.

## 3. Learn Edge Actions From Node Embeddings

Use the GNN's hidden node embeddings instead of only the final node actions:

```text
edge_action(i, j) = MLP([h_i, h_j, edge_attr(i, j)])
```

For undirected edges:

```text
edge_action(i, j) = MLP([
    h_i + h_j,
    abs(h_i - h_j),
    edge_attr(i, j)
])
```

Pros:
- Usually the strongest general approach.
- Uses richer information before it has been compressed into node actions.
- Can include edge features such as length, stiffness, joint type, actuator limits, or contact state.
- Can preserve symmetry for undirected graphs.

Cons:
- Requires adding an edge-level action head to the actor.
- Slightly more architecture work than summing node actions.

## 4. Sum Plus Learned Correction

Use the simple sum as an inductive bias, then let the network learn a correction:

```text
edge_action(i, j) =
    node_action(i)
    + node_action(j)
    + MLP([h_i, h_j, edge_attr(i, j)])
```

Pros:
- Keeps the simple physical intuition.
- Gives the policy a way to fix cases where pure addition is wrong.
- Good compromise between a hand-coded baseline and a fully learned edge decoder.

Cons:
- More complex than the simple sum.
- The learned correction may dominate the sum if the baseline assumption is not useful.

## Recommendation

Start with the sum of endpoint node actions as a baseline because it is fast to test. If it performs poorly or seems too limiting, move to an edge action head based on node embeddings:

```text
edge_action(i, j) = MLP([
    h_i + h_j,
    abs(h_i - h_j),
    edge_attr(i, j)
])
```

This gives the policy more expressive power while still respecting the symmetry of undirected edges.

