# Graph Triangle Information

This note describes how triangle-level structure could be added to the graph
representation used by GNN-SAC.

The main idea is to give the GNN information about which physical nodes form
meaningful triangular units, instead of asking it to infer all face-level or
cell-level geometry from pairwise edges alone.

## Motivation

The current graph representation focuses on physical nodes and structural or
actuated edges. That is a good first representation, but it only exposes local
pairwise relationships directly. If the robot is generated from triangular
units, then a triangle may be a meaningful structural object in its own right.

Adding triangle information may help the policy learn:

- which groups of three physical nodes form a local structural unit
- local shape quality, such as whether a triangle is close to equilateral or
  distorted
- local deformation, area change, or collapse modes
- more useful topology-general features than raw node or edge IDs
- better message passing across nodes that participate in the same triangular
  face or cell

This could be especially useful during multi-topology training, where the goal
is not to memorize one robot layout, but to learn reusable control behavior from
graph structure.

Triangle information should not be encoded as a raw enumerated triangle ID in
the node features. Enumeration is useful as internal metadata, but a feature like
`triangle_id = 7` is topology-specific and may encourage memorization. The model
should receive geometric and incidence information instead.

## Useful Triangle Features

Triangle-level features could include:

```text
area
perimeter
isoperimetric_ratio
normal_x
normal_y
normal_z
rest_area
current_area
area_error
mean_edge_length
min_edge_length
max_edge_length
```

One useful shape-quality feature is the isoperimetric ratio:

```text
4 * sqrt(3) * area / perimeter^2
```

This value is `1` for an equilateral triangle and decreases as the triangle
becomes more slender or distorted.

## Option 1: Homogeneous Graph With Triangle Nodes

The simplest PyTorch Geometric implementation is to keep using a normal
homogeneous `Data` object, but add extra nodes representing triangles.

The graph then contains:

- physical nodes
- triangle nodes
- physical-physical structural edges
- physical-triangle incidence edges

The triangle node connects to each of its three physical vertices.

Example:

```python
import torch
from torch_geometric.data import Data


def build_graph_with_triangle_nodes(
    physical_x,
    structural_edges,
    triangles,
    triangle_features,
):
    """
    physical_x: [num_physical_nodes, physical_node_dim]
    structural_edges: list[(i, j)]
    triangles: list[(i, j, k)]
    triangle_features: [num_triangles, triangle_feature_dim]
    """

    num_physical = physical_x.shape[0]
    num_triangles = len(triangles)

    physical_dim = physical_x.shape[1]
    triangle_dim = triangle_features.shape[1]
    node_dim = physical_dim + triangle_dim + 2

    x = torch.zeros((num_physical + num_triangles, node_dim), dtype=torch.float32)

    # Physical nodes.
    x[:num_physical, :physical_dim] = physical_x
    x[:num_physical, physical_dim + triangle_dim + 0] = 1.0

    # Triangle nodes.
    tri_start = num_physical
    x[tri_start:, physical_dim:physical_dim + triangle_dim] = triangle_features
    x[tri_start:, physical_dim + triangle_dim + 1] = 1.0

    edges = []

    for i, j in structural_edges:
        edges.append((i, j))
        edges.append((j, i))

    for t_idx, (i, j, k) in enumerate(triangles):
        tri_node = tri_start + t_idx
        for vertex in (i, j, k):
            edges.append((vertex, tri_node))
            edges.append((tri_node, vertex))

    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()

    return Data(
        x=x,
        edge_index=edge_index,
        num_physical_nodes=num_physical,
        num_triangle_nodes=num_triangles,
    )
```

This allows message passing paths like:

```text
physical node -> triangle node -> neighboring physical node
```

The main advantage is that this fits the current `Data(x, edge_index, edge_attr)`
contract with less model-side complexity.

The main disadvantage is that physical nodes and triangle nodes must share one
feature tensor width. This usually requires padding and node-type indicators.

## Option 2: Heterogeneous Graph With Triangle Nodes

The conceptually cleaner option is to use PyG `HeteroData`. In this
representation, physical nodes and triangle nodes are different node types.

Example:

```python
import torch
from torch_geometric.data import HeteroData


data = HeteroData()

data["node"].x = physical_x
data["triangle"].x = triangle_x

data["node", "connected_to", "node"].edge_index = structural_edge_index
data["node", "belongs_to", "triangle"].edge_index = node_to_triangle_edge_index
data["triangle", "contains", "node"].edge_index = triangle_to_node_edge_index
```

This representation makes it explicit that physical nodes and triangle nodes are
different kinds of objects. It also allows them to have different feature
dimensions.

This option seems preferable for this project because it more directly matches
the meaning of the data:

- physical nodes are simulator state points
- triangle nodes are structural or geometric units
- physical-physical edges represent structural connectivity
- physical-triangle edges represent incidence

The tradeoff is that the model becomes more complex. The GNN would likely need
heterogeneous message passing layers such as `HeteroConv`, with separate message
functions for each relation type.

## Action Outputs With Triangle Nodes

Right now, the graph environment gets one action per physical node. If triangle
nodes are added, the actor should not output actuator commands for triangle
nodes.

Triangle nodes are information-carrying nodes. They are not controllable
physical bodies and should not receive direct actions.

For a homogeneous graph, the actor can run message passing over all nodes and
then select only the physical node embeddings before producing actions:

```python
node_embeddings = gnn(data.x, data.edge_index, data.edge_attr)
physical_embeddings = node_embeddings[:data.num_physical_nodes]
actions = actor_head(physical_embeddings)
```

For a heterogeneous graph, this is even cleaner because physical nodes have
their own node type:

```python
embeddings = hetero_gnn(data.x_dict, data.edge_index_dict)
physical_embeddings = embeddings["node"]
actions = actor_head(physical_embeddings)
```

So the action shape remains:

```text
[num_physical_nodes, node_action_dim]
```

not:

```text
[num_physical_nodes + num_triangle_nodes, node_action_dim]
```

Triangle nodes can influence the physical node embeddings through message
passing, but they do not expand the action space.

Later, if the project moves from node actions to the cleaner edge-action tendon
decoder described in the main graph representation document, the same principle
still applies. The GNN can use triangle nodes during message passing, but the
actor should only decode actions for actuated tendon edges.

## Recommendation

For the first working GNN-SAC milestone, triangle information is optional. The
core system should first prove that graph observations and node-level actions
work on one topology.

For multi-topology training, triangle information is more attractive. The
preferred long-term design is the heterogeneous `HeteroData` version, because it
keeps the semantics of physical nodes, triangle nodes, and relation types
explicit.

If implementation speed matters more than representational cleanliness, the
homogeneous `Data` version is a reasonable stepping stone.
