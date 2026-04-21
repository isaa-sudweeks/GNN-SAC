# Problem Definition

This document defines the first version of the control problem. Its job is to keep the implementation tied to a defensible thesis claim instead of drifting into a general reinforcement learning framework.

## Thesis Target

The initial target claim is:

A graph-structured SAC policy can learn tendon-level control for compliant or tensegrity robots in a way that generalizes across robot topologies better than fixed-size MLP policies.

The first defensible result should focus on cross-topology control for one behavior. Multi-behavior control, such as rolling and walking with the same network, should be treated as a later extension.

## Initial Scope

- Train a trusted fixed-topology SAC baseline with a standard MLP policy.
- Implement a minimal custom SAC trainer and validate it on the same fixed-topology baseline.
- Add graph observations and graph actor/critic models on one fixed topology.
- Add an edge-action tendon decoder so actions are emitted per actuated tendon edge.
- Train across multiple robot topologies and evaluate on held-out topologies.

## Robot Representation

### Nodes

Nodes should represent physical points or bodies in the robot graph.

Candidate node definitions:

- tensegrity rod endpoints
- rigid body centers
- mass points used by the simulator

Initial recommendation:

Use simulator state points that have meaningful positions and velocities, and keep this definition consistent across all topologies.

Open decisions:

- Exact node type
- Whether fixed anchors or environment contacts become nodes
- Whether node type should be encoded as a categorical feature

### Edges

Edges should represent physical relationships between nodes.

Candidate edge types:

- rigid structure edges
- passive tendon or cable edges
- actuated tendon edges
- contact or proximity edges

Initial recommendation:

Use physical robot connectivity as the graph edge set. Mark actuated tendons with an edge attribute instead of representing only actuator edges.

Open decisions:

- Whether graph edges are directed or stored as paired directed edges
- Whether non-physical proximity edges are useful
- Whether contact edges should be dynamic or excluded from the first version

### Actuated Edges

Actuated tendons are the action-bearing edges.

Each actuated edge should receive one action from the policy. The action can represent:

- target tendon length
- tendon rest-length change
- motor position command
- normalized force or tension command

Initial recommendation:

Use a normalized continuous action per actuated tendon, then map that action to the simulator command through an environment wrapper.

Open decisions:

- Exact actuator command meaning
- Action bounds
- Whether actions should be absolute targets or deltas

## Observation Definition

Each transition should expose enough information to satisfy the Markov assumption for the chosen simulator and task.

### Node Features

Likely node features:

- position
- linear velocity
- orientation, if the node represents a rigid body
- angular velocity, if the node represents a rigid body
- node type
- mass or other physical parameters, if topology generalization requires them

Open decisions:

- World-frame vs body-frame features
- Whether to subtract robot center of mass from positions
- Whether to include gravity-aligned orientation features

### Edge Features

Likely edge features:

- edge type
- rest length
- current length
- length error
- relative velocity along the edge
- stiffness or damping parameters
- actuated flag

Open decisions:

- Which physical parameters vary across topologies
- Whether actuator state should be stored as an edge feature
- Whether edge features should include both static and dynamic fields

### Global Features

Optional graph-level features:

- command velocity
- target direction
- task phase
- terrain parameters

Initial recommendation:

Avoid global features unless the task needs them. Start with a single behavior and a fixed reward definition.

## Action Definition

The GNN actor should output one action distribution per actuated tendon edge.

For each actuated edge:

- The actor produces a Gaussian mean and log standard deviation.
- The sampled action is squashed and scaled to the actuator bounds.
- The log probability is tracked for SAC entropy terms.

Important design issue:

Robots with more actuated tendons produce more per-edge log probabilities. The SAC objective should define whether entropy is summed, averaged, or normalized by the number of actuated edges.

Initial recommendation:

Average log probabilities across actuated edges for temperature tuning and policy loss unless experiments show that summed entropy is more appropriate.

## Reward Definition

The first reward should measure one behavior clearly.

Candidate first task:

Forward locomotion over a fixed time horizon.

Likely reward terms:

- positive forward center-of-mass velocity
- control effort penalty
- stability or survival bonus
- orientation penalty, if needed
- constraint violation penalty, if needed

Open decisions:

- Exact forward axis
- Episode length
- Whether falling terminates the episode
- Whether reward is normalized across different robot sizes

## Evaluation Metrics

Primary metrics:

- final episode return
- success rate, if the task has a binary success condition
- sample efficiency
- held-out topology performance

Secondary metrics:

- robustness to node ordering
- robustness to edge ordering
- variance across random seeds
- action smoothness or actuator effort

## Baselines

Required baselines:

- SB3 or SBL SAC with an MLP policy on a fixed topology
- custom SAC with an MLP policy on the same fixed topology
- GNN SAC on the same fixed topology

Generalization baselines:

- train separate MLP policies per topology, if feasible
- train a padded or flattened MLP policy across multiple topologies, if a fair representation is possible

## Definition Of Done For First Milestone

- The custom SAC implementation matches the trusted baseline closely enough on one fixed-topology vector-observation task.
- The graph representation can encode one fixed topology without losing required simulator state.
- The GNN actor outputs valid bounded actions for each actuated tendon edge.
- The GNN critic outputs one scalar Q value for a graph-action pair.
