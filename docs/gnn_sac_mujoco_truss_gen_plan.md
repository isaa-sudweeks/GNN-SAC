# Plan: Get GNN-SAC Rolling With `mujoco_truss_gen`

## Summary

Implement the simplest working graph-control path: keep the GNN actor, critic, and replay buffer node-action based, and make the graph environment translate node actions into actuator controls by summing endpoint node actions for each actuated tendon. Use `mujoco_truss_gen` to generate/load the octahedron instead of relying on the local static XML path.

## Key Changes

- Replace the graph env's local truss base with `mujoco_truss_gen`.
- Fix Gymnasium graph observation validation by returning raw dict observations from the Gym env.
- Convert graph dict observations into PyG `Data` objects in the project wrapper layer.
- Keep node actions and translate them inside the graph env with endpoint summation.
- Separate flat SAC and graph SAC configs with a dedicated `config/gnn_config.yaml`.

## Implementation Notes

- Observation dict contract:
  - `x`: float32, shape `[num_nodes, node_feature_dim]`.
  - `edge_index`: int64, shape `[2, num_directed_structural_edges]`.
- Action mapping contract:
  - Build a stable `node_name -> node_index` map from `mj_model.node_names`.
  - Build a stable `tendon_id -> (node_a, node_b)` map from tendon names beginning with `tendon_`.
  - For each actuator id, read its tendon id from `model.actuator_trnid[actuator_id, 0]`.
  - Output zero for any actuator that is not backed by a two-node structural tendon.
  - Clip summed endpoint node actions to `[-1, 1]` before sending them through generated-env control scaling.
- Wrapper behavior:
  - Graph observations become PyG `Data`.
  - Normal non-graph observations keep existing tensor conversion behavior.

## Test Plan

- Verify `make_env()` with `gnn_config` resets to a PyG `Data` observation.
- Verify random node actions step the generated octahedron graph env.
- Verify passive structural edges do not change the MuJoCo actuator vector size.
- Verify `GNNSAC.act()`, `GNNBuffer.add()`, `GNNBuffer.sample()`, and one `agent.update()` complete on CPU.
- Run a short graph-training smoke command with W&B disabled.

## Assumptions

- "Mujoco TrustGen" means the installed `mujoco_truss_gen` package.
- The file referred to as `write.py` is `env/mujoco_gen/octehedron_graph_env_right.py`.
- v1 intentionally keeps node actions, even though edge-action decoding is the cleaner long-term architecture.
- The goal for this pass is working graph GNN-SAC on one generated octahedron, not cross-topology generalization yet.
