# Padded MLP baseline

`sac_backend=padded_mlp` is the shared fixed-width baseline for comparing a
plain MLP with the topology-general GNN. It deliberately uses `truss-graph`,
`use_control_graph=true`, and `GNNBuffer`, so both methods receive the same six
state values per control node, node-command mapping, topology-balanced replay,
reward, termination, and evaluation path. PyG objects are only a variable-size
transport; the MLP never reads `edge_index`.

The selected 13-topology distribution contains at most 21 control nodes and 17
active node actions. This differs from the maximum of nine abstract robot nodes.
The MLP input is therefore fixed at `21 * (6 + 1 + 1) + 1 = 169` values:
flattened node state, existence mask, action mask, and normalized rigidity. Its
21 output slots are masked down to the active nodes before simulation, replay
criticism, and entropy calculation.

Validate the upstream topology contract after changing `mujoco-truss-gen`:

```bash
python scripts/validate_padded_mlp_topologies.py
```

Run a short native two-topology smoke test:

```bash
python sac/train.py sac_backend=padded_mlp device=cpu enable_wandb=false \
  save_csv=false save_agent=false checkpoint_freq=0 eval_at_end=false \
  domain_randomization=false 'truss_topologies=[tetrahedron,octahedron]' \
  max_steps=2 nsubsteps=1 steps=1000 batch_size=256 buffer_size=1024
```

For paired experiments, explicitly set the same `pcgrad` value for both the GNN
and padded MLP launch commands. The padded MLP supports task-aware PCGrad, but
defaults to `false` so enabling it is an intentional experimental decision.

The default `[736, 736]` MLP has 2,074,828 trainable actor/critic parameters,
within 0.2% of the current production GNN's 2,078,008. Recheck both counts after
changing either architecture.
