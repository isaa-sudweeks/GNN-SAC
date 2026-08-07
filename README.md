
> **Big Idea**:
> By using GNNs there is a way to formulate learned control algorithms to be topologically generalizable, thus allowing the network to potentially learn how to move many different structures using the same network. It might even be able to extend to the point where it could control robots to both roll and walk.

# Setup with uv

The checked-in `uv.lock` provides a reproducible environment. On Linux, uv
installs the CUDA 13.0 (`cu130`) PyTorch build; on macOS it uses PyTorch's
native wheel instead.

```bash
uv sync --frozen
uv run python sac/gnn_train.py device=cpu steps=1000 enable_wandb=false
```

On a CUDA training machine, omit `device=cpu` (or set `device=cuda`). Commands
run through `uv run` automatically use the locked project environment.

# Thesis Claim
- A graph-structured SAC policy can learn tendon-level control for compliant or tensegrity robots in a way that generalizes across robot topologies better than fixed-size MLP policies.
- The important property is not that the controller is fully topologically invariant. It should be permutation equivariant to node and edge ordering while still using the robot topology to produce useful actions.
- The first defensible claim should probably be cross-topology control for one behavior. Controlling both rolling and walking with one network is interesting, but that should be treated as a later extension unless the first result is already working.

# Things to consider 
- GNN primarily output node embeddings making it a little harder to actuate tendons. There are three solutions that I can really think of to solve this problem: 
	- Reformulate the connections so that edges are nodes and nodes are edges or
	- Use the inverse kinematics to turn X,Y,Z actuation of nodes into tendon actions. The problem with this one is the model might output two node actions that conflict, so then how do we pick which one to do? This also makes the action meaning indirect, which could make credit assignment harder even if the environment technically remains Markov when the projection is deterministic.
	- Create an edge-action MLP that takes in the node pair features and edge features, then calculates tendon actions, like they do in the tensegrity paper. This honestly seems like the best right now, and I think this is the path I should follow.
- The best initial architecture is probably:
	- GNN encodes node and edge state.
	- For each actuated tendon edge, an MLP receives the two endpoint embeddings plus edge attributes.
	- The MLP outputs that tendon's action distribution.
	- The critic consumes the graph state plus edge actions and pools to one graph-level Q value.
- The replay buffer needs to contain the graphs that the policies are operating on, not just flat input data. This can likely be handled with PyTorch Geometric style batching, but it needs to be designed intentionally.
- SAC with variable-size graphs has a few extra issues:
	- Observations and actions may have different sizes across topologies.
	- The actor should output one action distribution per controllable tendon edge.
	- The critic needs to reduce a graph plus all edge actions into one scalar Q value.
	- Entropy and log probabilities may need to be normalized carefully so robots with more tendons do not automatically get different SAC temperature behavior.

# Rewrite Decision
- I do not think SB3/SBL will let me change only a few pieces and still support graph observations, edge-level tendon actions, and variable topology batching.
- A custom SAC implementation is probably justified, but it should be scoped as a minimal graph SAC implementation for this project rather than a general RL framework.
- The custom implementation should still be validated against SB3/SBL SAC on a fixed-topology baseline before I trust the graph results.

# Steps to Take: 
- [ ] Lock down the problem definition:
	- Define what counts as a node.
	- Define what counts as an edge.
	- Define which edges are actuated tendons.
	- Define node features, edge features, actions, rewards, and the first task.
- [x] Get a trusted fixed-topology baseline running with SB3/SBL SAC and a normal MLP policy.
- [] Implement a minimal SAC trainer in PyTorch:
	- Replay buffer.
	- Actor and critic losses.
	- Target networks.
	- Entropy temperature tuning.
	- Training and evaluation loop.
- [ ] Validate the custom SAC implementation against SB3/SBL SAC on one fixed MuJoCo robot with vector observations and vector actions.
- [ ] Add a GNN actor and critic while still using one fixed topology.
- [ ] Add the edge-action tendon decoder so the actor outputs one action distribution per actuated tendon.
- [ ] Update the replay buffer and batching code so it can store graph transitions:
	- Graph state.
	- Edge actions.
	- Reward.
	- Done flag.
	- Next graph state.
- [ ] Train the GNN SAC policy on one topology and compare it against the MLP SAC baseline.
- [ ] Train across multiple robot configurations and test on held-out configurations.
- [ ] Measure the actual thesis claim:
	- Final reward or success rate.
	- Sample efficiency.
	- Generalization to unseen topologies.
	- Robustness to node and edge ordering.
- [ ] If cross-topology control works, consider extending to multiple behaviors such as rolling and walking.
- [ ] If the results are strong, start writing the paper around the graph representation, edge-action decoder, and topology generalization experiments.

# Interesting Potential Avenues of advancement 
- I could replace the MSE style regression with a discrete regression over bins similar to TD-MPC2 which might add more stability to training and a better reward signal.
- I could do some sort of scaled entropy stuff similar to what is done in TD-MPC2. One of the reasons that this would be needed I think is that as the number of tendons changes so will the entropy. This could lead to some weirdness in training where the agent prefers high entropy in robots with many tendons and low entropy in robots with few tendons. This could be avoided by scaling the entropy.

# Resuming Preempted Training
- Training writes resumable checkpoints to `${work_dir}/checkpoints` every `checkpoint_freq` environment steps. Each checkpoint contains the agent weights, optimizer states, replay buffer, trainer counters, logger state, RNG state, and resolved config metadata.
- `latest.pt` is always updated alongside numbered `step_<N>.pt` checkpoints. Lightweight `*.agent.pt` companions contain only inference weights, and `checkpoint_keep_last` prunes both forms together.
- Resume a run by using the same training config and setting `resume_from_checkpoint=latest` with the same `work_dir`, or by passing an explicit checkpoint path:

```bash
python sac/gnn_train.py work_dir=/path/to/run resume_from_checkpoint=latest
python sac/gnn_train.py resume_from_checkpoint=/path/to/run/checkpoints/step_50000.pt
```
- When W&B is enabled, resumed training reuses the previous W&B run ID from `${work_dir}/wandb_run.json` or the checkpoint metadata, so resumed logs append to the original run instead of creating a new one.
- Set `set_wandb_offline=true` to force W&B offline logging. The `platform=supercomputer` profile enables this by default; local configs leave it false.
- For Slurm preemption/requeue runs, use the Submitit-backed supercomputer platform profile. It uses a stable `work_dir`, `resume_from_checkpoint=latest`, and passes `--requeue` through Submitit:

```bash
python sac/train.py platform=supercomputer sac_backend=gnn sim_backend=mjx --multirun
```

Each multirun job is isolated under
`${run_root}/${task}/${exp_name}/seed_${seed}/job_<number>_<override-hash>`.
The hash is derived from the Hydra job override set, while the job number
also separates duplicate configurations within one sweep. Requeued jobs retain
the same directory and therefore resume only their own checkpoint and W&B run.

Set `GNN_SAC_RUN_ROOT` to put runs on shared persistent storage, and override cluster-specific values on the command line as needed:

```bash
GNN_SAC_RUN_ROOT=/scratch/$USER/gnn-sac-runs \
python sac/train.py platform=supercomputer sac_backend=gnn sim_backend=mjx --multirun \
  hydra.launcher.partition=gpu hydra.launcher.account=my_account
```
- Resume starts a fresh environment episode at the restored global step. The replay buffer and optimizer state are restored; any episode that was in progress when the checkpoint was written is not continued because the MuJoCo environment state is not currently serialized.

# SAC Backend Profiles

`config/config.yaml` now composes a `sac_backend` profile. The default is
`mlp`; use `sac_backend=gnn` to switch the shared training config to graph
observations, GNN dimensions, and the GNN SAC agent/buffer:

```bash
python sac/train.py sac_backend=mlp steps=10000
python sac/train.py sac_backend=gnn steps=10000
```

Legacy wrapper configs live under `config/archieved/`; new runs should prefer
the explicit config groups.

Execution platform and simulator backend are separate config groups:

```bash
python sac/train.py sac_backend=gnn sim_backend=mujoco steps=10000
python sac/train.py sac_backend=gnn sim_backend=mjx 'topologies=[octahedron,tetrahedron]'
python sac/train.py platform=supercomputer sac_backend=gnn sim_backend=mjx --multirun
```

`config/archieved/supercomputer.yaml` remains as a legacy wrapper for
`platform=supercomputer sac_backend=gnn sim_backend=mjx`.

# Unified Truss Topology Environments
- `truss-graph` is the reusable graph-observation environment for `mujoco_truss_gen` presets. It emits PyTorch Geometric graph observations through the wrapper layer and maps one scalar node action per graph node to tendon actuator commands.
- `truss-mlp` is the reusable flat observation/action environment for standard MLP policies. It uses the same generated topology source, but keeps fixed-size vector observations and actions.
- Select one generated topology with `truss_topology`. Valid names come from `mujoco_truss_gen.PRESETS`; these include the canonical `octahedron`, `tetrahedron`, `icosahedron`, and `solar_array` models plus the enumerated Henneberg and Usevitch families.

For fixed-topology graph training on the cluster, select the supercomputer
platform, GNN SAC backend, and batch-native MJX simulator backend:

```bash
python sac/train.py platform=supercomputer sac_backend=gnn sim_backend=mjx \
  steps=10000 num_envs=256
```

MJX training uses a separate native MuJoCo evaluation environment by default
(`eval_backend=mujoco`). This avoids inefficient single-environment MJX
evaluation and permits evaluation video capture with `save_video=true` while
the training environments remain accelerator-native.

Learner work scales through `replay_ratio`, defined as replay samples consumed
per newly collected transition. With the default `replay_ratio=1`, a vector
step collecting 2,048 transitions and `batch_size=256` schedules eight optimizer
updates. Transitions enter replay after every vector step. By default the learner
spends its accumulated update budget every eight vector steps; set
`update_every_vector_steps=1` to update after every vector step. Pending learner
work carries across vector steps and checkpoints.
Set the deprecated `iterations` option only to reproduce the legacy schedule of
one full optimizer update per collected transition.

For multiple topologies, `num_envs` is the total vector-environment count and
must divide evenly across the requested robot configurations. For example, this
creates 500 octahedron environments and 500 tetrahedron environments while
retaining a single mixed-graph policy inference call:

```bash
python sac/train.py platform=supercomputer sac_backend=gnn sim_backend=mjx \
  num_envs=1000 'topologies=[octahedron,tetrahedron]'
```

Each topology has a separately compiled MJX step function and fixed-size state
batch. The trainer derives its total environment count as
`num_envs`, with `num_envs / len(truss_topologies)` slots per topology.

Run checkpoint evaluation in vectorized waves with the matching inference
profile:

```bash
python sac/gnn_infer.py --config-name inference/gnn_mjx \
  model=/path/to/final.pt episodes=256 num_envs=256
```

The MJX training path requires `mujoco-truss-gen>=0.12.0` and
training-environment rendering disabled. Native MuJoCo evaluation can render
and record videos. MJX owns one compiled model and one fixed environment batch
per topology, so realistic models and fixed-shape runtime domain randomization
are supported, but model-changing randomization is not. Use
`mujoco_backend=mujoco` for training-time rendering, length-scale
randomization, or physical-parameter randomization that rebuilds the generated
model.

JAX remains the default MJX physics implementation. On an NVIDIA CUDA host,
install the Warp extra and select the upstream Warp implementation explicitly:

```bash
python -m pip install 'mujoco-truss-gen[warp]>=0.12.0'
python sac/gnn_train.py sim_backend=mjx mjx_impl=warp device=cuda
```

`warp_graph_mode` defaults to the benchmark-leading `warp_staged` and also
accepts `warp` or `warp_staged_ex`.
`warp_naconmax` and `warp_njmax` optionally set fixed contact and constraint
capacities. Warp does not replace the `sim_backend=mjx` pipeline; it changes the
physics implementation used inside that batch-native environment.

```yaml
task: truss-graph
truss_topology: octahedron
```

```yaml
task: truss-mlp
truss_topology: solar_array
```

- Train or evaluate a graph policy across several topologies by listing `truss_topologies`. The shorter CLI alias `topologies` is also accepted. The environment factory expands this into topology-specific tasks such as `truss-graph:octahedron`.

```yaml
task: truss-graph
truss_topologies:
  - octahedron
  - tetrahedron
```

- Add `:realistic` to one topology entry to use the realistic generated variant for only that entry.

```yaml
task: truss-graph
truss_topologies:
  - octahedron
  - octahedron:realistic
  - solar_array
```

When overriding a topology list from zsh, quote the Hydra list so the shell does not treat square brackets as a filename pattern. Omitting spaces inside the list is the least fragile form:

```bash
uv run python sac/gnn_train.py exp_name=env_test_multi_task 'truss_topologies=[octahedron,octahedron:realistic,solar_array]'
```

- Use `eval_task` for a different evaluation topology. The `task:topology` form sets the base environment task and the generated topology in one string.

```yaml
task: truss-graph
truss_topologies:
  - octahedron
  - tetrahedron
eval_task: truss-graph:icosahedron
```

- For flat MLP baselines, different topologies are only valid together when their flat observation and action spaces match. Mismatched MLP topology lists fail early instead of padding or masking.

```yaml
task: truss-mlp
truss_topology: octahedron
eval_task: truss-mlp:tetrahedron
```

- `truss_realistic` requests realistic generated models. `truss_graph_view: auto` uses physical graph nodes by default and logical graph nodes for realistic models; set it explicitly to `physical` or `logical` only when needed.
- Generated model physical values live in `config/physics/physical_parameters.yaml` under `physical_parameters` and apply to both training and inference. Set `physical_parameters_enabled: false` to skip this config and use the `mujoco-truss-gen` package defaults. Domain randomization lives in `config/physics/domain_randomization.yaml`; use `domain_randomization` as the master switch. MJX and native MuJoCo support fixed-shape runtime ranges for body mass/inertia, DOF damping/armature/friction loss, actuator gain/bias/dynamics, all three geom-friction axes, tendon stiffness/damping/armature/friction loss, and vertical gravity. Native MuJoCo also supports model-level `length_scale` and `physical_parameters` randomization.

```yaml
physical_parameters_enabled: true
physical_parameters:
  node_radius: 0.1

domain_randomization: true
domain_randomization_params:
  length_scale:
    enabled: false
  body_mass_multiplier:
    enabled: true
    min: 0.8
    max: 1.2
  # Native MuJoCo/model-level randomization only; set true only for mujoco_backend=mujoco.
  physical_parameters:
    node_radius:
      enabled: false
      min: 0.08
      max: 0.12
```

# Current TODOs 
