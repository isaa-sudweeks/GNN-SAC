
> **Big Idea**:
> By using GNNs there is a way to formulate learned control algorithms to be topologically generalizable, thus allowing the network to potentially learn how to move many different structures using the same network. It might even be able to extend to the point where it could control robots to both roll and walk.

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
- `latest.pt` is always updated alongside numbered `step_<N>.pt` checkpoints. `checkpoint_keep_last` controls how many numbered checkpoints are retained.
- Resume a run by using the same training config and setting `resume_from_checkpoint=latest` with the same `work_dir`, or by passing an explicit checkpoint path:

```bash
python sac/gnn_train.py work_dir=/path/to/run resume_from_checkpoint=latest
python sac/gnn_train.py resume_from_checkpoint=/path/to/run/checkpoints/step_50000.pt
```
- When W&B is enabled, resumed training reuses the previous W&B run ID from `${work_dir}/wandb_run.json` or the checkpoint metadata, so resumed logs append to the original run instead of creating a new one.
- Set `set_wandb_offline=true` to force W&B offline logging. The supercomputer config enables this by default; local configs leave it false.
- For Slurm preemption/requeue runs, use the Submitit-backed supercomputer config. It uses a stable `work_dir`, `resume_from_checkpoint=latest`, and passes `--requeue` through Submitit:

```bash
python sac/gnn_train.py --config-name supercomputer --multirun
```

Set `GNN_SAC_RUN_ROOT` to put runs on shared persistent storage, and override cluster-specific values on the command line as needed:

```bash
GNN_SAC_RUN_ROOT=/scratch/$USER/gnn-sac-runs \
python sac/gnn_train.py --config-name supercomputer --multirun \
  hydra.launcher.partition=gpu hydra.launcher.account=my_account
```
- Resume starts a fresh environment episode at the restored global step. The replay buffer and optimizer state are restored; any episode that was in progress when the checkpoint was written is not continued because the MuJoCo environment state is not currently serialized.

# Current TODOs 
