from time import time 

import numpy as np 
import re
import torch 
from tensordict.tensordict import TensorDict 
from trainer.base import Trainer 

try:
    from torch_geometric.data import Data
except ImportError:
    Data = ()

REWARD_INFO_EXCLUDE_KEYS = {"success", "terminated", "truncated"}


class OnlineTrainer(Trainer):
    """
    Trainer class for single-task online SAC training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._step = 0 
        self._ep_idx = 0 
        self._start_time = time() 
        self._episode_reward_components = {}
        
        self.eval_env = self.env
        eval_task = getattr(self.cfg, "eval_task", None)
        has_domain_randomization = getattr(self.cfg, "domain_randomization", False)
        topology_bucket_metadata = self._topology_bucket_metadata(self.env)
        
        if (
            (eval_task is not None and eval_task != self.cfg.task)
            or has_domain_randomization
            or topology_bucket_metadata is not None
        ):
            from copy import deepcopy
            from env import make_env
            eval_cfg = deepcopy(self.cfg)
            eval_cfg.domain_randomization = False
            if topology_bucket_metadata is not None:
                # Evaluation needs one isolated slot per topology. Reusing the
                # training buckets would overwrite live rollout state.
                eval_cfg.num_envs = 1
            if eval_task is not None:
                eval_cfg.task = eval_task
                eval_cfg.env_name = eval_task
                if hasattr(eval_cfg, "tasks"):
                    eval_cfg.tasks = [eval_task]
            self.eval_env = make_env(eval_cfg)
            
        self.maybe_load_checkpoint()

    def common_metrics(self):
        """
        Return a dictionary of current metrics.
        """
        elapsed_time = time() - self._start_time 
        return dict(
            step= self._step,
            episode= self._ep_idx,
            elapsed_time= elapsed_time,
            steps_per_sec= self._step / elapsed_time if elapsed_time > 0 else 0,
        )

    def _extract_reward_components(self, info):
        """
        Return numeric reward component values from an env info dict.
        """
        components = {}
        for key, value in info.items():
            if key in REWARD_INFO_EXCLUDE_KEYS:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().cpu().item()
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                components[key] = value
        return components

    def _accumulate_reward_components(self, info):
        for key, value in self._extract_reward_components(info).items():
            current_value = self._episode_reward_components.get(key, 0.0)
            self._episode_reward_components[key] = current_value + value

    @staticmethod
    def _crossed_eval_interval(previous_step, current_step, eval_freq):
        return previous_step // eval_freq < current_step // eval_freq
    
    def eval(self):
        """
        Evaluate a SAC agent.
        """
        if self._topology_bucket_metadata(self.eval_env) is not None:
            return self._eval_topology_buckets()
        if bool(getattr(self.cfg, "multitask", False)) and int(getattr(self.eval_env, "num_envs", 1)) > 1:
            return self._eval_multitask()
        return self._eval_one()

    def _eval_one(self, task_idx=None, video_key="videos/eval_video"):
        ep_rewards, ep_successes, ep_lengths = [], [], []
        for i in range(self.cfg.eval_episodes):
            obs = self.eval_env.reset(task_idx=task_idx) if task_idx is not None else self.eval_env.reset()
            done, ep_reward, t = False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.eval_env, enabled=(i==0))
            while not done:
                #if getattr(self.cfg, 'device', 'cuda') == 'cuda':
                    #torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(obs, t0=t==0, eval_mode=True)
                obs, reward, done, info = self.eval_env.step(action)
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.eval_env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            ep_lengths.append(t)
            if self.cfg.save_video:
                self.logger.video.save(self._step, key=video_key)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_length=np.nanmean(ep_lengths),
        )

    def _eval_multitask(self):
        metrics = {}
        task_rewards, task_successes, task_lengths = [], [], []
        for task_idx in range(int(getattr(self.eval_env, "num_envs", 1))):
            task_name = self._eval_task_name(task_idx)
            task_key = self._metric_key(task_name)
            task_metrics = self._eval_one(
                task_idx=task_idx,
                video_key=f"videos/eval_video/{task_key}",
            )
            metrics[f"{task_key}_episode_reward"] = task_metrics["episode_reward"]
            metrics[f"{task_key}_episode_success"] = task_metrics["episode_success"]
            metrics[f"{task_key}_episode_length"] = task_metrics["episode_length"]
            task_rewards.append(task_metrics["episode_reward"])
            task_successes.append(task_metrics["episode_success"])
            task_lengths.append(task_metrics["episode_length"])
        metrics.update(
            episode_reward=np.nanmean(task_rewards),
            episode_success=np.nanmean(task_successes),
            episode_length=np.nanmean(task_lengths),
        )
        return metrics

    def _eval_topology_buckets(self):
        topologies, representative_indices = self._topology_bucket_metadata(self.eval_env)
        metrics = {}
        topology_rewards, topology_successes, topology_lengths = [], [], []
        for topology in topologies:
            env_idx = representative_indices[topology]
            topology_key = self._metric_key(topology)
            topology_metrics = self._eval_one(
                task_idx=env_idx,
                video_key=f"videos/eval_video/{topology_key}",
            )
            metrics[f"{topology_key}_episode_reward"] = topology_metrics["episode_reward"]
            metrics[f"{topology_key}_episode_success"] = topology_metrics["episode_success"]
            metrics[f"{topology_key}_episode_length"] = topology_metrics["episode_length"]
            topology_rewards.append(topology_metrics["episode_reward"])
            topology_successes.append(topology_metrics["episode_success"])
            topology_lengths.append(topology_metrics["episode_length"])
        metrics.update(
            episode_reward=np.nanmean(topology_rewards),
            episode_success=np.nanmean(topology_successes),
            episode_length=np.nanmean(topology_lengths),
        )
        return metrics

    @staticmethod
    def _topology_bucket_metadata(env):
        current = env
        while current is not None:
            namespace = getattr(current, "__dict__", {})
            is_topology_bucket = namespace.get(
                "is_topology_bucket",
                getattr(type(current), "is_topology_bucket", False),
            )
            if bool(is_topology_bucket):
                return (
                    list(current.topologies),
                    dict(current.topology_representative_indices),
                )
            current = getattr(current, "env", None)
        return None

    def _eval_task_name(self, task_idx):
        tasks = list(getattr(self.cfg, "tasks", []))
        if task_idx < len(tasks):
            return str(tasks[task_idx])
        return f"env_{task_idx}"

    def _metric_key(self, value):
        return re.sub(r"[^0-9a-zA-Z_.-]+", "_", str(value)).strip("_") or "env"

    def _cfg_get(self, obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _action_noise_cfg(self):
        params = self._cfg_get(self.cfg, "domain_randomization_params", {})
        return self._cfg_get(params, "action_noise", {})

    def _should_apply_action_noise(self, seed_action=False):
        if not bool(self._cfg_get(self.cfg, "domain_randomization", False)):
            return False
        noise_cfg = self._action_noise_cfg()
        if not bool(self._cfg_get(noise_cfg, "enabled", False)):
            return False
        if seed_action and not bool(self._cfg_get(noise_cfg, "apply_to_seed_actions", False)):
            return False
        return float(self._cfg_get(noise_cfg, "std", 0.0)) > 0.0

    def _apply_action_noise(self, action, seed_action=False):
        if not self._should_apply_action_noise(seed_action=seed_action):
            return action
        noise_cfg = self._action_noise_cfg()
        std = float(self._cfg_get(noise_cfg, "std", 0.0))
        low = float(self._cfg_get(noise_cfg, "clip_low", self._cfg_get(self.cfg, "action_low", -1.0)))
        high = float(self._cfg_get(noise_cfg, "clip_high", self._cfg_get(self.cfg, "action_high", 1.0)))
        if isinstance(action, torch.Tensor):
            return torch.clamp(action + torch.randn_like(action) * std, low, high)
        return np.clip(action + np.random.normal(0.0, std, size=np.asarray(action).shape), low, high).astype(
            np.float32,
            copy=False,
        )

    def _select_multi_env_actions(self, observations, episode_tds, env_indices):
        """Select actions for one vector step, batching compatible policy calls."""
        using_seed_actions = self._step <= self.cfg.seed_steps
        if using_seed_actions:
            actions = [self.env.rand_act(env_idx=env_idx) for env_idx in env_indices]
        elif hasattr(self.agent, "act_batch"):
            actions = self.agent.act_batch([observations[env_idx] for env_idx in env_indices])
        else:
            actions = [
                self.agent.act(observations[env_idx], t0=len(episode_tds[env_idx]) == 1)
                for env_idx in env_indices
            ]
        return [
            self._apply_action_noise(action, seed_action=using_seed_actions)
            for action in actions
        ]
    
    def to_td(self, obs, action=None, reward=None, terminated=None):
        """
        Creates a TensorDict for a new episode.
        """
        is_graph_obs = isinstance(obs, Data)
        if is_graph_obs:
            obs = obs.cpu()
        elif isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device='cpu').unsqueeze(0)
        else:
            obs = obs.unsqueeze(0).cpu()
        if action is None:
            action = torch.full_like(self.env.rand_act(), float('nan'))
        if reward is None:
            reward = torch.tensor(float('nan'))
        if terminated is None:
            terminated = torch.tensor(float('nan'))
        fields = {
            "obs": obs,
            "action": action.unsqueeze(0),
            "reward": reward.unsqueeze(0),
            "terminated": terminated.unsqueeze(0),
        }
        if is_graph_obs:
            td = fields
        else:
            td = TensorDict(fields, batch_size=(1,))
        return td

    def train(self):
        """
        Train the SAC agent.
        """
        num_envs = int(getattr(self.env, "num_envs", getattr(self.cfg, "num_envs", 1)))
        if num_envs > 1:
            return self._train_multi_env(num_envs)

        train_metrics, done, eval_next = {}, True, False 
        pretrain_steps = int(getattr(self.cfg, 'pretrain_steps', min(self.cfg.seed_steps, 1000)))
        while self._step <= self.cfg.steps:
            # Evaluate agent periodically 
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True 
            
            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    eval_metrics.update(self.common_metrics())
                    self.logger.log(eval_metrics, 'eval')
                    self.report_eval_metrics(eval_metrics, self._step)
                    eval_next = False
                if self._step > 0:
                    train_metrics.update(
                        episode_reward=torch.stack([td["reward"].view(()) for td in self._tds[1:]]).sum(),
                        episode_success=info["success"],
                        episode_length=len(self._tds) - 1,
                        episode_terminated=info["terminated"],
                        episode_truncated=info["truncated"],
                    )
                    train_metrics.update(self.common_metrics())
                    self.logger.log(train_metrics, 'train')
                    if self._episode_reward_components:
                        reward_metrics = dict(
                            step=self._step,
                            episode=self._ep_idx,
                            episode_length=len(self._tds) - 1,
                        )
                        reward_metrics.update(self._episode_reward_components)
                        self.logger.log(reward_metrics, 'training_rewards')
                    episode_td = self._tds if isinstance(self._tds[0]["obs"], Data) else torch.cat(self._tds)
                    self._ep_idx = self.buffer.add(episode_td)

                obs = self.env.reset()
                self._tds = [self.to_td(obs)]
                self._episode_reward_components = {}
            
            # Collect experience 
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(self._tds)==1)
            else:
                action = self.env.rand_act()
            action = self._apply_action_noise(action, seed_action=self._step <= self.cfg.seed_steps)
            obs, reward, done, info = self.env.step(action)
            self._accumulate_reward_components(info)
            terminated = info['terminated'] if self.cfg.episodic else torch.tensor(0.0)
            self._tds.append(self.to_td(obs, action, reward, terminated))

            # Update agent 
            if self._step >= self.cfg.seed_steps and self.buffer.size >= self.cfg.batch_size:
                if self._step == self.cfg.seed_steps:
                    num_updates = pretrain_steps
                    print(f'Pretraining agent on seed data for {num_updates} updates...')
                else:
                    num_updates = self.cfg.iterations
                if num_updates > 0:
                    for _ in range(num_updates):
                        _train_metrics = self.agent.update(self.buffer)
                    train_metrics.update(_train_metrics)

            previous_step = self._step
            self._step += 1
            self.maybe_save_checkpoint(previous_step)
        self.maybe_save_checkpoint(force=True)
        self.logger.finish(self.agent)
        if self.eval_env is not self.env:
            self.eval_env.close()
        return self._best_eval_metrics

    def _train_multi_env(self, num_envs):
        """
        Train from multiple independent copies of the same task.
        """
        train_metrics, eval_next = {}, False
        done = [True] * num_envs
        infos = [None] * num_envs
        observations = [None] * num_envs
        episode_tds = [None] * num_envs
        reward_components = [None] * num_envs
        pretrain_steps = int(getattr(self.cfg, 'pretrain_steps', min(self.cfg.seed_steps, 1000)))

        while self._step <= self.cfg.steps:
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            done_indices = [env_idx for env_idx, is_done in enumerate(done) if is_done]
            if done_indices:
                previous_components = self._episode_reward_components
                for env_idx in done_indices:
                    if eval_next:
                        if (
                            self._topology_bucket_metadata(self.eval_env) is None
                            and hasattr(self.eval_env, "set_active_env")
                        ):
                            self.eval_env.set_active_env(env_idx)
                        eval_metrics = self.eval()
                        eval_metrics.update(self.common_metrics())
                        self.logger.log(eval_metrics, 'eval')
                        self.report_eval_metrics(eval_metrics, self._step)
                        eval_next = False

                    if episode_tds[env_idx] is not None and len(episode_tds[env_idx]) > 1:
                        info = infos[env_idx]
                        train_metrics.update(
                            episode_reward=torch.stack([td["reward"].view(()) for td in episode_tds[env_idx][1:]]).sum(),
                            episode_success=info["success"],
                            episode_length=len(episode_tds[env_idx]) - 1,
                            episode_terminated=info["terminated"],
                            episode_truncated=info["truncated"],
                            episode_env_idx=env_idx,
                        )
                        train_metrics.update(self.common_metrics())
                        self.logger.log(train_metrics, 'train')
                        if reward_components[env_idx]:
                            reward_metrics = dict(
                                step=self._step,
                                episode=self._ep_idx,
                                episode_length=len(episode_tds[env_idx]) - 1,
                                episode_env_idx=env_idx,
                            )
                            reward_metrics.update(reward_components[env_idx])
                            self.logger.log(reward_metrics, 'training_rewards')
                        episode_td = (
                            episode_tds[env_idx]
                            if isinstance(episode_tds[env_idx][0]["obs"], Data)
                            else torch.cat(episode_tds[env_idx])
                        )
                        self._ep_idx = self.buffer.add(episode_td)
                self._episode_reward_components = previous_components

                reset_obs = self.env.reset_many(env_indices=done_indices)
                for env_idx, obs in zip(done_indices, reset_obs):
                    observations[env_idx] = obs
                    episode_tds[env_idx] = [self.to_td(obs)]
                    reward_components[env_idx] = {}
                    done[env_idx] = False

            remaining_steps = self.cfg.steps - self._step + 1
            env_indices = list(range(min(num_envs, remaining_steps)))
            if not env_indices:
                break

            actions = self._select_multi_env_actions(observations, episode_tds, env_indices)

            results = self.env.step_many(actions, env_indices=env_indices)
            for env_idx, action, (obs, reward, is_done, info) in zip(env_indices, actions, results):
                observations[env_idx] = obs
                done[env_idx] = is_done
                infos[env_idx] = info

                previous_components = self._episode_reward_components
                self._episode_reward_components = reward_components[env_idx]
                self._accumulate_reward_components(info)
                reward_components[env_idx] = self._episode_reward_components
                self._episode_reward_components = previous_components

                terminated = info['terminated'] if self.cfg.episodic else torch.tensor(0.0)
                episode_tds[env_idx].append(self.to_td(obs, action, reward, terminated))

            previous_step = self._step
            self._step += len(env_indices)
            if self._crossed_eval_interval(previous_step, self._step, self.cfg.eval_freq):
                eval_next = True

            if self._step >= self.cfg.seed_steps and self.buffer.size >= self.cfg.batch_size:
                if previous_step < self.cfg.seed_steps <= self._step:
                    num_updates = pretrain_steps
                    print(f'Pretraining agent on seed data for {num_updates} updates...')
                else:
                    num_updates = int(self.cfg.iterations) * len(env_indices)
                if num_updates > 0:
                    for _ in range(num_updates):
                        _train_metrics = self.agent.update(self.buffer)
                    train_metrics.update(_train_metrics)
            self.maybe_save_checkpoint(previous_step)

        self.maybe_save_checkpoint(force=True)
        self.logger.finish(self.agent)
        if self.eval_env is not self.env:
            self.eval_env.close()
        return self._best_eval_metrics
                    

                    
