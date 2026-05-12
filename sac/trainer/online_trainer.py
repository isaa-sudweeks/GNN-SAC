from time import time 

import numpy as np 
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
    
    def eval(self):
        """
        Evaluate a SAC agent.
        """
        ep_rewards, ep_successes, ep_lengths = [], [], []
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset(), False, 0, 0
            if self.cfg.save_video:
                self.logger.video.init(self.env, enabled=(i==0))
            while not done:
                if getattr(self.cfg, 'device', 'cuda') == 'cuda':
                    torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(obs, t0=t==0, eval_mode=True)
                obs, reward, done, info = self.env.step(action)
                ep_reward += reward
                t += 1
                if self.cfg.save_video:
                    self.logger.video.record(self.env)
            ep_rewards.append(ep_reward)
            ep_successes.append(info['success'])
            ep_lengths.append(t)
            if self.cfg.save_video:
                self.logger.video.save(self._step)
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_length=np.nanmean(ep_lengths),
        )
    
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
        if num_envs > 1 and not bool(getattr(self.cfg, "multitask", False)):
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

            self._step += 1
        self.logger.finish(self.agent)
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
                        self.env.set_active_env(env_idx)
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

            actions = []
            for env_idx in env_indices:
                if self._step > self.cfg.seed_steps:
                    action = self.agent.act(observations[env_idx], t0=len(episode_tds[env_idx]) == 1)
                else:
                    action = self.env.rand_act()
                actions.append(action)

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

        self.logger.finish(self.agent)
        return self._best_eval_metrics
                    

                    
