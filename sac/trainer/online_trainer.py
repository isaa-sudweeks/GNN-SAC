import inspect
from time import time 

import numpy as np 
import re
import torch 
from tensordict.tensordict import TensorDict 
from common.reward_normalizer import TaskRewardNormalizer
from common.training_profiler import TrainingProfiler
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
        self._eval_topology_indices = None
        self._update_budget = 0.0
        self._pending_update_transitions = 0
        self._vector_steps_since_update = 0
        self._pretrain_complete = False
        self._optimizer_updates = 0
        self._last_eval_step = None
        self.reward_normalizer = self._make_reward_normalizer()
        self.performance_profiler = TrainingProfiler.from_config(self.cfg, self.logger)
        self._safety_window_steps = {}
        self._safety_window_active = {}
        
        self.eval_env = self.env
        eval_task = getattr(self.cfg, "eval_task", None)
        has_domain_randomization = getattr(self.cfg, "domain_randomization", False)
        topology_bucket_metadata = self._topology_bucket_metadata(self.env)
        training_backend = str(getattr(self.cfg, "mujoco_backend", "mujoco")).lower()
        eval_backend = str(
            getattr(
                self.cfg,
                "eval_backend",
                "mujoco" if training_backend == "mjx" else training_backend,
            )
        ).lower()
        uses_distinct_eval_backend = eval_backend != training_backend
        
        if (
            (eval_task is not None and eval_task != self.cfg.task)
            or has_domain_randomization
            or topology_bucket_metadata is not None
            or uses_distinct_eval_backend
        ):
            from copy import deepcopy
            from env import make_env
            eval_cfg = deepcopy(self.cfg)
            eval_cfg.domain_randomization = False
            eval_cfg.mujoco_backend = eval_backend
            if training_backend == "mjx" and eval_backend == "mujoco":
                # Native evaluation needs one independent environment per task,
                # not the accelerator-sized training batch.
                eval_cfg.num_envs = 1
            if topology_bucket_metadata is not None:
                # Evaluation needs one isolated slot per topology. Reusing the
                # training buckets would overwrite live rollout state.
                topologies, _ = topology_bucket_metadata
                eval_cfg.num_envs = 1
                if eval_backend == "mujoco":
                    eval_cfg.multitask = True
                    eval_cfg.truss_topologies = list(topologies)
                    eval_cfg.tasks = [f"truss-graph:{topology}" for topology in topologies]
            if eval_task is not None:
                eval_cfg.task = eval_task
                eval_cfg.env_name = eval_task
                if hasattr(eval_cfg, "tasks"):
                    eval_cfg.tasks = [eval_task]
            self.eval_env = make_env(eval_cfg)
            if topology_bucket_metadata is not None:
                eval_topology_metadata = self._topology_bucket_metadata(self.eval_env)
                if eval_topology_metadata is not None:
                    _, representative_indices = eval_topology_metadata
                    self._eval_topology_indices = dict(representative_indices)
                else:
                    self._eval_topology_indices = {
                        topology: env_idx for env_idx, topology in enumerate(topologies)
                    }
            
        self.maybe_load_checkpoint()

    def _make_reward_normalizer(self):
        if not bool(getattr(self.cfg, "normalize_rewards", False)):
            return None
        configured_gamma = getattr(self.cfg, "reward_norm_gamma", None)
        gamma = self.agent.discount if configured_gamma is None else float(configured_gamma)
        task_names = getattr(self.buffer, "task_names", None)
        if task_names is None:
            task_names = [str(getattr(self.cfg, "task", "task"))]
        return TaskRewardNormalizer(
            gamma=gamma,
            epsilon=float(getattr(self.cfg, "reward_norm_epsilon", 1e-8)),
            clip=float(getattr(self.cfg, "reward_norm_clip", 10.0)),
            allowed_tasks=task_names,
        )

    def _normalization_task(self, info):
        task = info.get("task")
        if task is not None:
            return str(task)
        task_names = getattr(self.buffer, "task_names", None)
        if task_names is not None and len(task_names) == 1:
            return str(task_names[0])
        if task_names is None and not bool(getattr(self.cfg, "multitask", False)):
            return str(getattr(self.cfg, "task", "task"))
        raise ValueError("Multi-task reward normalization requires info['task'] on every transition.")

    def _normalize_reward(self, reward, info, *, stream, done):
        reward_normalizer = getattr(self, "reward_normalizer", None)
        if reward_normalizer is None:
            return reward
        return reward_normalizer.normalize(
            reward,
            task=self._normalization_task(info),
            stream=stream,
            done=done,
        )

    def common_metrics(self):
        """
        Return a dictionary of current metrics.
        """
        elapsed_time = time() - self._start_time 
        metrics = dict(
            step= self._step,
            episode= self._ep_idx,
            buffer_size=int(getattr(self.buffer, "size", 0)),
            optimizer_updates=self._optimizer_updates,
            elapsed_time= elapsed_time,
            steps_per_sec= self._step / elapsed_time if elapsed_time > 0 else 0,
        )
        for task, size in getattr(self.buffer, "sizes_by_task", {}).items():
            metrics[f"buffer_size/{self._metric_key(task)}"] = int(size)
        reward_normalizer = getattr(self, "reward_normalizer", None)
        if reward_normalizer is not None:
            for task, task_metrics in reward_normalizer.metrics().items():
                task_key = self._metric_key(task)
                for name, value in task_metrics.items():
                    metrics[f"reward_norm/{task_key}/{name}"] = value
        return metrics

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
        if self._eval_topology_indices is not None:
            return self._eval_topologies(self._eval_topology_indices)
        if self._topology_bucket_metadata(self.eval_env) is not None:
            return self._eval_topology_buckets()
        if bool(getattr(self.cfg, "multitask", False)) and int(getattr(self.eval_env, "num_envs", 1)) > 1:
            return self._eval_multitask()
        task_name = getattr(self.cfg, "eval_task", None)
        if task_name is None and self._safety_enabled():
            safety_tasks = list(getattr(self.agent, "_safety_tasks", []))
            if len(safety_tasks) == 1:
                task_name = safety_tasks[0]
        return self._eval_one(task_name=task_name)

    def _evaluate_and_log(self):
        eval_metrics = self.eval()
        eval_metrics.update(self.common_metrics())
        self.logger.log(eval_metrics, 'eval')
        self.report_eval_metrics(eval_metrics, self._step)
        self._last_eval_step = int(self._step)
        return eval_metrics

    def _evaluate_final_policy(self):
        if not bool(getattr(self.cfg, "eval_at_end", True)):
            return None
        if self._last_eval_step == int(self._step):
            return None
        self._activate_shared_eval_env(0)
        return self._evaluate_and_log()

    def _eval_one(self, task_idx=None, task_name=None, video_key="videos/eval_video"):
        ep_rewards, ep_successes, ep_lengths = [], [], []
        safety_predictions, safety_outcomes = [], []
        for i in range(self.cfg.eval_episodes):
            obs = self.eval_env.reset(task_idx=task_idx) if task_idx is not None else self.eval_env.reset()
            done, ep_reward, t = False, 0, 0
            safety_horizon = self._safety_horizon(task_name) if self._safety_enabled() else None
            if self._safety_enabled():
                safety_predictions.append(self.agent.predict_safety_risk(obs, task=task_name))
            collapsed_within_horizon = False
            if self.cfg.save_video:
                self.logger.video.init(self.eval_env, enabled=(i==0))
            while not done:
                #if getattr(self.cfg, 'device', 'cuda') == 'cuda':
                    #torch.compiler.cudagraph_mark_step_begin()
                action = self.agent.act(obs, t0=t==0, eval_mode=True)
                obs, reward, done, info = self.eval_env.step(action)
                ep_reward += reward
                t += 1
                if self._safety_enabled() and t <= safety_horizon:
                    collapsed_within_horizon = collapsed_within_horizon or bool(
                        self._scalar_value(info.get("collapse_cost", 0.0))
                    )
                if self.cfg.save_video:
                    self.logger.video.record(self.eval_env)
            ep_rewards.append(self._scalar_value(ep_reward))
            ep_successes.append(self._scalar_value(info['success']))
            ep_lengths.append(t)
            if self._safety_enabled():
                safety_outcomes.append(float(collapsed_within_horizon))
            if self.cfg.save_video:
                self.logger.video.save(self._step, key=video_key)
        metrics = dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
            episode_length=np.nanmean(ep_lengths),
        )
        if self._safety_enabled():
            predictions = np.asarray(safety_predictions, dtype=np.float64)
            outcomes = np.asarray(safety_outcomes, dtype=np.float64)
            metrics.update(
                safety_predicted_risk=np.mean(predictions),
                safety_horizon_collapse_rate=np.mean(outcomes),
                safety_brier_score=np.mean(np.square(predictions - outcomes)),
                safety_active_horizon=safety_horizon,
                safety_max_horizon=int(self._safety_horizon()),
            )
        return metrics

    def _eval_multitask(self):
        metrics = {}
        task_rewards, task_successes, task_lengths = [], [], []
        for task_idx in range(int(getattr(self.eval_env, "num_envs", 1))):
            task_name = self._eval_task_name(task_idx)
            task_key = self._metric_key(task_name)
            task_metrics = self._eval_one(
                task_idx=task_idx,
                task_name=task_name,
                video_key=f"videos/eval_video/{task_key}",
            )
            metrics[f"{task_key}_episode_reward"] = task_metrics["episode_reward"]
            metrics[f"{task_key}_episode_success"] = task_metrics["episode_success"]
            metrics[f"{task_key}_episode_length"] = task_metrics["episode_length"]
            if self._safety_enabled():
                metrics[f"{task_key}_safety_predicted_risk"] = task_metrics["safety_predicted_risk"]
                metrics[f"{task_key}_safety_horizon_collapse_rate"] = task_metrics["safety_horizon_collapse_rate"]
                metrics[f"{task_key}_safety_brier_score"] = task_metrics["safety_brier_score"]
                metrics[f"{task_key}_safety_active_horizon"] = task_metrics["safety_active_horizon"]
                metrics[f"{task_key}_safety_max_horizon"] = task_metrics["safety_max_horizon"]
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
        _, representative_indices = self._topology_bucket_metadata(self.eval_env)
        return self._eval_topologies(representative_indices)

    def _eval_topologies(self, topology_indices):
        metrics = {}
        topology_rewards, topology_successes, topology_lengths = [], [], []
        for topology, env_idx in topology_indices.items():
            topology_key = self._metric_key(topology)
            task_name = f"{str(getattr(self.cfg, 'task', 'truss-graph')).split(':', 1)[0]}:{topology}"
            topology_metrics = self._eval_one(
                task_idx=env_idx,
                task_name=task_name,
                video_key=f"videos/eval_video/{topology_key}",
            )
            metrics[f"{topology_key}_episode_reward"] = topology_metrics["episode_reward"]
            metrics[f"{topology_key}_episode_success"] = topology_metrics["episode_success"]
            metrics[f"{topology_key}_episode_length"] = topology_metrics["episode_length"]
            if self._safety_enabled():
                metrics[f"{topology_key}_safety_predicted_risk"] = topology_metrics["safety_predicted_risk"]
                metrics[f"{topology_key}_safety_horizon_collapse_rate"] = topology_metrics["safety_horizon_collapse_rate"]
                metrics[f"{topology_key}_safety_brier_score"] = topology_metrics["safety_brier_score"]
                metrics[f"{topology_key}_safety_active_horizon"] = topology_metrics["safety_active_horizon"]
                metrics[f"{topology_key}_safety_max_horizon"] = topology_metrics["safety_max_horizon"]
            topology_rewards.append(topology_metrics["episode_reward"])
            topology_successes.append(topology_metrics["episode_success"])
            topology_lengths.append(topology_metrics["episode_length"])
        metrics.update(
            episode_reward=np.nanmean(topology_rewards),
            episode_success=np.nanmean(topology_successes),
            episode_length=np.nanmean(topology_lengths),
        )
        return metrics

    def _activate_shared_eval_env(self, env_idx):
        """Select a completed slot only when evaluation reuses the training env."""
        if (
            self.eval_env is self.env
            and self._topology_bucket_metadata(self.eval_env) is None
            and int(getattr(self.eval_env, "num_envs", 1)) > 1
            and hasattr(self.eval_env, "set_active_env")
        ):
            self.eval_env.set_active_env(env_idx)

    def _scheduled_updates(self, collected_transitions):
        """Convert collected transitions into optimizer steps with carry-over."""
        collected_transitions = int(collected_transitions)
        if collected_transitions < 0:
            raise ValueError("collected_transitions must be non-negative")

        legacy_iterations = getattr(self.cfg, "iterations", None)
        if legacy_iterations is not None:
            update_increment = float(legacy_iterations) * collected_transitions
        else:
            replay_ratio = float(getattr(self.cfg, "replay_ratio", 1.0))
            batch_size = int(self.cfg.batch_size)
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            update_increment = replay_ratio * collected_transitions / batch_size

        if not np.isfinite(update_increment) or update_increment < 0:
            raise ValueError("The configured replay/update ratio must be finite and non-negative")

        self._update_budget += update_increment
        num_updates = int(np.floor(self._update_budget + 1e-12))
        self._update_budget -= num_updates
        return num_updates

    def _updates_after_collection(self, collected_transitions, pretrain_steps):
        """Run pretraining once replay is ready, then follow the ratio schedule."""
        if not self._pretrain_complete:
            self._pretrain_complete = True
            self._update_budget = 0.0
            print(f'Pretraining agent on seed data for {pretrain_steps} updates...')
            return int(pretrain_steps)
        return self._scheduled_updates(collected_transitions)

    def _run_agent_updates(self, num_updates):
        self._ensure_performance_profiler()
        update_metrics = {}
        for _ in range(int(num_updates)):
            next_update = self._optimizer_updates + 1
            diagnostics_enabled = bool(getattr(self.cfg, "gradient_diagnostics", False))
            diagnostics_freq = int(getattr(self.cfg, "gradient_diagnostics_freq", 100))
            if diagnostics_enabled and diagnostics_freq <= 0:
                raise ValueError("gradient_diagnostics_freq must be positive")
            run_diagnostics = diagnostics_enabled and next_update % diagnostics_freq == 0
            update_parameters = inspect.signature(self.agent.update).parameters
            update_kwargs = {}
            if "compute_diagnostics" in update_parameters:
                update_kwargs["compute_diagnostics"] = run_diagnostics
            if "performance_profiler" in update_parameters:
                update_kwargs["performance_profiler"] = self.performance_profiler
                update_metrics = self.agent.update(self.buffer, **update_kwargs)
            else:
                with self.performance_profiler.phase("optimization"):
                    update_metrics = self.agent.update(self.buffer, **update_kwargs)
            self._optimizer_updates = next_update
            diagnostics = update_metrics.pop("gradient_diagnostics", None)
            if diagnostics:
                self.logger.log(
                    {"step": self._step, **diagnostics},
                    "gradient_diagnostics",
                )
        return update_metrics

    def _update_cadence(self):
        cadence = int(getattr(self.cfg, "update_every_vector_steps", 8))
        if cadence <= 0:
            raise ValueError("update_every_vector_steps must be positive")
        return cadence

    def _queue_collected_transitions(self, collected_transitions):
        """Accumulate learner work earned by one environment/vector step."""
        collected_transitions = int(collected_transitions)
        if collected_transitions < 0:
            raise ValueError("collected_transitions must be non-negative")
        self._pending_update_transitions += collected_transitions
        self._vector_steps_since_update += 1

    def _updates_due(self, pretrain_steps, *, force=False):
        """Return accumulated optimizer work when the configured cadence is due."""
        if self._pending_update_transitions <= 0:
            return 0
        if not force and self._vector_steps_since_update < self._update_cadence():
            return 0
        buffer_ready = bool(
            getattr(self.buffer, "ready", self.buffer.size >= self.cfg.batch_size)
        )
        if self._step < self.cfg.seed_steps or not buffer_ready:
            return 0

        collected_transitions = self._pending_update_transitions
        self._pending_update_transitions = 0
        self._vector_steps_since_update = 0
        return self._updates_after_collection(collected_transitions, pretrain_steps)

    def _add_trajectory_to_buffer(self, trajectory, *, completed):
        """Store every transition in a trajectory payload."""
        if trajectory is None or len(trajectory) <= 1:
            return 0
        payload = (
            trajectory
            if isinstance(trajectory[0]["obs"], Data)
            else torch.cat(trajectory)
        )
        self._ep_idx = self.buffer.add(payload, count_episode=completed)
        return len(trajectory) - 1

    def _add_transition_to_buffer(self, previous_td, current_td, *, completed, task=None):
        """Store one transition without waiting for its episode to finish."""
        trajectory = [previous_td, current_td]
        payload = trajectory if isinstance(previous_td["obs"], Data) else torch.cat(trajectory)
        if hasattr(self.buffer, "task_names"):
            self._ep_idx = self.buffer.add(payload, count_episode=completed, task=task)
        else:
            self._ep_idx = self.buffer.add(payload, count_episode=completed)
        return 1

    def _log_collection_progress(self, previous_step=None, *, force=False):
        progress_freq = int(getattr(self.cfg, "progress_freq", 10_000) or 0)
        should_log = force
        if not should_log and progress_freq > 0 and previous_step is not None:
            should_log = self._crossed_eval_interval(previous_step, self._step, progress_freq)
        if should_log:
            self.logger.log(self.common_metrics(), 'train')

    def _ensure_performance_profiler(self):
        """Initialize lazily for lightweight trainer fixtures and restored callers."""
        if not hasattr(self, "performance_profiler"):
            self.performance_profiler = TrainingProfiler.from_config(self.cfg, self.logger)
        return self.performance_profiler

    @staticmethod
    def _scalar_value(value):
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Expected a scalar tensor, got shape {tuple(value.shape)}")
            return value.detach().cpu().item()
        return float(value)

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

    def _observation_noise_cfg(self):
        params = self._cfg_get(self.cfg, "domain_randomization_params", {})
        return self._cfg_get(params, "observation_noise", {})

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

    def _should_apply_observation_noise(self):
        if not bool(self._cfg_get(self.cfg, "domain_randomization", False)):
            return False
        noise_cfg = self._observation_noise_cfg()
        if not bool(self._cfg_get(noise_cfg, "enabled", False)):
            return False
        return float(self._cfg_get(noise_cfg, "std", 0.0)) > 0.0

    def _apply_observation_noise(self, obs):
        if not self._should_apply_observation_noise():
            return obs
        noise_cfg = self._observation_noise_cfg()
        std = float(self._cfg_get(noise_cfg, "std", 0.0))
        low = self._cfg_get(noise_cfg, "clip_low", None)
        high = self._cfg_get(noise_cfg, "clip_high", None)
        low = None if low in (None, "null") else low
        high = None if high in (None, "null") else high
        return self._apply_observation_noise_to_value(obs, std, low, high)

    def _apply_observation_noise_to_value(self, value, std, low, high):
        if isinstance(value, Data):
            noisy = value.clone()
            noisy.x = self._apply_observation_noise_to_value(noisy.x, std, low, high)
            return noisy
        if isinstance(value, TensorDict):
            noisy = value.clone()
            for key in noisy.keys():
                noisy[key] = self._apply_observation_noise_to_value(noisy[key], std, low, high)
            return noisy
        if isinstance(value, dict):
            return {
                key: self._apply_observation_noise_to_value(item, std, low, high)
                for key, item in value.items()
            }
        if isinstance(value, torch.Tensor):
            if not torch.is_floating_point(value):
                return value
            noisy = value + torch.randn_like(value) * std
            if low is not None:
                noisy = torch.clamp(noisy, min=float(low))
            if high is not None:
                noisy = torch.clamp(noisy, max=float(high))
            return noisy
        if isinstance(value, np.ndarray):
            if not np.issubdtype(value.dtype, np.floating):
                return value
            noisy = value + np.random.normal(0.0, std, size=value.shape)
            if low is not None or high is not None:
                noisy = np.clip(noisy, low, high)
            return noisy.astype(value.dtype, copy=False)
        return value

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
    
    def _safety_enabled(self):
        safety_cfg = getattr(self.cfg, "safety_constraint", None)
        if safety_cfg is None:
            return False
        if hasattr(safety_cfg, "get"):
            return bool(safety_cfg.get("enabled", False))
        return bool(getattr(safety_cfg, "enabled", False))

    def _safety_horizon(self, task=None):
        active_horizon = getattr(self.agent, "active_safety_horizon", None)
        if task is not None and callable(active_horizon):
            return int(active_horizon(task))
        safety_cfg = getattr(self.cfg, "safety_constraint", None)
        value = safety_cfg.get("horizon", 250) if hasattr(safety_cfg, "get") else getattr(safety_cfg, "horizon", 250)
        return int(value)

    def _learning_reward(self, reward, info):
        if self._safety_enabled():
            if "locomotion_reward" not in info:
                raise KeyError(
                    "Constrained safety requires info['locomotion_reward'] from the environment"
                )
            return info["locomotion_reward"]
        return reward

    def _start_safety_window(self, env_idx):
        if not hasattr(self, "_safety_window_steps"):
            self._safety_window_steps = {}
        if not hasattr(self, "_safety_window_active"):
            self._safety_window_active = {}
        if not hasattr(self, "_safety_window_tasks"):
            self._safety_window_tasks = {}
        if not hasattr(self, "_safety_window_horizons"):
            self._safety_window_horizons = {}
        self._safety_window_steps[int(env_idx)] = 0
        self._safety_window_active[int(env_idx)] = self._safety_enabled()
        self._safety_window_tasks.pop(int(env_idx), None)
        self._safety_window_horizons.pop(int(env_idx), None)

    def _safety_task(self, info):
        task = info.get("task")
        if task is not None:
            return str(task)
        task_names = getattr(self.buffer, "task_names", None)
        if task_names is not None and len(task_names) == 1:
            return str(task_names[0])
        return str(getattr(self.cfg, "task", "task"))

    def _record_safety_transition(self, env_idx, info, done):
        env_idx = int(env_idx)
        if not self._safety_window_active.get(env_idx, False):
            return {}
        task = self._safety_task(info)
        if env_idx not in self._safety_window_horizons:
            self._safety_window_tasks[env_idx] = task
            self._safety_window_horizons[env_idx] = self._safety_horizon(task)
        elif self._safety_window_tasks[env_idx] != task:
            raise ValueError("Safety task changed within an active reset window")
        horizon = self._safety_window_horizons[env_idx]
        steps = self._safety_window_steps.get(env_idx, 0) + 1
        self._safety_window_steps[env_idx] = steps
        collapse = bool(self._scalar_value(info.get("collapse_cost", 0.0)))
        if not (collapse or bool(done) or steps >= horizon):
            return {}
        self._safety_window_active[env_idx] = False
        outcome = 1.0 if collapse and steps <= horizon else 0.0
        observe = getattr(self.agent, "observe_safety_outcome", None)
        if observe is None:
            return {}
        return observe(task, outcome, horizon=horizon)

    def to_td(
        self,
        obs,
        action=None,
        reward=None,
        terminated=None,
        raw_reward=None,
        collapse_cost=None,
        episode_end=None,
    ):
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
        elif not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, dtype=torch.float32)
        if raw_reward is None:
            raw_reward = reward
        elif not isinstance(raw_reward, torch.Tensor):
            raw_reward = torch.as_tensor(raw_reward, dtype=torch.float32)
        if terminated is None:
            terminated = torch.tensor(float('nan'))
        if collapse_cost is None:
            collapse_cost = torch.tensor(float('nan'))
        elif not isinstance(collapse_cost, torch.Tensor):
            collapse_cost = torch.as_tensor(collapse_cost, dtype=torch.float32)
        if episode_end is None:
            episode_end = torch.tensor(float('nan'))
        elif not isinstance(episode_end, torch.Tensor):
            episode_end = torch.as_tensor(episode_end, dtype=torch.float32)
        fields = {
            "obs": obs,
            "action": action.unsqueeze(0),
            "reward": reward.unsqueeze(0),
            "raw_reward": raw_reward.unsqueeze(0),
            "terminated": terminated.unsqueeze(0),
            "collapse_cost": collapse_cost.reshape(()).unsqueeze(0),
            "episode_end": episode_end.reshape(()).unsqueeze(0),
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
        self._ensure_performance_profiler()
        num_envs = int(getattr(self.env, "num_envs", getattr(self.cfg, "num_envs", 1)))
        if num_envs > 1:
            return self._train_multi_env(num_envs)

        train_metrics, done, eval_next = {}, True, False 
        pretrain_steps = int(getattr(self.cfg, 'pretrain_steps', min(self.cfg.seed_steps, 1000)))
        while self._step < self.cfg.steps:
            self.performance_profiler.begin_vector_step(global_step=self._step)
            updates_before_step = self._optimizer_updates
            inserted_transitions = 0
            # Evaluate agent periodically 
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True 
            
            # Reset environment
            if done:
                if eval_next:
                    with self.performance_profiler.phase("evaluation"):
                        self._evaluate_and_log()
                    eval_next = False
                with self.performance_profiler.phase("episode_reset"):
                    if self._step > 0:
                        train_metrics.update(
                            episode_reward=torch.stack([td["raw_reward"].view(()) for td in self._tds[1:]]).sum(),
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
                    obs = self._apply_observation_noise(self.env.reset())
                    self._tds = [self.to_td(obs)]
                    self._episode_reward_components = {}
                    self._start_safety_window(0)
            
            # Collect experience 
            with self.performance_profiler.phase("action_selection"):
                if self._step > self.cfg.seed_steps:
                    action = self.agent.act(obs, t0=len(self._tds)==1)
                else:
                    action = self.env.rand_act()
                action = self._apply_action_noise(action, seed_action=self._step <= self.cfg.seed_steps)
            with self.performance_profiler.phase("environment_step"):
                obs, reward, done, info = self.env.step(action)
            with self.performance_profiler.phase("transition_processing"):
                obs = self._apply_observation_noise(obs)
                self._accumulate_reward_components(info)
                terminated = info['terminated'] if self.cfg.episodic else torch.tensor(0.0)
                previous_td = self._tds[-1]
                learning_reward = self._learning_reward(reward, info)
                normalized_reward = self._normalize_reward(learning_reward, info, stream=0, done=bool(done))
                current_td = self.to_td(
                    obs,
                    action,
                    normalized_reward,
                    terminated,
                    raw_reward=reward,
                    collapse_cost=info.get("collapse_cost", 0.0),
                    episode_end=float(bool(done)),
                )
                self._tds.append(current_td)
                train_metrics.update(self._record_safety_transition(0, info, done))
            with self.performance_profiler.phase("replay_insertion"):
                inserted_transitions += self._add_transition_to_buffer(
                    previous_td,
                    current_td,
                    completed=bool(done),
                    task=info.get("task"),
                )
            self._queue_collected_transitions(inserted_transitions)

            # Update agent 
            previous_step = self._step
            self._step += 1
            with self.performance_profiler.phase("update_scheduling"):
                num_updates = self._updates_due(pretrain_steps)
            if num_updates > 0:
                train_metrics.update(self._run_agent_updates(num_updates))
            if self._step < self.cfg.steps:
                with self.performance_profiler.phase("progress_logging"):
                    self._log_collection_progress(previous_step)
            with self.performance_profiler.phase("checkpoint_dispatch"):
                self.maybe_save_checkpoint(previous_step)
            self.performance_profiler.end_vector_step(
                transitions=inserted_transitions,
                optimizer_updates=self._optimizer_updates - updates_before_step,
                global_step=self._step,
            )
        num_updates = self._updates_due(pretrain_steps, force=True)
        if num_updates > 0:
            train_metrics.update(self._run_agent_updates(num_updates))
        self._log_collection_progress(force=True)
        self._evaluate_final_policy()
        self.maybe_save_checkpoint(force=True)
        self.performance_profiler.finalize(global_step=self._step)
        self.logger.finish(self.agent)
        if self.eval_env is not self.env:
            self.eval_env.close()
        return self._best_eval_metrics

    def _train_multi_env(self, num_envs):
        """
        Train from multiple independent copies of the same task.
        """
        self._ensure_performance_profiler()
        train_metrics, eval_next = {}, False
        done = [True] * num_envs
        infos = [None] * num_envs
        observations = [None] * num_envs
        episode_tds = [None] * num_envs
        reward_components = [None] * num_envs
        pretrain_steps = int(getattr(self.cfg, 'pretrain_steps', min(self.cfg.seed_steps, 1000)))

        while self._step < self.cfg.steps:
            self.performance_profiler.begin_vector_step(global_step=self._step)
            updates_before_step = self._optimizer_updates
            inserted_transitions = 0
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True

            done_indices = [env_idx for env_idx, is_done in enumerate(done) if is_done]
            if done_indices:
                if eval_next:
                    with self.performance_profiler.phase("evaluation"):
                        self._activate_shared_eval_env(done_indices[0])
                        self._evaluate_and_log()
                    eval_next = False
                with self.performance_profiler.phase("episode_reset"):
                    previous_components = self._episode_reward_components
                    for env_idx in done_indices:
                        if episode_tds[env_idx] is not None and len(episode_tds[env_idx]) > 1:
                            info = infos[env_idx]
                            train_metrics.update(
                                episode_reward=torch.stack([td["raw_reward"].view(()) for td in episode_tds[env_idx][1:]]).sum(),
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
                    self._episode_reward_components = previous_components

                    reset_obs = [
                        self._apply_observation_noise(obs)
                        for obs in self.env.reset_many(env_indices=done_indices)
                    ]
                    for env_idx, obs in zip(done_indices, reset_obs):
                        observations[env_idx] = obs
                        episode_tds[env_idx] = [self.to_td(obs)]
                        reward_components[env_idx] = {}
                        done[env_idx] = False
                        self._start_safety_window(env_idx)

            remaining_steps = self.cfg.steps - self._step
            env_indices = list(range(min(num_envs, remaining_steps)))
            if not env_indices:
                break

            with self.performance_profiler.phase("action_selection"):
                actions = self._select_multi_env_actions(observations, episode_tds, env_indices)

            with self.performance_profiler.phase("environment_step"):
                results = self.env.step_many(actions, env_indices=env_indices)
            with self.performance_profiler.phase("transition_processing"):
                pending_insertions = []
                for env_idx, action, (obs, reward, is_done, info) in zip(env_indices, actions, results):
                    obs = self._apply_observation_noise(obs)
                    observations[env_idx] = obs
                    done[env_idx] = is_done
                    infos[env_idx] = info

                    previous_components = self._episode_reward_components
                    self._episode_reward_components = reward_components[env_idx]
                    self._accumulate_reward_components(info)
                    reward_components[env_idx] = self._episode_reward_components
                    self._episode_reward_components = previous_components

                    terminated = info['terminated'] if self.cfg.episodic else torch.tensor(0.0)
                    previous_td = episode_tds[env_idx][-1]
                    learning_reward = self._learning_reward(reward, info)
                    normalized_reward = self._normalize_reward(
                        learning_reward,
                        info,
                        stream=env_idx,
                        done=bool(is_done),
                    )
                    current_td = self.to_td(
                        obs,
                        action,
                        normalized_reward,
                        terminated,
                        raw_reward=reward,
                        collapse_cost=info.get("collapse_cost", 0.0),
                        episode_end=float(bool(is_done)),
                    )
                    episode_tds[env_idx].append(current_td)
                    train_metrics.update(
                        self._record_safety_transition(env_idx, info, is_done)
                    )
                    pending_insertions.append(
                        (previous_td, current_td, bool(is_done), info.get("task"))
                    )
            with self.performance_profiler.phase("replay_insertion"):
                for previous_td, current_td, is_done, task in pending_insertions:
                    inserted_transitions += self._add_transition_to_buffer(
                        previous_td,
                        current_td,
                        completed=is_done,
                        task=task,
                    )

            previous_step = self._step
            self._step += len(env_indices)
            self._queue_collected_transitions(inserted_transitions)
            if self._crossed_eval_interval(previous_step, self._step, self.cfg.eval_freq):
                eval_next = True

            with self.performance_profiler.phase("update_scheduling"):
                num_updates = self._updates_due(pretrain_steps)
            if num_updates > 0:
                train_metrics.update(self._run_agent_updates(num_updates))
            if self._step < self.cfg.steps:
                with self.performance_profiler.phase("progress_logging"):
                    self._log_collection_progress(previous_step)
            with self.performance_profiler.phase("checkpoint_dispatch"):
                self.maybe_save_checkpoint(previous_step)
            self.performance_profiler.end_vector_step(
                transitions=inserted_transitions,
                optimizer_updates=self._optimizer_updates - updates_before_step,
                global_step=self._step,
            )

        num_updates = self._updates_due(pretrain_steps, force=True)
        if num_updates > 0:
            train_metrics.update(self._run_agent_updates(num_updates))
        self._log_collection_progress(force=True)
        self._evaluate_final_policy()
        self.maybe_save_checkpoint(force=True)
        self.performance_profiler.finalize(global_step=self._step)
        self.logger.finish(self.agent)
        if self.eval_env is not self.env:
            self.eval_env.close()
        return self._best_eval_metrics
                    

                    
