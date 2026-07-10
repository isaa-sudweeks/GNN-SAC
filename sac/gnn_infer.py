from __future__ import annotations

from pathlib import Path
import argparse
import csv
import os
import pickle
import platform
import re
import sys
import time
import types
import warnings
from urllib.parse import parse_qs, unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SAC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

_default_mujoco_gl = "glfw" if platform.system() == "Darwin" else "egl"
os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", _default_mujoco_gl)
os.environ["LAZY_LEGACY_OP"] = "0"
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

import hydra
import numpy as np
import torch
from omegaconf import open_dict
from termcolor import colored

if sys.version_info >= (3, 14):
    if not getattr(argparse._ActionsContainer._check_help, "_hydra_py314_compat", False):
        _argparse_check_help = argparse._ActionsContainer._check_help

        def _check_help_compat(self, action):
            if action.help is not None and not isinstance(action.help, str):
                return
            return _argparse_check_help(self, action)

        _check_help_compat._hydra_py314_compat = True
        argparse._ActionsContainer._check_help = _check_help_compat

from common.parser import parse_cfg
from common.seed import set_seed
from env import make_env
from gnn_sac import GNNSAC


def _is_wandb_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and parsed.netloc.endswith("wandb.ai")


def _wandb_cache_dir(cfg) -> Path:
    return Path(cfg.work_dir) / "wandb_models"


def _select_checkpoint_file(download_dir: Path, model_file: str | None = None) -> Path:
    if model_file:
        candidate = download_dir / model_file
        if not candidate.exists():
            raise FileNotFoundError(f'W&B artifact did not contain "{model_file}" under {download_dir}')
        return candidate

    preferred = [download_dir / "models" / "final.pt", download_dir / "final.pt"]
    for candidate in preferred:
        if candidate.exists():
            return candidate

    matches = sorted(download_dir.rglob("*.pt"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No .pt checkpoint found in downloaded W&B artifact {download_dir}")
    choices = "\n".join(str(path.relative_to(download_dir)) for path in matches)
    raise ValueError(
        "Multiple .pt files found in the W&B artifact. Set model_file to one of:\n"
        f"{choices}"
    )


def _artifact_ref_from_url(model_ref: str) -> tuple[str, str | None]:
    parsed = urlparse(model_ref)
    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    if len(parts) < 5 or parts[2] != "artifacts":
        raise ValueError(f"Unsupported W&B artifact URL: {model_ref}")

    entity, project = parts[0], parts[1]
    artifact_name = parts[4]
    version = parts[5] if len(parts) > 5 else "latest"
    file_path = parse_qs(parsed.query).get("path", [None])[0]
    if file_path is None and "files" in parts:
        files_index = parts.index("files")
        file_path = "/".join(parts[files_index + 1 :]) or None
    if file_path is not None:
        file_path = file_path.lstrip("/")
    if version.startswith("v") and version[1:].isdigit():
        version = version
    elif ":" in artifact_name:
        artifact_name, version = artifact_name.rsplit(":", 1)
    return f"{entity}/{project}/{artifact_name}:{version}", file_path


def _download_wandb_artifact(model_ref: str, cfg) -> Path:
    import wandb

    if model_ref.startswith("wandb-artifact://"):
        artifact_ref = model_ref.removeprefix("wandb-artifact://")
        url_model_file = None
    elif _is_wandb_url(model_ref):
        artifact_ref, url_model_file = _artifact_ref_from_url(model_ref)
    else:
        artifact_ref = model_ref
        url_model_file = None

    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    root = _wandb_cache_dir(cfg) / re.sub(r"[^0-9a-zA-Z_.-]+", "-", artifact_ref)
    download_dir = Path(artifact.download(root=str(root)))
    return _select_checkpoint_file(download_dir, cfg.model_file or url_model_file)


def _download_wandb_run_file(model_ref: str, cfg) -> Path:
    import wandb

    parsed = urlparse(model_ref)
    parts = [unquote(part) for part in parsed.path.strip("/").split("/") if part]
    try:
        runs_index = parts.index("runs")
        files_index = parts.index("files")
    except ValueError as exc:
        raise ValueError(f"Unsupported W&B run file URL: {model_ref}") from exc

    if runs_index < 2 or files_index <= runs_index + 1:
        raise ValueError(f"Unsupported W&B run file URL: {model_ref}")
    entity, project = parts[0], parts[1]
    run_id = parts[runs_index + 1]
    file_path = "/".join(parts[files_index + 1 :])
    if not file_path:
        raise ValueError(f"W&B run file URL does not include a file path: {model_ref}")

    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    root = _wandb_cache_dir(cfg) / f"{entity}-{project}-{run_id}"
    root.mkdir(parents=True, exist_ok=True)
    downloaded = run.file(file_path).download(root=str(root), replace=True)
    return Path(downloaded.name)


def resolve_checkpoint(model_ref: str, cfg) -> Path:
    path = Path(model_ref).expanduser()
    if path.exists():
        return path

    if _is_wandb_url(model_ref):
        if "/runs/" in urlparse(model_ref).path and "/files/" in urlparse(model_ref).path:
            return _download_wandb_run_file(model_ref, cfg)
        return _download_wandb_artifact(model_ref, cfg)

    if model_ref.startswith("wandb-artifact://") or re.match(r"^[^/]+/[^/]+/[^/]+:[^/]+$", model_ref):
        return _download_wandb_artifact(model_ref, cfg)

    raise FileNotFoundError(
        f"Model reference does not exist locally and is not a recognized W&B reference: {model_ref}"
    )


def _agent_checkpoint_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}.agent{checkpoint.suffix}")


class _AgentStateFound(Exception):
    def __init__(self, state_dict):
        super().__init__("agent state loaded before replay buffer")
        self.state_dict = state_dict


class _AgentOnlyUnpickler(pickle._Unpickler):
    dispatch = pickle._Unpickler.dispatch.copy()

    def _stop_before_replay_buffer(self) -> None:
        if not self.stack or self.stack[-1] != "buffer":
            return
        for index in range(len(self.stack) - 2, -1, -1):
            if self.stack[index] != "agent" or index + 1 >= len(self.stack):
                continue
            agent_state = self.stack[index + 1]
            if isinstance(agent_state, dict) and "model" in agent_state:
                raise _AgentStateFound(agent_state)

    def load_binunicode(self) -> None:
        pickle._Unpickler.load_binunicode(self)
        self._stop_before_replay_buffer()

    def load_short_binunicode(self) -> None:
        pickle._Unpickler.load_short_binunicode(self)
        self._stop_before_replay_buffer()

    def load_binunicode8(self) -> None:
        pickle._Unpickler.load_binunicode8(self)
        self._stop_before_replay_buffer()

    def load_binget(self) -> None:
        pickle._Unpickler.load_binget(self)
        self._stop_before_replay_buffer()

    def load_long_binget(self) -> None:
        pickle._Unpickler.load_long_binget(self)
        self._stop_before_replay_buffer()

    dispatch[pickle.BINUNICODE[0]] = load_binunicode
    dispatch[pickle.SHORT_BINUNICODE[0]] = load_short_binunicode
    dispatch[pickle.BINUNICODE8[0]] = load_binunicode8
    dispatch[pickle.BINGET[0]] = load_binget
    dispatch[pickle.LONG_BINGET[0]] = load_long_binget


class _AgentOnlyPickleModule:
    __name__ = pickle.__name__
    Unpickler = _AgentOnlyUnpickler
    load = pickle.load


def _torch_load(checkpoint: Path, *, pickle_module=None):
    kwargs = {
        "map_location": "cpu",
        "weights_only": False,
        "mmap": True,
    }
    if pickle_module is not None:
        kwargs["pickle_module"] = pickle_module
    try:
        return torch.load(checkpoint, **kwargs)
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        kwargs["mmap"] = False
        return torch.load(checkpoint, **kwargs)


def _load_checkpoint_file(checkpoint: Path, *, agent_only: bool = False):
    if agent_only:
        try:
            return _torch_load(checkpoint, pickle_module=_AgentOnlyPickleModule), False
        except _AgentStateFound as found:
            return found.state_dict, True
    return _torch_load(checkpoint), False


def load_agent_checkpoint(agent: GNNSAC, checkpoint: Path) -> dict:
    agent_checkpoint = _agent_checkpoint_path(checkpoint)
    load_path = checkpoint
    if agent_checkpoint.exists() and agent_checkpoint.stat().st_mtime_ns >= checkpoint.stat().st_mtime_ns:
        load_path = agent_checkpoint

    state_dict, skipped_replay = _load_checkpoint_file(load_path, agent_only=load_path == checkpoint)
    if isinstance(state_dict, dict) and "agent" in state_dict:
        agent_state = state_dict["agent"]
    else:
        agent_state = state_dict
    agent.load(agent_state)

    if load_path == checkpoint and (skipped_replay or isinstance(state_dict, dict) and "buffer" in state_dict):
        try:
            agent.save(agent_checkpoint)
            print(colored("Cached agent-only checkpoint:", "green", attrs=["bold"]), agent_checkpoint)
        except (OSError, RuntimeError) as exc:
            warnings.warn(f"Could not cache agent-only checkpoint at {agent_checkpoint}: {exc}")

    return agent_state if isinstance(agent_state, dict) else {}


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _unwrap_env(env):
    current = env
    while hasattr(current, "env"):
        current = current.env
    return current


def _iter_wrapped_envs(env):
    current = env
    while current is not None:
        yield current
        current = getattr(current, "env", None)


def _simulation_step_seconds(env) -> float:
    current = env
    while current is not None:
        mj_model = getattr(current, "mj_model", None)
        model = getattr(mj_model, "model", None)
        timestep = getattr(getattr(model, "opt", None), "timestep", None)
        if timestep is not None:
            nsubsteps = int(getattr(current, "nsubsteps", 1))
            return float(timestep) * float(nsubsteps)
        current = getattr(current, "env", None)

    render_fps = getattr(getattr(env, "metadata", {}), "get", lambda *_: None)("render_fps")
    if render_fps:
        return 1.0 / float(render_fps)
    render_fps = getattr(_unwrap_env(env), "metadata", {}).get("render_fps", None)
    return 1.0 / float(render_fps) if render_fps else 0.0


def _action_to_numpy(action) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float64)


def _velocity_command_from_action(action, cfg) -> np.ndarray:
    return _action_to_numpy(action) * float(getattr(cfg, "speed", 1.0))


def _command_node_names(env, command: np.ndarray) -> list[str]:
    for env_obj in _iter_wrapped_envs(env):
        node_names = getattr(env_obj, "graph_node_names", None)
        if node_names is None and hasattr(env_obj, "node_velocity_controller"):
            node_names = getattr(env_obj.node_velocity_controller, "node_names", None)
        if node_names is None:
            mj_model = getattr(env_obj, "mj_model", None)
            node_names = getattr(mj_model, "node_names", None)
        if node_names is not None and len(node_names) == command.shape[0]:
            return [str(name) for name in node_names]
    return [f"node_{idx}" for idx in range(command.shape[0])]


def _iterable_names(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        value = value.keys()
    if isinstance(value, str):
        return {value}
    try:
        return {str(name) for name in value}
    except TypeError:
        return set()


def _active_command_nodes(env, node_names: list[str]) -> set[str] | None:
    node_name_set = set(node_names)
    active_attrs = (
        "active_node_names",
        "actuated_node_names",
        "controllable_node_names",
        "controlled_node_names",
        "command_node_names",
    )
    passive_attrs = ("passive_node_names", "inactive_node_names", "uncontrolled_node_names")

    for env_obj in _iter_wrapped_envs(env):
        for owner in (env_obj, getattr(env_obj, "node_velocity_controller", None), getattr(env_obj, "mj_model", None)):
            if owner is None:
                continue
            for attr in active_attrs:
                names = _iterable_names(getattr(owner, attr, None)) & node_name_set
                if names:
                    return names
            for attr in passive_attrs:
                passive = _iterable_names(getattr(owner, attr, None)) & node_name_set
                if passive:
                    return node_name_set - passive

        actuator_edges = getattr(env_obj, "_actuator_edges", None)
        if actuator_edges:
            active = set()
            for node_pair in actuator_edges:
                if node_pair is not None:
                    active.update(str(name) for name in node_pair)
            active &= node_name_set
            if active:
                return active

    return None


def _zero_passive_commands(command: np.ndarray, env) -> tuple[np.ndarray, list[str]]:
    command = np.asarray(command, dtype=np.float64)
    if command.ndim == 1:
        command = command[:, None]
    node_names = _command_node_names(env, command)
    active_nodes = _active_command_nodes(env, node_names)
    if active_nodes is None:
        return command, node_names

    command = command.copy()
    for idx, node_name in enumerate(node_names):
        if node_name not in active_nodes:
            command[idx] = 0.0
    return command, node_names


def _command_dict(command: np.ndarray, env) -> dict[str, float | list[float]]:
    command, node_names = _zero_passive_commands(command, env)
    command_by_node = {}
    for node_name, node_command in zip(node_names, command):
        flattened = np.asarray(node_command, dtype=np.float64).reshape(-1)
        if flattened.size == 1:
            command_by_node[node_name] = float(flattened[0])
        else:
            command_by_node[node_name] = [float(value) for value in flattened]
    return command_by_node


def _update_viewer_camera(env_obj) -> None:
    viewer = getattr(env_obj, "viewer", None)
    if viewer is None:
        return
    if hasattr(env_obj, "_update_camera_lookat"):
        env_obj._update_camera_lookat(viewer.cam)
        return
    mj_model = getattr(env_obj, "mj_model", None)
    if mj_model is not None and hasattr(mj_model, "get_node_position_matrix"):
        viewer.cam.lookat[:] = np.mean(mj_model.get_node_position_matrix(), axis=0)


def _enable_smooth_human_rendering(env, cfg) -> bool:
    if not (bool(getattr(cfg, "visualize", False)) and bool(getattr(cfg, "visualize_smooth", True))):
        return False
    if not bool(getattr(cfg, "visualize_realtime", True)):
        return False

    try:
        import mujoco
    except ImportError:
        return False

    for env_obj in _iter_wrapped_envs(env):
        mj_model = getattr(env_obj, "mj_model", None)
        if mj_model is None or not hasattr(env_obj, "_advance"):
            continue
        model = getattr(mj_model, "model", None)
        data = getattr(mj_model, "data", None)
        if model is None or data is None:
            continue
        if bool(getattr(mj_model, "uses_mjx", False)):
            continue
        original_advance = env_obj._advance
        original_code = getattr(getattr(original_advance, "__func__", original_advance), "__code__", None)
        increments_steps = original_code is not None and "steps" in original_code.co_names

        def smooth_advance(self, ctrl, _mujoco=mujoco, _increments_steps=increments_steps, _cfg=cfg):
            smooth_cfg = getattr(self, "config", _cfg)
            speed = max(float(getattr(smooth_cfg, "visualize_speed", 1.0) or 1.0), 1e-9)
            target_fps = max(float(getattr(smooth_cfg, "visualize_fps", 60) or 60), 1.0)
            nsubsteps = max(int(getattr(self, "nsubsteps", 1)), 1)
            timestep = float(self.mj_model.model.opt.timestep)
            render_every = max(1, int(round(1.0 / (target_fps * timestep))))
            start_time = time.perf_counter()

            if hasattr(self, "_apply_control_noise") and hasattr(self.mj_model, "set_external_ctrl"):
                self.mj_model.set_external_ctrl(self._apply_control_noise(ctrl))
            elif hasattr(self.mj_model, "set_ctrl"):
                self.mj_model.set_ctrl(ctrl)
            else:
                self.mj_model.data.ctrl[:] = ctrl

            for substep in range(nsubsteps):
                if hasattr(self.mj_model, "apply_angle_bisector_control"):
                    self.mj_model.apply_angle_bisector_control()
                _mujoco.mj_step(self.mj_model.model, self.mj_model.data)

                is_render_step = (substep + 1) % render_every == 0 or substep + 1 == nsubsteps
                if self.viewer is not None and is_render_step:
                    _update_viewer_camera(self)
                    self.viewer.sync()
                    target_time = start_time + ((substep + 1) * timestep / speed)
                    sleep_seconds = target_time - time.perf_counter()
                    if sleep_seconds > 0:
                        time.sleep(sleep_seconds)
            if _increments_steps:
                self.steps += 1

        env_obj._advance = types.MethodType(smooth_advance, env_obj)
        return True

    return False


def _render_if_enabled(env, cfg, step_started_at: float | None = None, realtime_pacing: bool = True) -> None:
    if not bool(getattr(cfg, "visualize", False)):
        return
    env.render()
    sleep_seconds = 0.0
    if realtime_pacing and bool(getattr(cfg, "visualize_realtime", True)) and step_started_at is not None:
        speed = max(float(getattr(cfg, "visualize_speed", 1.0) or 1.0), 1e-9)
        target_seconds = _simulation_step_seconds(env) / speed
        elapsed_seconds = time.perf_counter() - step_started_at
        sleep_seconds = max(0.0, target_seconds - elapsed_seconds)
    sleep_seconds += float(getattr(cfg, "visualize_sleep", 0.0) or 0.0)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def _run_vectorized_inference(cfg, env, agent) -> list[dict]:
    """Evaluate independent episodes with one batched actor and environment step."""
    results = []
    episode_count = int(cfg.episodes)
    max_steps = getattr(cfg, "inference_max_steps", None)
    deterministic = bool(cfg.deterministic)
    print_position_command = bool(getattr(cfg, "print_position_command", True))
    position_command_step = getattr(cfg, "position_command_step", None)
    position_command_dt = getattr(cfg, "position_command_dt", None)
    if position_command_dt is None:
        position_command_dt = _simulation_step_seconds(env)
    position_command_dt = float(position_command_dt or 1.0)

    for episode_start in range(0, episode_count, int(env.num_envs)):
        batch_size = min(int(env.num_envs), episode_count - episode_start)
        env_indices = list(range(batch_size))
        observations = env.reset_many(env_indices=env_indices)
        active = list(env_indices)
        rewards = [None] * batch_size
        lengths = [0] * batch_size
        final_info = [{} for _ in range(batch_size)]
        position_commands = [None] * batch_size
        printed_position_commands = [False] * batch_size

        while active:
            active_observations = [observations[env_idx] for env_idx in active]
            actions = agent.act_batch(active_observations, eval_mode=deterministic)
            if print_position_command:
                action_batch = torch.stack(actions).detach().cpu().numpy()
                for action_idx, env_idx in enumerate(active):
                    velocity_command = action_batch[action_idx] * float(getattr(cfg, "speed", 1.0))
                    velocity_command, _ = _zero_passive_commands(velocity_command, env)
                    if position_commands[env_idx] is None:
                        position_commands[env_idx] = np.zeros_like(
                            velocity_command, dtype=np.float64
                        )
                    position_commands[env_idx] += velocity_command * position_command_dt
                    next_step = lengths[env_idx] + 1
                    if position_command_step is not None and next_step == int(position_command_step):
                        episode = episode_start + env_idx
                        print(
                            f"episode={episode} step={next_step} "
                            f"position_command={_command_dict(position_commands[env_idx], env)}"
                        )
                        printed_position_commands[env_idx] = True
            step_results = env.step_many(actions, env_indices=active)
            next_active = []
            for env_idx, (obs, reward, done, info) in zip(active, step_results):
                observations[env_idx] = obs
                rewards[env_idx] = reward if rewards[env_idx] is None else rewards[env_idx] + reward
                lengths[env_idx] += 1
                final_info[env_idx] = info
                hit_limit = max_steps is not None and lengths[env_idx] >= int(max_steps)
                if not done and not hit_limit:
                    next_active.append(env_idx)
            active = next_active

        for env_idx in env_indices:
            episode = episode_start + env_idx
            info = final_info[env_idx]
            if (
                print_position_command
                and position_commands[env_idx] is not None
                and not printed_position_commands[env_idx]
            ):
                print(
                    f"episode={episode} step={lengths[env_idx]} "
                    f"position_command={_command_dict(position_commands[env_idx], env)}"
                )
            row = {
                "episode": episode,
                "episode_reward": _to_float(rewards[env_idx]),
                "episode_success": _to_float(info.get("success", 0.0)),
                "episode_length": lengths[env_idx],
                "terminated": _to_float(info.get("terminated", 0.0)),
                "truncated": _to_float(info.get("truncated", 0.0)),
            }
            results.append(row)
            print(
                f"episode={episode} reward={row['episode_reward']:.6f} "
                f"success={row['episode_success']:.3f} length={row['episode_length']}"
            )
    return results


def _run_serialized_inference(cfg, env, agent, smooth_rendering: bool) -> list[dict]:
    results = []
    for episode in range(int(cfg.episodes)):
        obs = env.reset()
        _render_if_enabled(env, cfg)
        done = False
        ep_reward = 0.0
        ep_success = 0.0
        info = {}
        step = 0
        max_steps = getattr(cfg, "inference_max_steps", None)
        print_position_command = bool(getattr(cfg, "print_position_command", True))
        position_command_step = getattr(cfg, "position_command_step", None)
        position_command_dt = getattr(cfg, "position_command_dt", None)
        if position_command_dt is None:
            position_command_dt = _simulation_step_seconds(env)
        position_command_dt = float(position_command_dt or 1.0)
        position_command = None
        printed_position_command = False
        while not done:
            step_started_at = time.perf_counter()
            action = agent.act(obs, t0=step == 0, eval_mode=bool(cfg.deterministic))
            velocity_command = _velocity_command_from_action(action, cfg)
            velocity_command, _ = _zero_passive_commands(velocity_command, env)
            if position_command is None:
                position_command = np.zeros_like(velocity_command, dtype=np.float64)
            position_command += velocity_command * position_command_dt
            next_step = step + 1
            if (
                print_position_command
                and position_command_step is not None
                and next_step == int(position_command_step)
            ):
                print(
                    f"episode={episode} step={next_step} "
                    f"position_command={_command_dict(position_command, env)}"
                )
                printed_position_command = True
            obs, reward, done, info = env.step(action)
            _render_if_enabled(env, cfg, step_started_at, realtime_pacing=not smooth_rendering)
            ep_reward += _to_float(reward)
            ep_success = float(info.get("success", ep_success))
            step = next_step
            if max_steps is not None and step >= int(max_steps):
                break
        if print_position_command and position_command is not None and not printed_position_command:
            print(
                f"episode={episode} step={step} "
                f"position_command={_command_dict(position_command, env)}"
            )
        row = {
            "episode": episode,
            "episode_reward": ep_reward,
            "episode_success": ep_success,
            "episode_length": step,
            "terminated": _to_float(info.get("terminated", 0.0)),
            "truncated": _to_float(info.get("truncated", 0.0)),
        }
        results.append(row)
        print(
            f"episode={episode} reward={ep_reward:.6f} "
            f"success={ep_success:.3f} length={step}"
        )
    return results


def run_inference(cfg):
    if getattr(cfg, "device", "cuda") == "cuda":
        assert torch.cuda.is_available(), "CUDA not available, please run on a GPU"

    with open_dict(cfg):
        cfg.enable_wandb = False
        cfg.save_agent = False
        cfg.save_csv = False

    cfg = parse_cfg(cfg)
    if str(getattr(cfg, "mujoco_backend", "mujoco")).lower() == "mjx":
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    set_seed(cfg.seed)

    checkpoint = resolve_checkpoint(str(cfg.model), cfg)
    print(colored("Checkpoint:", "yellow", attrs=["bold"]), checkpoint)
    print(colored("Task:", "yellow", attrs=["bold"]), cfg.task)

    env = make_env(cfg)
    smooth_rendering = _enable_smooth_human_rendering(env, cfg)
    agent = GNNSAC(cfg)
    try:
        load_agent_checkpoint(agent, checkpoint)
        agent.model.eval()
        if int(getattr(env, "num_envs", 1)) > 1 and not bool(getattr(cfg, "visualize", False)):
            results = _run_vectorized_inference(cfg, env, agent)
        else:
            results = _run_serialized_inference(cfg, env, agent, smooth_rendering)

        rewards = np.asarray([row["episode_reward"] for row in results], dtype=np.float64)
        successes = np.asarray([row["episode_success"] for row in results], dtype=np.float64)
        print(
            colored("Summary:", "cyan", attrs=["bold"]),
            f"reward_mean={rewards.mean():.6f}",
            f"reward_std={rewards.std():.6f}",
            f"success_mean={successes.mean():.6f}",
        )

        if cfg.output_csv not in (None, "", "???"):
            output_csv = Path(cfg.output_csv).expanduser()
            if not output_csv.is_absolute():
                output_csv = Path(cfg.work_dir) / output_csv
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            with output_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
                writer.writeheader()
                writer.writerows(results)
            print(colored("Wrote:", "green", attrs=["bold"]), output_csv)

        return {
            "episode_reward": float(rewards.mean()),
            "episode_success": float(successes.mean()),
        }
    finally:
        env.close()


@hydra.main(config_name="inference/gnn", config_path="../config", version_base=None)
def infer(cfg):
    """
    Run a trained GNN SAC checkpoint on any configured graph environment.

    Examples:
        python sac/gnn_infer.py model=/path/to/final.pt task=octahedron-graph-right-realistic
        python sac/gnn_infer.py model=entity/project/artifact-name:latest task=octahedron-graph-right-realistic
    """
    return run_inference(cfg)


if __name__ == "__main__":
    infer()
