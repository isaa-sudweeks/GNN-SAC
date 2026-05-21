from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from env.truss.model import MujocoModel


def _load_mjx_deps():
    try:
        import jax
        import jax.numpy as jnp
        from mujoco import mjx
    except ImportError as exc:
        raise ImportError(
            "The batched MJX backend requires CUDA-enabled JAX and mujoco.mjx. "
            "Install a GPU JAX build before using mjx_vectorized=true."
        ) from exc
    return jax, jnp, mjx


@dataclass(frozen=True)
class _TaskDirection:
    axis: int
    sign: float


_TASK_DIRECTIONS = {
    "truss-velocity-command-right": _TaskDirection(axis=0, sign=1.0),
    "truss-velocity-command-left": _TaskDirection(axis=0, sign=-1.0),
    "truss-velocity-command-up": _TaskDirection(axis=1, sign=1.0),
    "truss-velocity-command-down": _TaskDirection(axis=1, sign=-1.0),
}


class BatchedMJXTrussEnv(gym.Env):
    """
    Vectorized MJX implementation for the flat velocity-command truss tasks.

    This keeps all simulator state for all environments in one MJX Data pytree
    and advances selected environments with one JIT-compiled JAX program.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.task = cfg.task
        if self.task not in _TASK_DIRECTIONS:
            raise ValueError(f"Batched MJX truss env does not support task {self.task!r}")
        self.direction = _TASK_DIRECTIONS[self.task]
        self.num_envs = int(getattr(cfg, "num_envs", 1))
        if self.num_envs < 1:
            raise ValueError("BatchedMJXTrussEnv requires at least one environment")

        self.jax, self.jnp, self.mjx = _load_mjx_deps()
        self.xml_path = self._resolve_xml_path(cfg.xml_path)
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.host_data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.host_data)

        metadata_model = MujocoModel(self.xml_path, backend="mujoco")
        self.node_names = metadata_model.node_names
        self.active_axes = metadata_model.active_axes
        self.axis_indices_np = np.asarray(metadata_model.axis_indices, dtype=np.int32)
        self.node_body_ids_np = np.asarray(
            [metadata_model.node_body_ids[name] for name in self.node_names],
            dtype=np.int32,
        )
        node_index = {name: idx for idx, name in enumerate(self.node_names)}
        self.edge_pairs_np = np.asarray(
            [(node_index[a], node_index[b]) for a, b in metadata_model.structural_edges],
            dtype=np.int32,
        )
        self.initial_critical_eig = float(metadata_model.initial_critical_eig)

        self.init_qpos = np.asarray(self.host_data.qpos, dtype=np.float32).copy()
        self.init_qvel = np.asarray(self.host_data.qvel, dtype=np.float32).copy()
        self.ctrl_home = np.zeros(self.model.nu, dtype=np.float32)
        self.act_home = np.ones(self.model.na, dtype=np.float32)
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32, copy=True)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32, copy=True)

        self.nsubsteps = int(getattr(cfg, "nsubsteps", 1))
        self.max_steps = int(getattr(cfg, "max_steps", 1000))
        self.active_env_idx = 0
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.rng = np.random.default_rng(int(getattr(cfg, "seed", 1)))
        self._keys = self.jax.random.split(
            self.jax.random.PRNGKey(int(getattr(cfg, "seed", 1))),
            self.num_envs,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_size = len(self.node_body_ids_np) * len(self.axis_indices_np) * 2 + self.model.nu
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self._mjx_model = self.mjx.put_model(self.model)
        base_data = self.mjx.put_data(self.model, self.host_data)
        self._data = self.jax.vmap(lambda _: base_data)(self.jnp.arange(self.num_envs))

        self._node_body_ids = self.jnp.asarray(self.node_body_ids_np)
        self._axis_indices = self.jnp.asarray(self.axis_indices_np)
        self._edge_pairs = self.jnp.asarray(self.edge_pairs_np)
        self._ctrl_low = self.jnp.asarray(self.ctrl_low)
        self._ctrl_high = self.jnp.asarray(self.ctrl_high)
        self._init_qpos = self.jnp.asarray(self.init_qpos)
        self._init_qvel = self.jnp.asarray(self.init_qvel)
        self._ctrl_home = self.jnp.asarray(self.ctrl_home)
        self._act_home = self.jnp.asarray(self.act_home)
        self._compile_kernels()
        self.reset_many()

    @staticmethod
    def _resolve_xml_path(xml_path):
        path = Path(xml_path).expanduser()
        if path.is_absolute() and path.exists():
            return str(path)
        repo_root = Path(__file__).resolve().parents[2]
        for candidate in (Path.cwd() / path, repo_root / path):
            if candidate.exists():
                return str(candidate.resolve())
        raise FileNotFoundError(f"Could not resolve MuJoCo XML path: {xml_path}")

    def _compile_kernels(self):
        jax = self.jax
        jnp = self.jnp
        mjx = self.mjx
        model = self._mjx_model
        nsubsteps = self.nsubsteps
        node_body_ids = self._node_body_ids
        axis_indices = self._axis_indices
        edge_pairs = self._edge_pairs
        ctrl_low = self._ctrl_low
        ctrl_high = self._ctrl_high
        init_qpos = self._init_qpos
        init_qvel = self._init_qvel
        ctrl_home = self._ctrl_home
        act_home = self._act_home
        initial_critical_eig = self.initial_critical_eig
        forward_axis = self.direction.axis
        forward_sign = self.direction.sign
        forward_weight = float(self.cfg.forward_weight)
        energy_weight = float(self.cfg.energy_weight)
        alive_bonus = float(self.cfg.alive_bonus)
        rigidity_weight = float(self.cfg.rigidity_weight)
        slip_weight = float(self.cfg.slip_weight)
        slip_height = float(self.cfg.slip_height)
        critical_eig_threshold = float(self.cfg.critical_eig_threshold)
        speed = float(self.cfg.speed)
        na = int(self.model.na)
        dims = int(len(self.axis_indices_np))
        num_nodes = int(len(self.node_body_ids_np))
        num_edges = int(len(self.edge_pairs_np))
        rigid_body_modes = dims + (dims * (dims - 1)) // 2

        def obs_one(data):
            node_pos = data.xpos[node_body_ids]
            node_vel = data.cvel[node_body_ids, 3:]
            com = jnp.mean(node_pos, axis=0)

            pos_features = node_pos[:, axis_indices]
            com_features = com[axis_indices]
            relative_mask = axis_indices != 2
            pos_features = jnp.where(relative_mask[None, :], pos_features - com_features[None, :], pos_features)
            vel_features = node_vel[:, axis_indices]
            return jnp.concatenate([pos_features.reshape(-1), vel_features.reshape(-1), data.ctrl])

        def critical_eig_one(data):
            if num_edges == 0:
                return jnp.asarray(0.0, dtype=jnp.float32)
            node_pos = data.xpos[node_body_ids][:, axis_indices]
            pa = node_pos[edge_pairs[:, 0]]
            pb = node_pos[edge_pairs[:, 1]]
            delta = pb - pa
            length = jnp.linalg.norm(delta, axis=1)
            direction = delta / jnp.maximum(length[:, None], 1e-8)
            rows = jnp.zeros((num_edges, num_nodes * dims), dtype=jnp.float32)
            cols_a = edge_pairs[:, 0, None] * dims + jnp.arange(dims)[None, :]
            cols_b = edge_pairs[:, 1, None] * dims + jnp.arange(dims)[None, :]
            edge_rows = jnp.arange(num_edges)[:, None]
            rows = rows.at[edge_rows, cols_a].set(-direction)
            rows = rows.at[edge_rows, cols_b].set(direction)
            eigvals = jnp.linalg.eigvalsh(rows.T @ rows)
            eigvals = jnp.sort(jnp.real(eigvals))
            critical = jnp.where(eigvals.size > rigid_body_modes, eigvals[rigid_body_modes], 0.0)
            return jnp.maximum(critical, 0.0) / initial_critical_eig

        def reward_one(data, action):
            critical_eig = critical_eig_one(data)
            node_pos = data.xpos[node_body_ids]
            node_vel = data.cvel[node_body_ids, 3:]
            forward_vel = jnp.mean(node_vel[:, forward_axis])
            contact_mask = node_pos[:, 2] < slip_height
            slip_penalty = jnp.sum(jnp.abs(node_vel[:, forward_axis]) * contact_mask.astype(jnp.float32))
            energy_penalty = jnp.sum(jnp.square(action))
            forward = forward_weight * forward_sign * forward_vel
            energy = -energy_weight * energy_penalty
            rigidity = rigidity_weight * critical_eig
            slip = -slip_weight * slip_penalty
            reward = forward + alive_bonus + energy + rigidity + slip
            terminated = critical_eig < critical_eig_threshold
            components = jnp.stack([forward, jnp.asarray(alive_bonus), energy, rigidity, slip, reward])
            return reward, terminated, components

        def step_one(data, action):
            action = jnp.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            action = jnp.clip(action, -1.0, 1.0)
            ctrl = jnp.clip(data.ctrl + action * speed, ctrl_low, ctrl_high)
            data = data.replace(ctrl=ctrl)

            def body(_, current):
                return mjx.step(model, current)

            data = jax.lax.fori_loop(0, nsubsteps, body, data)
            obs = obs_one(data)
            reward, terminated, components = reward_one(data, action)
            return data, obs, reward, terminated, components

        def reset_one(key):
            key_qpos, key_qvel = jax.random.split(key)
            qpos = init_qpos + jax.random.uniform(key_qpos, init_qpos.shape, minval=-0.005, maxval=0.005)
            qvel = init_qvel + jax.random.uniform(key_qvel, init_qvel.shape, minval=-0.005, maxval=0.005)
            data = self.mjx.make_data(model)
            kwargs = {"qpos": qpos, "qvel": qvel, "ctrl": ctrl_home}
            if na:
                kwargs["act"] = act_home
            data = data.replace(**kwargs)
            data = mjx.forward(model, data)
            return data, obs_one(data)

        self._step_batch = jax.jit(jax.vmap(step_one))
        self._obs_batch = jax.jit(jax.vmap(obs_one))
        self._reset_batch = jax.jit(jax.vmap(reset_one))

    def _normalize_indices(self, env_indices=None):
        if env_indices is None:
            return np.arange(self.num_envs, dtype=np.int32)
        return np.asarray(list(env_indices), dtype=np.int32)

    def _slice_data(self, indices):
        indices_jax = self.jnp.asarray(indices)
        return self.jax.tree_util.tree_map(lambda x: x[indices_jax], self._data)

    def _scatter_data(self, indices, new_data):
        indices_jax = self.jnp.asarray(indices)
        self._data = self.jax.tree_util.tree_map(
            lambda old, new: old.at[indices_jax].set(new),
            self._data,
            new_data,
        )

    def set_active_env(self, env_idx):
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"Environment index {env_idx} is out of range for {self.num_envs} envs")
        self.active_env_idx = int(env_idx)

    def reset(self, task_idx=None):
        if task_idx is not None:
            self.set_active_env(task_idx)
        return self.reset_many([self.active_env_idx])[0]

    def reset_many(self, env_indices=None):
        indices = self._normalize_indices(env_indices)
        keys = self._keys[indices]
        split_keys = self.jax.vmap(self.jax.random.split)(keys)
        new_keys = split_keys[:, 0]
        reset_keys = split_keys[:, 1]
        keys_host = np.array(self._keys)
        keys_host[indices] = np.asarray(new_keys)
        self._keys = self.jnp.asarray(keys_host)
        new_data, obs = self._reset_batch(reset_keys)
        self._scatter_data(indices, new_data)
        self.steps[indices] = 0
        obs = np.asarray(self.jax.block_until_ready(obs), dtype=np.float32)
        return [obs[i] for i in range(obs.shape[0])]

    def rand_act(self):
        return self.action_space.sample().astype(np.float32)

    def step(self, action):
        return self.step_many([action], [self.active_env_idx])[0]

    def step_many(self, actions, env_indices=None):
        indices = self._normalize_indices(env_indices)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (len(indices), self.model.nu):
            actions = np.reshape(actions, (len(indices), self.model.nu))

        data = self._slice_data(indices)
        new_data, obs, reward, terminated, components = self._step_batch(data, self.jnp.asarray(actions))
        self._scatter_data(indices, new_data)
        self.steps[indices] += 1
        truncated = self.steps[indices] >= self.max_steps
        done = np.asarray(terminated) | truncated

        obs = np.asarray(self.jax.block_until_ready(obs), dtype=np.float32)
        reward = np.asarray(self.jax.block_until_ready(reward), dtype=np.float32)
        terminated = np.asarray(self.jax.block_until_ready(terminated), dtype=bool)
        components = np.asarray(self.jax.block_until_ready(components), dtype=np.float32)

        results = []
        for local_idx, env_idx in enumerate(indices):
            info = {
                "success": False,
                "terminated": bool(terminated[local_idx]),
                "truncated": bool(truncated[local_idx]),
                "task": self.task,
                "env_idx": int(env_idx),
                "forward": float(components[local_idx, 0]),
                "alive": float(components[local_idx, 1]),
                "energy": float(components[local_idx, 2]),
                "rigidity": float(components[local_idx, 3]),
                "slip": float(components[local_idx, 4]),
                "total_raw": float(components[local_idx, 5]),
            }
            results.append((obs[local_idx], float(reward[local_idx]), bool(done[local_idx]), info))
        return results

    @property
    def unwrapped(self):
        return self

    @property
    def max_episode_steps(self):
        return self.max_steps

    @property
    def mj_model(self):
        return self

    def close(self):
        pass
