import numpy as np


class FirstNonRigidEigenvalueRewardMixin:
    """Use the first non-rigid eigenvalue for graph rewards and observations."""

    def _on_model_changed(self) -> None:
        if hasattr(self.mj_model, "set_wcrm"):
            self.mj_model.set_wcrm(False)
        elif hasattr(self.mj_model, "wcrm"):
            self.mj_model.wcrm = False
        self._initial_critical_eig = max(float(self.mj_model._critical_eig()), 1e-8)
        self._observation_rigidity = None
        super()._on_model_changed()

    def reset(self, *args, **kwargs):
        self._observation_rigidity = None
        return super().reset(*args, **kwargs)

    def _rigidity_ratio(self, critical_eig=None):
        if critical_eig is None:
            critical_eig = float(self.mj_model._critical_eig())
        if not np.isfinite(critical_eig):
            return 0.0
        return float(max(critical_eig, 0.0) / self._initial_critical_eig)

    def _current_observation_rigidity(self):
        if self._observation_rigidity is None:
            return self._rigidity_ratio()
        rigidity = self._observation_rigidity
        self._observation_rigidity = None
        return rigidity

    def _compute_reward(self, action, previous_com=None):
        critical_eig_raw = float(self.mj_model._critical_eig())
        critical_eig = self._rigidity_ratio(critical_eig_raw)
        terminated = (
            not np.isfinite(critical_eig_raw)
            or critical_eig < self.config.critical_eig_threshold
        )
        self._observation_rigidity = critical_eig

        com_delta_x = 0.0
        if previous_com is None:
            raw_forward_vel = self.mj_model.get_forward_velocity()
        else:
            current_com = self._center_of_mass()
            com_delta_x = float(current_com[0] - previous_com[0])
            dt = float(self.nsubsteps) * float(self.mj_model.model.opt.timestep)
            raw_forward_vel = 0.0 if dt <= 0.0 else com_delta_x / dt

        reward_forward_vel = float(raw_forward_vel) if np.isfinite(raw_forward_vel) else 0.0
        if self.config.max_forward_velocity is None:
            forward_vel = reward_forward_vel
        else:
            velocity_limit = abs(float(self.config.max_forward_velocity))
            forward_vel = float(np.clip(reward_forward_vel, -velocity_limit, velocity_limit))

        if terminated and self.config.zero_positive_forward_reward_on_termination:
            forward_vel = min(forward_vel, 0.0)

        energy_penalty = float(np.sum(np.square(action)))
        if terminated and self.config.zero_velocity_shaping_on_termination:
            slip_penalty = 0.0
        else:
            slip_penalty = float(self.mj_model.get_slip_penalty(height=self.config.slip_height))
            if not np.isfinite(slip_penalty):
                slip_penalty = 0.0

        forward_reward = (
            self.config.forward_weight
            * forward_vel
            / max(float(self.mj_model.initial_bounding_box_diagonal), 1e-8)
        )
        energy_reward = -self.config.energy_weight * energy_penalty
        rigidity_reward = self.config.rigidity_weight * critical_eig
        if terminated and self.config.zero_rigidity_reward_on_termination:
            rigidity_reward = 0.0
        slip_reward = -self.config.slip_weight * slip_penalty
        alive_reward = float(self.config.alive_bonus)
        if terminated and self.config.zero_alive_bonus_on_termination:
            alive_reward = 0.0
        collapse_penalty = -abs(float(self.config.collapse_penalty)) if terminated else 0.0

        total_reward = (
            forward_reward
            + alive_reward
            + energy_reward
            + rigidity_reward
            + slip_reward
            + collapse_penalty
        )
        info = {
            "forward": forward_reward,
            "forward_velocity": forward_vel,
            "forward_velocity_raw": float(raw_forward_vel),
            "com_delta_x": com_delta_x,
            "alive": alive_reward,
            "energy": energy_reward,
            "rigidity": rigidity_reward,
            "slip": slip_reward,
            "critical_eig": critical_eig,
            "critical_eig_raw": critical_eig_raw,
            "collapse_penalty": collapse_penalty,
            "terminated_by_collapse": terminated,
        }

        return float(total_reward), info, terminated
