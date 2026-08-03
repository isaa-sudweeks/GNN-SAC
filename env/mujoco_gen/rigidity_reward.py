import numpy as np


def danger_zone_rigidity_penalty(
    critical_eig,
    *,
    collapse_threshold: float,
    safe_threshold: float,
    weight: float,
    power: float,
    epsilon: float,
    max_penalty: float,
    array_module=np,
):
    """Return a bounded barrier cost inside the rigidity danger zone."""
    if safe_threshold <= collapse_threshold:
        raise ValueError("rigidity_safe_threshold must exceed critical_eig_threshold.")
    if weight < 0.0:
        raise ValueError("rigidity_barrier_weight must be non-negative.")
    if power <= 0.0:
        raise ValueError("rigidity_barrier_power must be positive.")
    if epsilon <= 0.0:
        raise ValueError("rigidity_barrier_epsilon must be positive.")
    if max_penalty < 0.0:
        raise ValueError("rigidity_barrier_max_penalty must be non-negative.")

    xp = array_module
    eig = xp.asarray(critical_eig)
    distance_into_zone = xp.maximum(safe_threshold - eig, 0.0)
    distance_from_collapse = xp.maximum(eig - collapse_threshold, 0.0)
    ratio = distance_into_zone / (distance_from_collapse + epsilon)
    penalty = -weight * xp.power(ratio, power)
    penalty = xp.maximum(penalty, -max_penalty)
    return xp.where(eig >= safe_threshold, 0.0, penalty)


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
        terminated = (
            not np.isfinite(critical_eig_raw)
            or critical_eig_raw < self.config.critical_eig_threshold
        )
        critical_eig = critical_eig_raw if np.isfinite(critical_eig_raw) else 0.0
        self._observation_rigidity = self._rigidity_ratio(critical_eig_raw)

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
        source_config = self.source_config
        rigidity_reward = float(
            danger_zone_rigidity_penalty(
                critical_eig,
                collapse_threshold=float(self.config.critical_eig_threshold),
                safe_threshold=float(source_config.rigidity_safe_threshold),
                weight=float(source_config.rigidity_barrier_weight),
                power=float(source_config.rigidity_barrier_power),
                epsilon=float(source_config.rigidity_barrier_epsilon),
                max_penalty=float(source_config.rigidity_barrier_max_penalty),
            )
        )
        if terminated:
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
            "energy_penalty_raw": energy_penalty,
            "rigidity": rigidity_reward,
            "rigidity_barrier": rigidity_reward,
            "slip": slip_reward,
            "slip_penalty_raw": slip_penalty,
            "critical_eig": critical_eig,
            "critical_eig_raw": critical_eig_raw,
            "collapse_penalty": collapse_penalty,
            "terminated_by_collapse": terminated,
        }

        return float(total_reward), info, terminated
