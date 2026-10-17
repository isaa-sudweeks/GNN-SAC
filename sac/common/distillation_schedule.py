"""Performance-triggered, irreversible release of topology teacher guidance."""
import math


class PerformanceDecay:
    """Consume matched evaluation summaries; never inspect held-out metrics."""

    def __init__(self, options: dict, tasks: dict[str, str], sources: dict):
        self.ratio = float(options.get("distance_ratio", 0.9))
        self.tolerance = float(options.get("survival_tolerance", 0.05))
        self.patience = int(options.get("patience", 3))
        self.release_steps = int(options.get("release_steps", 100000))
        if not (0 < self.ratio <= 1 and 0 <= self.tolerance <= 1
                and self.patience > 0 and self.release_steps > 0):
            raise ValueError("Invalid performance decay thresholds/duration.")
        references = options.get("references", {})
        self.references = {}
        for task, topology in tasks.items():
            if topology not in references:
                raise ValueError(f"Missing audited performance reference for {topology!r}.")
            ref = dict(references[topology])
            distance = float(ref["distance"])
            survival = float(ref["survival"])
            horizon = int(ref["horizon"])
            if (not math.isfinite(distance) or distance <= 0
                    or not math.isfinite(survival) or not 0 < survival <= 1 or horizon <= 0
                    or ref.get("teacher_sha256") != sources[task]["sha256"]):
                raise ValueError(f"Invalid or mismatched performance reference for {topology!r}.")
            self.references[task] = dict(distance=distance, survival=survival, horizon=horizon,
                                         teacher_sha256=ref["teacher_sha256"])
        self.settings = dict(distance_ratio=self.ratio, survival_tolerance=self.tolerance,
                             patience=self.patience, release_steps=self.release_steps,
                             references=self.references)
        self.progress = {task: dict(streak=0, release_start=None, last_step=-1)
                         for task in tasks}

    def observe(self, task: str, step: int, distance: float, survival: float, horizon: int) -> None:
        """Count distinct online evaluations, resetting the streak on invalid evidence."""
        if task not in self.progress:
            return
        state, ref = self.progress[task], self.references[task]
        if step <= state["last_step"] or state["release_start"] is not None:
            return
        state["last_step"] = step
        qualifies = (horizon == ref["horizon"] and math.isfinite(distance)
                     and math.isfinite(survival) and 0 <= survival <= 1
                     and distance >= self.ratio * ref["distance"]
                     and survival >= ref["survival"] - self.tolerance)
        state["streak"] = state["streak"] + 1 if qualifies else 0
        if state["streak"] >= self.patience:
            state["release_start"] = step

    def multiplier(self, task: str, step: int) -> float:
        start = self.progress[task]["release_start"]
        return 1.0 if start is None else max(0.0, 1.0 - max(0, step - start) / self.release_steps)

    def state_dict(self) -> dict:
        return {task: dict(value) for task, value in self.progress.items()}

    def load_state_dict(self, state: dict) -> None:
        if set(state) != set(self.progress):
            raise ValueError("Performance decay checkpoint tasks changed.")
        for value in state.values():
            streak, start, last = value["streak"], value["release_start"], value["last_step"]
            if (not isinstance(streak, int) or not 0 <= streak <= self.patience
                    or not isinstance(last, int) or last < -1
                    or (start is not None and (not isinstance(start, int) or start < 0
                                               or start > last or streak != self.patience))):
                raise ValueError("Invalid performance decay checkpoint progress.")
        self.progress = {task: dict(value) for task, value in state.items()}
