"""Small isolated preview of the exact estimated policy input positions."""

from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full
import sys
import time

import numpy as np


def draw_frame(ax, names, edges, frame):
    """Draw a frame; also used for headless rendering tests."""
    ax.clear()
    held = frame["held_tracker_ages"]
    tracking = "Live tracking"
    if held:
        ages = ", ".join(f"{tracker} ({age:.1f}s)" for tracker, age in sorted(held.items()))
        tracking = f"Held poses: {ages}"
    status = "PAUSED" if frame["paused"] else "RUNNING"
    sample_age = time.monotonic() - frame["generated_at"]
    stale = f"\nDISPLAY SAMPLE {sample_age:.1f}s OLD" if sample_age > .5 else ""
    if frame["error"]:
        tracking = f"{tracking}\n{frame['error']}" if held else frame["error"]
        ax.set_axis_off()
        status = "WAITING FOR POSES"
    else:
        ax.set_axis_on()
        _draw_graph(ax, names, edges, np.asarray(frame["positions"]))
    ax.set_title(
        f"{status} | {tracking}{stale}\nPolicy position inputs (m) | Ctrl+P pause | E emergency stop",
        fontsize=9,
        color="darkorange" if held or frame["error"] or stale else "black",
    )


def _draw_graph(ax, names, edges, positions):
    from matplotlib.ticker import MaxNLocator

    ax.scatter(*positions.T, color="tab:blue", s=18)
    for a, b in edges:
        line = positions[[a, b]]
        ax.plot(*line.T, color="steelblue", linewidth=1)
    grouped = {}
    for name, point in zip(names, positions):
        grouped.setdefault(tuple(np.round(point, 9)), []).append(name)
    for point, node_names in grouped.items():
        label = node_names[0]
        if len(node_names) > 1:
            label = f"{label.split('_tri_')[0]} ({len(node_names)} copies)"
        ax.text(*point, label, fontsize=7)
    center = positions.mean(axis=0)
    span = max(float(np.ptp(positions, axis=0).max()), 0.2)
    for axis, label in enumerate(("x", "y", "z")):
        getattr(ax, f"set_{label}lim")(center[axis] - span * .65, center[axis] + span * .65)
        getattr(ax, f"set_{label}label")(f"{label} (m)")
        getattr(ax, f"{label}axis").set_major_locator(MaxNLocator(5))
    ax.set_box_aspect((1, 1, 1))


def _preview_process(names, edges, frames, events):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=.15, top=.76)
    pause = Button(fig.add_axes([.14, .03, .3, .06]), "Pause / resume (Ctrl+P)")
    stop = Button(fig.add_axes([.55, .03, .3, .06]), "EMERGENCY STOP", color="salmon")
    pause.on_clicked(lambda _: events.put("toggle_pause"))
    stop.on_clicked(lambda _: events.put("stop"))

    def key(event):
        if event.key == "ctrl+p":
            events.put("toggle_pause")
        elif event.key in ("e", "E", "ctrl+c"):
            events.put("stop")
    fig.canvas.mpl_connect("key_press_event", key)

    last_frame = None

    def update():
        nonlocal last_frame
        try:
            while True:
                last_frame = frames.get_nowait()
        except Empty:
            pass
        if last_frame is not None:
            draw_frame(ax, names, edges, last_frame)
            fig.canvas.draw_idle()
    timer = fig.canvas.new_timer(interval=80)
    timer.add_callback(update)
    timer.start()
    plt.show()


class RobotPreview:
    """One-slot best-effort display queue: rendering cannot block motor control."""

    def __init__(self, names: tuple[str, ...], edge_index: np.ndarray) -> None:
        context = mp.get_context("spawn")
        self.frames = context.Queue(maxsize=1)
        self.events = context.Queue()
        edges = sorted({tuple(sorted((int(a), int(b)))) for a, b in np.asarray(edge_index).T})
        self.process = context.Process(
            target=_preview_process, args=(names, edges, self.frames, self.events), daemon=True
        )
        self.process.start()
        self.warned = False

    def publish(
        self,
        positions: np.ndarray | None,
        held_tracker_ages: dict[str, float],
        paused: bool,
        error: str | None = None,
    ) -> None:
        if not self.process.is_alive():
            if not self.warned:
                print("WARNING: state preview closed. Press Ctrl+C in the terminal to stop.", file=sys.stderr, flush=True)
                self.warned = True
            return
        frame = {
            "positions": None if positions is None else positions.tolist(),
            "held_tracker_ages": held_tracker_ages,
            "paused": paused,
            "error": error,
            "generated_at": time.monotonic(),
        }
        try:
            self.frames.put_nowait(frame)
        except Full:
            pass

    def close(self) -> None:
        self.process.terminate()
        self.process.join(timeout=1)
        self.frames.cancel_join_thread()
        self.frames.close()
        self.events.close()
