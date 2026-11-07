"""Small isolated preview of the exact estimated policy input positions."""

from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full
import sys
import time

import numpy as np


def draw_frame(ax, names, edges, frame):
    """Draw a frame; also used for headless rendering tests."""
    from matplotlib.ticker import MaxNLocator
    ax.clear()
    positions = np.asarray(frame["positions"])
    ax.scatter(*positions.T, color="tab:blue", s=18)
    for a, b in edges:
        line = positions[[a, b]]
        ax.plot(*line.T, color="steelblue", linewidth=1)
    grouped = {}
    for name, point in zip(names, positions):
        grouped.setdefault(tuple(np.round(point, 9)), []).append(name)
    for point, node_names in grouped.items():
        label = node_names[0] if len(node_names) == 1 else f"{node_names[0].split('_tri_')[0]} ({len(node_names)} copies)"
        ax.text(*point, label, fontsize=7)
    center = positions.mean(axis=0)
    span = max(float(np.ptp(positions, axis=0).max()), 0.2)
    for axis, label in enumerate(("x", "y", "z")):
        getattr(ax, f"set_{label}lim")(center[axis] - span * .65, center[axis] + span * .65)
        getattr(ax, f"set_{label}label")(f"{label} (m)")
        getattr(ax, f"{label}axis").set_major_locator(MaxNLocator(5))
    ax.set_box_aspect((1, 1, 1))
    health = frame.get("health", {})
    transform = health.get("world_from_shape")
    if transform:
        rotation = np.asarray(transform["rotation_matrix"])
        for i, color in enumerate(("red", "green", "blue")):
            ax.quiver(*center, *rotation[:, i], length=span * .2, color=color)
    status = "PAUSED" if frame.get("paused") else "RUNNING"
    tracking = health.get("status", "unknown")
    tracking_label = {"two_trackers": "Two trackers", "one_tracker": "One tracker",
                      "held_global_transform": "Global pose held", "uninitialized": "Waiting for trackers"}.get(tracking, tracking)
    age = health.get("global_pose_age_s")
    suffix = "" if age is None else f" | anchor age {age:.2f}s"
    encoder = "" if health.get("encoder_freshness_verified", True) else "\nENCODER FRESHNESS UNVERIFIED"
    disagreement = ""
    if health.get("anchor_consistent") is False:
        disagreement = (f"\nTRACKER POSES DISAGREE: {health['position_disagreement_m']:.2f} m, "
                        f"{np.degrees(health['rotation_disagreement_rad']):.1f} deg")
    sample_age = time.monotonic() - frame.get("generated_at", time.monotonic())
    stale = f"\nDISPLAY SAMPLE {sample_age:.1f}s OLD" if sample_age > .5 else ""
    ax.set_title(f"{status} | {tracking_label}{suffix}{encoder}{disagreement}{stale}\nExact policy positions; Ctrl+P pause, E emergency stop", fontsize=9,
                 color="firebrick" if disagreement else "darkorange" if tracking != "two_trackers" or encoder else "black")


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
            if last_frame.get("close"):
                plt.close(fig)
                return
            draw_frame(ax, names, edges, last_frame)
            fig.canvas.draw_idle()
    timer = fig.canvas.new_timer(interval=80)
    timer.add_callback(update)
    timer.start()
    plt.show()


class RobotPreview:
    """One-slot best-effort display queue: rendering cannot block motor control."""

    def __init__(self, names, edge_index):
        context = mp.get_context("spawn")
        self.frames = context.Queue(maxsize=1)
        self.events = context.Queue()
        edges = sorted({tuple(sorted((int(a), int(b)))) for a, b in np.asarray(edge_index).T})
        self.process = context.Process(target=_preview_process, args=(list(names), edges, self.frames, self.events), daemon=True)
        self.process.start()
        self.warned = False

    def publish(self, positions, health, paused):
        if not self.process.is_alive():
            if not self.warned:
                print("WARNING: state preview closed; terminal controls remain available", file=sys.stderr, flush=True)
                self.warned = True
            return
        frame = {"positions": np.asarray(positions).tolist(), "health": health, "paused": paused, "generated_at": time.monotonic()}
        try:
            self.frames.put_nowait(frame)
        except Full:
            pass

    def close(self):
        self.process.terminate()
        self.process.join(timeout=1)
        self.frames.cancel_join_thread()
        self.frames.close()
        self.events.close()
