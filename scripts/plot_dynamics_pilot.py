"""Render figures from saved prerequisite measurements, without fitted claims."""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def save(figure, output, name, title, ytitle):
    figure.update_layout(template="plotly_white", title=title, width=1050, height=560,
                         font=dict(size=16), margin=dict(l=90, r=35, b=85, t=90),
                         yaxis_title=ytitle, legend=dict(orientation="h", y=-0.22))
    figure.write_image(output / f"{name}.png", scale=1.5)
    figure.write_html(output / f"{name}.html", include_plotlyjs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/dynamics-pilot"))
    args = parser.parse_args()
    output = args.root / "figures"
    output.mkdir(exist_ok=True)
    state = json.loads((args.root / "state/report.json").read_text())["variants"]
    figure = go.Figure()
    labels = ["Physical graph", "Control graph"]
    for label, key in (("Prior command changed", "omitted_control_next_position_max_difference_m"),
                       ("Actuator target changed", "omitted_activation_next_position_max_difference_m")):
        figure.add_bar(name=label, x=labels, y=[v[key] * 1000 for v in state],
                       text=[f"{v[key]*1000:.3f}" for v in state], textposition="outside")
    figure.update_layout(barmode="group")
    save(figure, output, "hidden-state", "Same visible state + same action: hidden state changes the result",
         "Largest next-node-position difference (mm)")
    branch = json.loads((args.root / "simulator-branch.json").read_text())["benchmarks"]
    figure = go.Figure()
    for row in branch:
        figure.add_box(name=f"{row['horizon']} steps ({row['requested_seconds']:g} s)",
                       y=row["candidate_seconds"], boxpoints="all", jitter=0.2)
    save(figure, output, "branch-latency", "Measured native simulator time per candidate (16 sequences per horizon)",
         "Branch restore + rollout time (wall seconds)")
    old_validation = args.root / "source-evaluation-probe/validation.json"
    new_validation = args.root / "source-evaluation-probe-v2/validation.json"
    if old_validation.exists() and new_validation.exists():
        figure = go.Figure()
        for label, path in (("Integration state only", old_validation), ("Full data + wrapper snapshot", new_validation)):
            checks = json.loads(path.read_text())["restored_checks"]
            figure.add_scatter(name=label, x=[f"{c['episode_id']}:{c['timestep']}" for c in checks],
                               y=[c["raw_reward_error"] for c in checks], mode="lines+markers")
        figure.update_xaxes(title="Saved episode : timestep (same 12 indexed transitions)", tickangle=-45)
        save(figure, output, "snapshot-reward-check", "Saved-state reproduction: full snapshots preserve raw reward",
             "Absolute raw reward difference after restoration")
        manifest_path = args.root / "source-evaluation-probe-v2/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        figure = go.Figure()
        for episode in manifest["episodes"]:
            with np.load(manifest_path.parent / episode["file"]) as arrays:
                positions = np.concatenate([arrays["physical_positions"][:1], arrays["next_physical_positions"]])
                mean_x = positions.mean(axis=1)[:, 0]
                figure.add_scatter(name=f"Seed {episode['seed']}", mode="lines",
                                   x=np.arange(len(mean_x)) * manifest["control_interval_seconds"],
                                   y=(mean_x-mean_x[0])*1000)
        figure.update_xaxes(title="Simulated seconds")
        save(figure, output, "evaluation-probe-movement", "Evaluation logging probe: four smooth-random source episodes",
             "Physical-node mean forward movement (mm)")
    path = args.root / "simulator-mpc-source.json"
    if path.exists():
        data = json.loads(path.read_text())
        results = data["results"]
        seeds = sorted({r["seed"] for r in results if r["controller"] == "simulator_mpc"})
        colors = {"zero": "#8b9095", "smooth_random": "#d99832", "simulator_mpc": "#1677a2"}
        figure = go.Figure()
        for controller in colors:
            rows = [r for r in results if r["controller"] == controller and r["seed"] in seeds]
            figure.add_scatter(name=controller.replace("_", " "), mode="lines+markers",
                               x=[str(r["seed"]) for r in rows], y=[r["displacement"] for r in rows],
                               line_color=colors[controller])
        figure.update_xaxes(title="Paired reset seed (exact initial state shared within each seed)")
        save(figure, output, "source-displacement", f"Source tetrahedron: {len(seeds)} completed paired resets, 8 simulated seconds",
             "Signed forward movement / initial diagonal")
        figure = make_subplots(rows=1, cols=max(1, len(seeds)), shared_yaxes=True,
                               subplot_titles=[str(seed) for seed in seeds])
        for column, seed in enumerate(seeds, 1):
            for controller in colors:
                row = next(r for r in results if r["seed"] == seed and r["controller"] == controller)
                position = np.asarray(row["node_mean_position"])[:, 0]
                figure.add_scatter(x=np.arange(len(position)) * data["control_interval_seconds"],
                                   y=position-position[0], mode="lines", name=controller.replace('_', ' '),
                                   line_color=colors[controller], showlegend=column == 1,
                                   row=1, col=column)
        figure.update_xaxes(title="Simulated seconds", row=1, col=max(1, (len(seeds)+1)//2))
        save(figure, output, "source-trajectories", "Individual source trials: forward movement through time",
             "Forward movement (m)")
        figure = go.Figure()
        for row in results:
            if row["controller"] == "simulator_mpc":
                figure.add_box(name=str(row["seed"]), y=row["planning_seconds"], boxpoints="all")
        figure.add_hline(y=data["control_interval_seconds"], line_dash="dash",
                         annotation_text="0.2 s control interval")
        figure.update_xaxes(title="Reset seed")
        save(figure, output, "source-planning-latency", "Actual CEM planning time: diagnostic runs slower than the control interval",
             "Planning time per decision (wall seconds)")
    print(output.resolve())


if __name__ == "__main__":
    main()
