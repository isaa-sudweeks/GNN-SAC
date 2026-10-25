"""Source-only fitting and independent-episode recursive dynamics evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from .model import ObservableDynamicsGNN, physical_graph_loss
from .rigidity import normalized_rigidity


def tensor_graph(episode: dict, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Convert static graph metadata, preserving physical alias semantics."""
    node_to_physical = np.asarray(episode["node_to_physical"], dtype=np.int64)
    physical = np.asarray(episode["physical_mask"], dtype=bool)
    representatives = {int(node_to_physical[index]): int(index) for index in np.flatnonzero(physical)}
    rep = np.array([representatives[int(index)] for index in node_to_physical])
    multiplicity = np.array([(node_to_physical == index).sum() for index in node_to_physical])
    edges = np.asarray(episode["edge_index"], dtype=np.int64)
    initial = np.asarray(episode["state"])[0, :, :3]
    edge_static = np.asarray(episode.get("edge_static", np.zeros((edges.shape[1], 2))))
    result = {
        "edge_index": torch.as_tensor(edges, dtype=torch.long, device=device),
        "physical_mask": torch.as_tensor(physical, device=device),
        "representative_index": torch.as_tensor(rep, dtype=torch.long, device=device),
        "physical_order_index": torch.as_tensor([representatives[i] for i in range(len(representatives))], dtype=torch.long, device=device),
        "action_mask": torch.as_tensor(episode["action_mask"], dtype=torch.float32, device=device),
        "multiplicity": torch.as_tensor(multiplicity, dtype=torch.float32, device=device),
        "edge_role": torch.as_tensor(edge_static[:, 0], dtype=torch.float32, device=device),
        "initial_edge_length": torch.as_tensor(episode.get("initial_edge_length",
            np.linalg.norm(initial[edges[1]] - initial[edges[0]], axis=-1)), dtype=torch.float32, device=device),
    }
    if "routing_incidence" in episode:
        result["routing_incidence"] = torch.as_tensor(episode["routing_incidence"], dtype=torch.float32, device=device)
        result["ctrlrange"] = torch.as_tensor(episode["ctrlrange"], dtype=torch.float32, device=device)
        result["speed"] = torch.tensor(float(episode["speed"]), dtype=torch.float32, device=device)
    return result


def target_deltas(episode: dict) -> np.ndarray:
    """Express next positions in the current centered coordinate frame."""
    delta = np.asarray(episode["next_state"]) - np.asarray(episode["state"])
    delta = delta.copy()
    delta[:, :, :2] += np.asarray(episode["progress"])[:, None]
    return delta


def fitting_statistics(episodes: list[dict]) -> dict:
    """Compute topology-balanced scales using fitting episodes only."""
    if not episodes or any(episode["role"] not in ("train", "fitting") for episode in episodes):
        raise ValueError("Normalization must use fitting-role episodes only")
    by_topology: dict[str, list] = {}
    for episode in episodes:
        by_topology.setdefault(episode["topology"], []).append(episode)
    state_mean, state_second, delta_second, geom = [], [], [], []
    for values in by_topology.values():
        states = np.concatenate([e["state"][:, np.asarray(e["physical_mask"], bool)].reshape(-1, 6) for e in values])
        deltas = np.concatenate([target_deltas(e)[:, np.asarray(e["physical_mask"], bool)].reshape(-1, 6) for e in values])
        state_mean.append(states.mean(0)); state_second.append((states ** 2).mean(0))
        delta_second.append((deltas ** 2).mean(0))
        geom.append(float(values[0]["initial_diagonal"]))
    mean = np.mean(state_mean, axis=0)
    scale = np.sqrt(np.maximum(np.mean(state_second, axis=0) - mean ** 2, 1e-6))
    delta_scale = np.sqrt(np.maximum(np.mean(delta_second, axis=0), 1e-6))
    return dict(mean=mean.tolist(), scale=scale.tolist(), delta_scale=delta_scale.tolist(),
                geometry_scale=float(np.mean(geom)), fitting_topologies=sorted(by_topology),
                fitting_episodes=len(episodes), fitting_transitions=sum(len(e["state"]) for e in episodes))


def _cache(episodes: list[dict], device: str) -> list[dict]:
    result = []
    for episode in episodes:
        item = dict(episode)
        item["graph"] = tensor_graph(episode, device)
        for key in ("state", "next_state", "action", "progress", "dt"):
            item[key] = torch.as_tensor(episode[key], dtype=torch.float32, device=device)
        item["target"] = torch.as_tensor(target_deltas(episode), dtype=torch.float32, device=device)
        result.append(item)
    return result


@torch.no_grad()
def validation_loss(model: ObservableDynamicsGNN, episodes: list[dict]) -> float:
    """One-step source validation loss, equally weighted by topology then episode."""
    values: dict[str, list[float]] = {}
    model.eval()
    for episode in episodes:
        prediction = model(episode["state"], episode["action"], episode["graph"])
        loss = physical_graph_loss(prediction, episode["target"], episode["graph"]["physical_mask"], model.delta_scale)
        values.setdefault(episode["topology"], []).append(float(loss))
    return float(np.mean([np.mean(v) for v in values.values()]))



def rollout_training_loss(model: ObservableDynamicsGNN, episode: dict,
                          starts: torch.Tensor, horizon: int) -> torch.Tensor:
    """Compare recursive observable states and cumulative progress, without resets.

    Only fitting-episode targets supervise gradients; future true states never
    enter the recurrent model input. Windows remain inside one indexed episode.
    """
    if horizon < 1 or bool((starts < 0).any()) or bool((starts + horizon > len(episode["state"])).any()):
        raise ValueError("Rollout window crosses episode boundary")
    predicted = episode["state"][starts]
    movement = torch.zeros((len(starts), 2), device=predicted.device)
    truth_movement = torch.zeros_like(movement)
    loss = predicted.new_zeros(())
    for offset in range(horizon):
        predicted, progress = model.step(predicted, episode["action"][starts + offset], episode["graph"])
        movement = movement + progress
        truth_movement = truth_movement + episode["progress"][starts + offset]
        loss = loss + physical_graph_loss(predicted, episode["next_state"][starts + offset],
            episode["graph"]["physical_mask"], model.state_scale)
        progress_scale = model.delta_scale[:2] * (offset + 1)
        loss = loss + (((movement - truth_movement) / progress_scale) ** 2).mean()
    return loss / horizon


@torch.no_grad()
def validation_progress_loss(model: ObservableDynamicsGNN, episodes: list[dict], horizon: int) -> float:
    """Select repaired checkpoints by topology-balanced planning-horizon progress."""
    model.eval()
    values: dict[str, list[float]] = {}
    for episode in episodes:
        count = len(episode["state"]) - horizon + 1
        if count < 1:
            raise ValueError("Validation episode is shorter than planning horizon")
        starts = torch.arange(count, device=episode["state"].device)
        state = episode["state"][starts]
        movement = state.new_zeros((count, 2))
        truth = state.new_zeros((count, 2))
        for offset in range(horizon):
            state, progress = model.step(state, episode["action"][starts + offset], episode["graph"])
            movement = movement + progress
            truth = truth + episode["progress"][starts + offset]
        error = (movement[:, 0] - truth[:, 0]).abs() / float(episode["initial_diagonal"])
        error = torch.where(torch.isfinite(error), error, torch.full_like(error, 100.))
        values.setdefault(episode["topology"], []).append(float(error.mean()))
    return float(np.mean([np.mean(v) for v in values.values()]))


def fit_model(fitting: list[dict], validation: list[dict], settings: dict,
              output: Path) -> tuple[ObservableDynamicsGNN, dict]:
    """Fit a predeclared source baseline and select by independent validation."""
    if any(e["role"] != "validation" for e in validation):
        raise ValueError("Checkpoint selection requires validation-role episodes")
    fit_ids = {(e["topology"], e.get("seed", e.get("reset_seed", e.get("episode_id")))) for e in fitting}
    val_ids = {(e["topology"], e.get("seed", e.get("reset_seed", e.get("episode_id")))) for e in validation}
    if fit_ids & val_ids:
        raise ValueError("Fitting and validation episode identities overlap")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(settings["threads"]))
    torch.manual_seed(int(settings["seed"]))
    np.random.seed(int(settings["seed"]))
    random.seed(int(settings["seed"]))
    device = settings["device"]
    stats = fitting_statistics(fitting)
    model = ObservableDynamicsGNN(settings["width"], settings["layers"]).to(device)
    model.configure_normalization(torch.tensor(stats["mean"], device=device),
                                  torch.tensor(stats["scale"], device=device),
                                  torch.tensor(stats["delta_scale"], device=device), stats["geometry_scale"])
    if settings.get("initial_checkpoint"):
        initial = torch.load(Path(settings["initial_checkpoint"]), map_location=device, weights_only=False)
        model.load_state_dict(initial["model"])
    training, valid = _cache(fitting, device), _cache(validation, device)
    rollout_horizon = int(settings.get("rollout_horizon", 1))
    rollout_weight = float(settings.get("rollout_weight", 0.))
    if rollout_horizon < 1 or rollout_weight < 0:
        raise ValueError("Invalid rollout horizon or weight")
    by_topology: dict[str, list[dict]] = {}
    for episode in training:
        by_topology.setdefault(episode["topology"], []).append(episode)
    topologies = sorted(by_topology)
    rng = np.random.default_rng(int(settings["seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    nominal_dt = float(np.median(np.concatenate([np.asarray(e['dt']) for e in fitting])))
    timing = {role: dict(transitions=sum(len(e['dt']) for e in values),
        short_dt_transitions=sum(int((np.asarray(e['dt']) < nominal_dt - 1e-7).sum()) for e in values),
        min_dt=min(float(np.asarray(e['dt']).min()) for e in values))
        for role, values in [('fitting', fitting), ('validation', validation)]}
    protocol = dict(settings=settings, statistics=stats, nominal_control_interval=nominal_dt,
                    timing=timing, time_contract='Fixed commanded control interval; actual shortened terminal dt is a label diagnostic, not a future-known model input',
                    checkpoint_rule=("lowest topology-balanced independent-validation planning-horizon forward progress MAE"
                                     if rollout_weight else "lowest topology-balanced independent-validation one-step normalized MSE"),
                    initial_checkpoint_sha256=(hashlib.sha256(Path(settings["initial_checkpoint"]).read_bytes()).hexdigest()
                                               if settings.get("initial_checkpoint") else None),
                    rollout_update_to_data_ratio=settings["updates"] * settings["batch_size"] * rollout_horizon / stats["fitting_transitions"],
                    sampling="cycle source topologies; uniformly sample fitting episodes and transitions within each",
                    update_to_data_ratio=settings["updates"] * settings["batch_size"] / stats["fitting_transitions"],
                    stopping=("fixed maximum updates; patience stopping disabled"
                              if not settings.get("early_stopping", True)
                              else "maximum updates or patience checks without improved validation after minimum_updates"),
                    evaluation="all valid recursive windows; nonfinite failures retained with count and fixed 100-diagonal error",
                    forbidden="evaluation probe seeds 8100–8103, held-out topologies, hidden actuator model inputs")
    (output / "fit-protocol.json").write_text(json.dumps(protocol, indent=2))
    started = time.perf_counter()
    best, stale, best_update = float("inf"), 0, 0
    history = []
    interval_losses, interval_one_step, interval_rollout = [], [], []
    interval_topologies: dict[str, list[float]] = {}
    for update in range(1, int(settings["updates"]) + 1):
        topology = topologies[(update - 1) % len(topologies)]
        choices = by_topology[topology]
        episode = choices[int(rng.integers(len(choices)))]
        indices = torch.as_tensor(rng.integers(len(episode["state"]), size=settings["batch_size"]), device=device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction = model(episode["state"][indices], episode["action"][indices], episode["graph"])
        loss = physical_graph_loss(prediction, episode["target"][indices], episode["graph"]["physical_mask"], model.delta_scale)
        one_step_loss = loss
        rollout_loss = loss.new_zeros(())
        if rollout_weight:
            count = len(episode["state"]) - rollout_horizon + 1
            if count < 1:
                raise ValueError("Fitting episode shorter than rollout horizon")
            starts = torch.as_tensor(rng.integers(count, size=settings["batch_size"]), device=device)
            rollout_loss = rollout_training_loss(model, episode, starts, rollout_horizon)
            loss = loss + rollout_weight * rollout_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite fitting loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
        batch_loss = float(loss.detach())
        interval_losses.append(batch_loss)
        interval_one_step.append(float(one_step_loss.detach()))
        interval_rollout.append(float(rollout_loss.detach()))
        interval_topologies.setdefault(topology, []).append(batch_loss)
        if update % int(settings["evaluate_every"]) == 0 or update == int(settings["updates"]):
            measured = (validation_progress_loss(model, valid, rollout_horizon)
                        if rollout_weight else validation_loss(model, valid))
            row = dict(update=update, train_loss=batch_loss, validation_loss=measured,
                       train_loss_mean=float(np.mean(interval_losses)),
                       one_step_loss_mean=float(np.mean(interval_one_step)),
                       rollout_loss_mean=float(np.mean(interval_rollout)),
                       training_batches_averaged=len(interval_losses),
                       train_loss_by_topology={name: dict(mean=float(np.mean(values)), batches=len(values))
                                               for name, values in interval_topologies.items()},
                       wall_seconds=time.perf_counter() - started)
            interval_losses.clear(); interval_one_step.clear(); interval_rollout.clear(); interval_topologies.clear()
            history.append(row); print(json.dumps(row), flush=True)
            if measured < best:
                best, stale, best_update = measured, 0, update
                torch.save(dict(model=model.state_dict(), settings=settings, statistics=stats,
                                update=update, validation_loss=best), output / "best-model.pt")
            else:
                stale += 1
            (output / "fit-history.json").write_text(json.dumps(history, indent=2))
            if settings.get("early_stopping", True) and update >= settings["minimum_updates"] and stale >= settings["patience"]:
                break
    torch.save(dict(model=model.state_dict(), settings=settings, statistics=stats,
                    update=update, validation_loss=measured), output / "last-model.pt")
    saved = torch.load(output / "best-model.pt", map_location=device, weights_only=False)
    model.load_state_dict(saved["model"]); model.eval()
    result = dict(updates_completed=update, best_update=best_update, best_validation_loss=best,
                  wall_seconds=time.perf_counter() - started, device=device,
                  checkpoint_sha256=hashlib.sha256((output / "best-model.pt").read_bytes()).hexdigest(),
                  training_seed=settings["seed"], statistics=stats, timing=timing, history=history)
    (output / "fit-result.json").write_text(json.dumps(result, indent=2))
    return model, result


def load_model(path: Path, device: str = "cpu") -> ObservableDynamicsGNN:
    """Load an explicitly trusted local pilot checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ObservableDynamicsGNN(checkpoint["settings"]["width"], checkpoint["settings"]["layers"]).to(device)
    model.load_state_dict(checkpoint["model"]); model.eval()
    return model


@torch.no_grad()
def evaluate_recursive(model: ObservableDynamicsGNN, episodes: list[dict],
                       horizons: list[int], device: str = "cpu") -> dict:
    """Compare identical action windows without injecting future true state."""
    model.eval()
    rows = []
    for episode in episodes:
        graph = tensor_graph(episode, device)
        state = torch.as_tensor(episode["state"], dtype=torch.float32, device=device)
        next_state = torch.as_tensor(episode["next_state"], dtype=torch.float32, device=device)
        action = torch.as_tensor(episode["action"], dtype=torch.float32, device=device)
        progress = torch.as_tensor(episode["progress"], dtype=torch.float32, device=device)
        dt = torch.as_tensor(episode["dt"], dtype=torch.float32, device=device).reshape(-1)
        physical = graph["physical_mask"]
        diagonal = float(episode["initial_diagonal"])
        nominal_dt = float(np.median(np.asarray(episode["dt"])))
        for horizon in horizons:
            count = len(state) - horizon + 1
            if count <= 0:
                continue
            starts = torch.arange(count, device=device)
            # Every within-episode window is retained, including terminal-ending windows.
            truth = next_state[starts + horizon - 1]
            true_progress = torch.stack([progress[index:index + count] for index in range(horizon)]).sum(0)
            for name in ("gnn", "persistence", "constant_velocity"):
                predicted = state[starts].clone()
                movement = torch.zeros((count, 2), device=device)
                for offset in range(horizon):
                    if name == "gnn":
                        predicted, step_progress = model.step(predicted, action[starts + offset], graph)
                    elif name == "constant_velocity":
                        predicted = predicted.clone()
                        predicted[:, :, :3] += predicted[:, :, 3:] * dt[starts + offset, None, None]
                        step_progress = predicted[:, physical, :2].mean(1)
                        predicted[:, :, :2] -= step_progress[:, None]
                    else:
                        step_progress = torch.zeros_like(movement)
                    movement += step_progress
                finite = torch.isfinite(predicted).all(dim=(1, 2)) & torch.isfinite(movement).all(1)
                pos_error = predicted[:, physical, :3] - truth[:, physical, :3]
                pos_error[:, :, :2] += (movement - true_progress)[:, None]
                vel_error = predicted[:, physical, 3:] - truth[:, physical, 3:]
                progress_error = (movement[:, 0] - true_progress[:, 0]).abs() / diagonal
                # Fixed finite-error convention was predeclared in fit-protocol.json.
                progress_error = torch.where(finite, progress_error, torch.full_like(progress_error, 100.0))
                pos_mse = torch.where(finite, (pos_error ** 2).mean((1, 2)), torch.full_like(progress_error, (100 * diagonal) ** 2))
                vel_mse = torch.where(finite, (vel_error ** 2).mean((1, 2)), torch.full_like(progress_error, 10000.0))
                shortened = torch.stack([dt[index:index + count] < nominal_dt - 1e-7 for index in range(horizon)]).any(0)
                timing_metrics = dict(short_dt_windows=int(shortened.sum()),
                    progress_mae_short_dt=float(progress_error[shortened].mean()) if bool(shortened.any()) else None,
                    progress_mae_full_dt=float(progress_error[~shortened].mean()) if bool((~shortened).any()) else None)
                rigidity_metrics = {}
                if all(key in episode for key in ("structural_edges", "initial_rigidity_reference", "next_rigidity")):
                    physical_positions = predicted[:, graph["physical_order_index"], :3].cpu().numpy()
                    predicted_rigidity = np.asarray([normalized_rigidity(positions,
                        np.asarray(episode["structural_edges"], dtype=np.int64),
                        float(episode["initial_rigidity_reference"]),
                        axis_indices=tuple(episode.get("axis_indices", [0, 1, 2])),
                        weighted=bool(episode.get("wcrm", False))) for positions in physical_positions])
                    true_rigidity = np.asarray(episode["next_rigidity"])[np.arange(count) + horizon - 1]
                    rig_error = np.abs(predicted_rigidity - true_rigidity)
                    rig_error = np.where(np.isfinite(rig_error), rig_error, 100.)
                    threshold = float(episode.get("critical_eig_threshold", episode.get("source_config", {}).get("critical_eig_threshold", .03)))
                    true_collapse = np.asarray(episode["terminated"])[np.arange(count) + horizon - 1].astype(bool)
                    predicted_collapse = (~np.isfinite(predicted_rigidity)) | (predicted_rigidity < threshold)
                    rigidity_metrics = dict(rigidity_mae=float(rig_error.mean()),
                        endpoint_true_collapse_windows=int(true_collapse.sum()),
                        endpoint_collapse_false_negatives=int((true_collapse & ~predicted_collapse).sum()),
                        endpoint_collapse_false_positives=int((~true_collapse & predicted_collapse).sum()),
                        collapse_threshold=threshold)
                rows.append(dict(topology=episode["topology"], role=episode["role"],
                    episode_id=str(episode.get("episode_id")), amplitude=episode.get("amplitude"), horizon=horizon, predictor=name,
                    windows=count, nonfinite_windows=int((~finite).sum()),
                    progress_mae=float(progress_error.mean()), position_mse=float(pos_mse.mean()),
                    velocity_mse=float(vel_mse.mean()),
                    true_progress_abs_mean=float((true_progress[:, 0].abs() / diagonal).mean()), **timing_metrics, **rigidity_metrics))
    groups: dict[tuple, list] = {}
    for row in rows:
        groups.setdefault((row["topology"], row["horizon"], row["predictor"]), []).append(row)
    summary = []
    for (topology, horizon, predictor), values in groups.items():
        weights = np.array([r["windows"] for r in values]); weights = weights / weights.sum()
        average = lambda key: float(sum(w * r[key] for w, r in zip(weights, values)))
        rigidity_metrics = {}
        if "rigidity_mae" in values[0]:
            rigidity_metrics = dict(rigidity_mae=average("rigidity_mae"),
                endpoint_true_collapse_windows=sum(r["endpoint_true_collapse_windows"] for r in values),
                endpoint_collapse_false_negatives=sum(r["endpoint_collapse_false_negatives"] for r in values),
                endpoint_collapse_false_positives=sum(r["endpoint_collapse_false_positives"] for r in values))
        short_count = sum(r['short_dt_windows'] for r in values)
        full_count = sum(r['windows'] - r['short_dt_windows'] for r in values)
        timing_metrics = dict(progress_mae_short_dt=sum(r['progress_mae_short_dt'] * r['short_dt_windows']
            for r in values if r['progress_mae_short_dt'] is not None) / short_count if short_count else None,
            progress_mae_full_dt=sum(r['progress_mae_full_dt'] * (r['windows'] - r['short_dt_windows'])
            for r in values if r['progress_mae_full_dt'] is not None) / full_count if full_count else None)
        summary.append(dict(topology=topology, horizon=horizon, predictor=predictor,
            episodes=len(values), windows=sum(r["windows"] for r in values),
            nonfinite_windows=sum(r["nonfinite_windows"] for r in values),
            short_dt_windows=sum(r["short_dt_windows"] for r in values),
            progress_mae=average("progress_mae"), position_rmse=np.sqrt(average("position_mse")),
            velocity_rmse=np.sqrt(average("velocity_mse")), true_progress_abs_mean=average("true_progress_abs_mean"), **timing_metrics, **rigidity_metrics))
    return dict(summary=summary, episodes=rows,
                metric="physical-node-mean forward progress MAE / initial bbox diagonal; world-aligned physical-node position RMSE (m); velocity RMSE (m/s)",
                independence="windows overlap; episode and topology counts are reported, not independent sample claims",
                timing="GNN predicts fixed commanded-control-step deltas. CV uses actual logged dt (oracle near early terminal steps); short-dt and full-duration windows reported separately. Neither establishes near-collapse timing prediction.",
                collapse="terminal-ending windows retained; endpoint geometry rigidity and collapse errors reported; intermediate imagined collapse does not truncate prediction metrics; candidate selection requires separate ranking evaluation",
                claim_boundary="Single training seed source screening; no held-out transfer or learned control result")
