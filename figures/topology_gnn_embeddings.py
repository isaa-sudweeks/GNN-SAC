"""Extract per-fold pooled GNN-policy embeddings for every topology.

Leave-one-group-out analysis needs one embedding space *per fold*: the k-NN
distance from a held-out topology to the fold's training topologies is only
meaningful (and leak-free) if the model producing the embeddings never
trained on the held-out topology. So every fold is embedded with its own
checkpoint(s), listed in a manifest (default ``figures/cv_base_checkpoints.yaml``)::

    folds:
      node_4:
        - seed: 2
          wandb_run: i-suds/CV/qb4c0r24
          checkpoint: figures/checkpoints/CV-BASE_seed2_node4heldout_latest.agent.pt
          training_config: null   # optional; else downloaded from wandb_run

A fold with no available checkpoint (``checkpoint: null`` or a missing file)
is skipped with a warning -- it is never filled in from another fold.

For each checkpoint the model is rebuilt from its own training-run config
(W&B ``config.yaml``, cached in ``figures/checkpoints/configs/``), and the
run's ``cross_validation.held_out_group`` is checked against the manifest
fold. Each topology (all CV groups plus final_test) is then embedded by
resetting its environment ``--n-resets`` times (domain randomization off),
pooling the actor GNN's pre-action-head node embeddings over the physical
nodes, and averaging over resets.

Output: one ``{output_dir}/{fold}.npz`` per fold with keys

* ``{topology}``                 -- (d,) mean over seeds and resets (use this),
* ``seed{n}/{topology}``         -- (d,) per-seed mean over resets,
* ``resets/seed{n}/{topology}``  -- (n_resets, d) raw per-reset embeddings,
* ``_meta``                      -- JSON string with provenance and reset-noise stats.

Downstream code should read these with :func:`load_fold_embeddings`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import warnings

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch_geometric.data import Batch

ROOT = Path(__file__).resolve().parents[1]
SAC_ROOT = ROOT / "sac"
FIGURES_ROOT = ROOT / "figures"
for path in (ROOT, SAC_ROOT, FIGURES_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _topology_catalog

from common.parser import parse_cfg
from common.seed import set_seed
from common.graph_transforms import graph_feature_flags, physical_node_mask, prepare_graph
from env import make_env
from gnn_infer import _make_agent, load_agent_checkpoint, resolve_checkpoint

DEFAULT_MANIFEST = FIGURES_ROOT / "cv_base_checkpoints.yaml"
DEFAULT_OUTPUT_DIR = FIGURES_ROOT / "topology_gnn_embeddings"
DEFAULT_CONFIG_CACHE = FIGURES_ROOT / "checkpoints" / "configs"
META_KEY = "_meta"


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckpointEntry:
    """One trained checkpoint for one fold."""

    fold: str
    seed: int
    wandb_run: str | None
    checkpoint: Path | None
    training_config: Path | None = None

    @property
    def available(self) -> bool:
        return self.checkpoint is not None and self.checkpoint.exists()


def _resolve_path(value, base: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base / path


def load_manifest(path: Path, *, base_dir: Path = ROOT) -> dict[str, list[CheckpointEntry]]:
    """Parse a checkpoint manifest into ``{fold: [CheckpointEntry, ...]}``.

    Relative ``checkpoint`` / ``training_config`` paths resolve against
    ``base_dir`` (the repo root by default). Folds with an empty entry list
    are kept (as ``[]``) so the caller can report them as skipped.
    """
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
    folds_raw = raw.get("folds")
    if not isinstance(folds_raw, dict):
        raise ValueError(f"{path}: expected a top-level 'folds' mapping.")

    manifest: dict[str, list[CheckpointEntry]] = {}
    for fold, entries in folds_raw.items():
        entries = entries or []
        if not isinstance(entries, list):
            raise ValueError(f"{path}: fold {fold!r} must map to a list of entries.")
        parsed = []
        seen_seeds = set()
        for entry in entries:
            if not isinstance(entry, dict) or "seed" not in entry:
                raise ValueError(f"{path}: fold {fold!r} entry {entry!r} needs at least a 'seed'.")
            seed = int(entry["seed"])
            if seed in seen_seeds:
                raise ValueError(f"{path}: fold {fold!r} lists seed {seed} more than once.")
            seen_seeds.add(seed)
            parsed.append(
                CheckpointEntry(
                    fold=str(fold),
                    seed=seed,
                    wandb_run=entry.get("wandb_run"),
                    checkpoint=_resolve_path(entry.get("checkpoint"), base_dir),
                    training_config=_resolve_path(entry.get("training_config"), base_dir),
                )
            )
        manifest[str(fold)] = parsed
    return manifest


def available_entries(
    manifest: dict[str, list[CheckpointEntry]],
) -> dict[str, list[CheckpointEntry]]:
    """Keep only entries whose checkpoint exists locally; warn about the rest.

    Folds left with no usable checkpoint are dropped entirely (with a warning)
    rather than borrowed from another fold, which would leak held-out
    topologies into the "training distribution".
    """
    usable: dict[str, list[CheckpointEntry]] = {}
    for fold, entries in manifest.items():
        present = []
        for entry in entries:
            if entry.available:
                present.append(entry)
            else:
                reason = "checkpoint: null" if entry.checkpoint is None else f"missing file {entry.checkpoint}"
                warnings.warn(
                    f"Fold {fold} seed {entry.seed} ({entry.wandb_run}): {reason}; skipping this seed.",
                    stacklevel=2,
                )
        if present:
            usable[fold] = present
        else:
            warnings.warn(
                f"Fold {fold}: no checkpoint available -- skipping the fold (no embeddings written).",
                stacklevel=2,
            )
    return usable


# --------------------------------------------------------------------------- #
# Training configs
# --------------------------------------------------------------------------- #


def _load_training_config(path: Path) -> dict:
    """Flatten a W&B-exported config.yaml (each top-level key wrapped as
    ``{value: ...}``) into a plain dict of resolved hyperparameters.

    An already-flat config (no ``{value: ...}`` wrapping) is returned as-is.
    """
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    wrapped = {
        key: entry["value"]
        for key, entry in raw.items()
        if isinstance(entry, dict) and "value" in entry and not key.startswith("_")
    }
    if wrapped:
        return wrapped
    return {key: value for key, value in raw.items() if not str(key).startswith("_")}


def fetch_training_config(entry: CheckpointEntry, cache_dir: Path = DEFAULT_CONFIG_CACHE) -> Path:
    """Return a local W&B-format config.yaml for ``entry``'s training run.

    Uses ``entry.training_config`` if given, else ``{cache_dir}/{run_id}.yaml``,
    downloading the run's ``config.yaml`` file into that cache on first use.
    """
    if entry.training_config is not None:
        if not entry.training_config.exists():
            raise FileNotFoundError(f"training_config not found: {entry.training_config}")
        return entry.training_config
    if not entry.wandb_run:
        raise ValueError(
            f"Fold {entry.fold} seed {entry.seed}: need either training_config or wandb_run "
            "to rebuild the checkpoint's training architecture."
        )
    run_id = entry.wandb_run.rstrip("/").split("/")[-1]
    cached = cache_dir / f"{run_id}.yaml"
    if cached.exists():
        return cached

    import wandb

    run = wandb.Api().run(entry.wandb_run)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        downloaded = run.file("config.yaml").download(root=tmp_dir, replace=True)
        shutil.move(downloaded.name, cached)
    return cached


def validate_training_config(entry: CheckpointEntry, config: dict) -> None:
    """Raise if the run's CV fold disagrees with the manifest fold or catalog."""
    cv = config.get("cross_validation") or {}
    held_out = cv.get("held_out_group")
    if held_out != entry.fold:
        raise ValueError(
            f"Manifest fold {entry.fold} seed {entry.seed} points at run {entry.wandb_run}, "
            f"whose cross_validation.held_out_group is {held_out!r}. Refusing to embed: "
            "this would leak held-out topologies."
        )
    run_seed = config.get("seed")
    if run_seed is not None and int(run_seed) != entry.seed:
        raise ValueError(
            f"Manifest fold {entry.fold} lists seed {entry.seed} but run {entry.wandb_run} used seed {run_seed}."
        )
    expected = _topology_catalog.fold_membership().get(entry.fold)
    run_train = cv.get("training_topologies")
    if expected is not None and run_train is not None and sorted(run_train) != sorted(expected["train"]):
        raise ValueError(
            f"Run {entry.wandb_run} trained on {sorted(run_train)}, but the catalog's fold "
            f"{entry.fold} trains on {sorted(expected['train'])}."
        )
    if entry.checkpoint is not None:
        match = re.search(r"node(\d+)heldout", entry.checkpoint.name)
        if match and f"node_{match.group(1)}" != entry.fold:
            raise ValueError(
                f"Checkpoint file {entry.checkpoint.name} looks like fold node_{match.group(1)}, "
                f"but the manifest lists it under {entry.fold}."
            )


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #


def _build_cfg(
    checkpoint: str, topology: str, device: str, seed: int, *, training_config: Path | None = None
):
    # work_dir is set explicitly because parse_cfg() otherwise falls back to
    # HydraConfig.get(), which is only populated inside a real @hydra.main
    # run -- not for a bare compose() call like this one.
    work_dir = ROOT / "logs" / "topology_gnn_embeddings" / topology

    if training_config is not None:
        # A checkpoint's architecture (layer widths, virtual-node/edge-role
        # flags, etc.) is whatever its own training run used, not whatever
        # inference/gnn.yaml happens to default to -- so rebuild the actual
        # training cfg from its W&B config.yaml and only override the fields
        # needed to run one topology, once, on CPU.
        base = _load_training_config(training_config)
        overrides = {
            **base,
            "model": checkpoint,
            "truss_topology": topology,
            "truss_topologies": None,
            "topologies": None,
            "multitask": False,
            "num_envs": 1,
            "mujoco_backend": "mujoco",
            "eval_extra_topologies": None,
            "cross_validation": {"enabled": False, "name": None, "groups": {}, "final_test": [], "held_out_group": None},
            "device": device,
            "seed": seed,
            "domain_randomization": False,
            "work_dir": str(work_dir),
            "enable_wandb": False,
            "save_agent": False,
            "save_csv": False,
            "save_video": False,
        }
        cfg = OmegaConf.create(overrides)
        return parse_cfg(cfg)

    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        cfg = compose(
            config_name="inference/gnn",
            overrides=[
                f"model={checkpoint}",
                f"truss_topology={topology}",
                f"device={device}",
                f"seed={seed}",
                "domain_randomization=false",
                f"work_dir={work_dir}",
            ],
        )
    return parse_cfg(cfg)


@torch.no_grad()
def embed_topology_resets(
    checkpoint: str,
    topology: str,
    *,
    device: str = "cpu",
    seed: int = 0,
    n_resets: int = 5,
    training_config: Path | None = None,
) -> np.ndarray:
    """Return ``(n_resets, d)`` pooled pre-action-head actor embeddings.

    The RNGs are seeded once with ``seed``; the ``n_resets`` successive
    ``env.reset()`` calls then draw different initial-state perturbations.
    """
    if n_resets < 1:
        raise ValueError("n_resets must be >= 1")
    cfg = _build_cfg(checkpoint, topology, device, seed, training_config=training_config)
    set_seed(cfg.seed)
    env = make_env(cfg)
    try:
        agent = _make_agent(cfg)
        # GNNSAC.load() uses a strict load_state_dict, so any architecture
        # mismatch between checkpoint and training config raises here.
        load_agent_checkpoint(agent, resolve_checkpoint(str(cfg.model), cfg))
        agent.model.eval()

        # Mirrors GNNSAC.act_batch(): the raw env observation must be run
        # through prepare_graph() (role/virtual-node augmentation) before it
        # matches the shape the actor's GNN was built for.
        use_virtual_node = bool(getattr(cfg, "use_virtual_node", False))
        feature_flags = graph_feature_flags(cfg)
        pooled = []
        for _ in range(n_resets):
            obs = env.reset()
            prepared = prepare_graph(obs, use_virtual_node=use_virtual_node, **feature_flags)
            obs_batch = Batch.from_data_list([prepared]).to(cfg.device)
            node_embeddings = agent.model._pi(
                obs_batch.x, obs_batch.edge_index, getattr(obs_batch, "edge_attr", None)
            )
            mask = physical_node_mask(obs_batch)
            pooled.append(node_embeddings[mask].mean(dim=0).cpu().numpy())
        embeddings = np.stack(pooled)
    finally:
        env.close()
    if not np.all(np.isfinite(embeddings)):
        raise FloatingPointError(f"Non-finite embedding for {topology} from {checkpoint}.")
    return embeddings


def embed_topology(
    checkpoint: str,
    topology: str,
    *,
    device: str = "cpu",
    seed: int = 0,
    n_resets: int = 1,
    training_config: Path | None = None,
) -> np.ndarray:
    """Return the ``(d,)`` reset-averaged pooled actor embedding for one topology."""
    return embed_topology_resets(
        checkpoint, topology, device=device, seed=seed, n_resets=n_resets,
        training_config=training_config,
    ).mean(axis=0)


def reset_noise_stats(per_reset: dict[str, np.ndarray]) -> dict[str, float]:
    """Compare within-topology reset spread to between-topology distances.

    ``per_reset`` maps topology -> ``(n_resets, d)``. Reset spread is the RMS
    Euclidean distance of each reset embedding from its topology centroid;
    between-topology distances are pairwise Euclidean distances of centroids.
    """
    names = list(per_reset)
    centroids = np.stack([per_reset[name].mean(axis=0) for name in names])
    spreads = np.array([
        float(np.sqrt(np.mean(np.sum((per_reset[name] - centroids[i]) ** 2, axis=1))))
        for i, name in enumerate(names)
    ])
    diffs = centroids[:, None, :] - centroids[None, :, :]
    pairwise = np.sqrt(np.sum(diffs ** 2, axis=-1))[np.triu_indices(len(names), k=1)]
    stats = {
        "reset_spread_mean": float(spreads.mean()),
        "reset_spread_max": float(spreads.max()),
        "reset_spread_max_topology": names[int(spreads.argmax())],
    }
    if pairwise.size:
        stats.update({
            "between_topology_min": float(pairwise.min()),
            "between_topology_median": float(np.median(pairwise)),
            "spread_max_over_between_min": float(spreads.max() / pairwise.min()) if pairwise.min() > 0 else float("inf"),
        })
    return stats


def embed_fold(
    entries: list[CheckpointEntry],
    topologies: list[str],
    *,
    device: str = "cpu",
    seed: int = 0,
    n_resets: int = 5,
    config_cache_dir: Path = DEFAULT_CONFIG_CACHE,
) -> tuple[dict[str, np.ndarray], dict]:
    """Embed every topology with each of one fold's checkpoints.

    Returns the npz payload (see module docstring) and a metadata dict.
    """
    if not entries:
        raise ValueError("embed_fold needs at least one checkpoint entry.")
    folds = {entry.fold for entry in entries}
    if len(folds) != 1:
        raise ValueError(f"embed_fold got entries from several folds: {sorted(folds)}")

    arrays: dict[str, np.ndarray] = {}
    per_seed_means: dict[int, dict[str, np.ndarray]] = {}
    meta = {"fold": entries[0].fold, "n_resets": n_resets, "reset_seed": seed, "seeds": [], "topologies": topologies}
    for entry in entries:
        config_path = fetch_training_config(entry, config_cache_dir)
        validate_training_config(entry, _load_training_config(config_path))
        per_reset = {
            topology: embed_topology_resets(
                str(entry.checkpoint), topology, device=device, seed=seed,
                n_resets=n_resets, training_config=config_path,
            )
            for topology in topologies
        }
        per_seed_means[entry.seed] = {name: emb.mean(axis=0) for name, emb in per_reset.items()}
        for name, emb in per_reset.items():
            arrays[f"resets/seed{entry.seed}/{name}"] = emb
            arrays[f"seed{entry.seed}/{name}"] = per_seed_means[entry.seed][name]
        meta["seeds"].append({
            "seed": entry.seed,
            "wandb_run": entry.wandb_run,
            "checkpoint": str(entry.checkpoint),
            "training_config": str(config_path),
            "reset_noise": reset_noise_stats(per_reset),
        })
    for name in topologies:
        arrays[name] = np.mean([means[name] for means in per_seed_means.values()], axis=0)
    return arrays, meta


def save_fold_embeddings(path: Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, **{META_KEY: np.array(json.dumps(meta))})


def load_fold_embeddings(
    directory: Path,
    topologies: list[str],
    *,
    folds: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Load ``{fold: (n_topologies, d)}`` seed/reset-averaged embeddings.

    Rows follow ``topologies`` order. Reads every ``{directory}/{fold}.npz``
    (or just ``folds`` if given, raising ``FileNotFoundError`` if one is
    absent). Raises ``KeyError`` if a fold file lacks any requested topology.
    Each fold's embeddings come from that fold's own model, so distances are
    only comparable *within* a fold, never across folds.
    """
    directory = Path(directory)
    if folds is None:
        paths = sorted(directory.glob("*.npz"))
    else:
        paths = [directory / f"{fold}.npz" for fold in folds]
        missing_files = [str(p) for p in paths if not p.exists()]
        if missing_files:
            raise FileNotFoundError(f"Missing fold embedding files: {missing_files}")

    result: dict[str, np.ndarray] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in topologies if name not in data.files]
            if missing:
                raise KeyError(f"{path} is missing embeddings for topologies: {missing}")
            result[path.stem] = np.stack([data[name] for name in topologies])
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config-cache-dir", type=Path, default=DEFAULT_CONFIG_CACHE)
    parser.add_argument("--folds", nargs="*", default=None, help="Subset of manifest folds to embed.")
    parser.add_argument("--n-resets", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed before the reset sequence.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topologies = _topology_catalog.all_topologies()
    manifest = load_manifest(args.manifest)
    if args.folds:
        unknown = sorted(set(args.folds) - set(manifest))
        if unknown:
            raise SystemExit(f"Folds not in manifest: {unknown}")
        manifest = {fold: manifest[fold] for fold in args.folds}

    usable = available_entries(manifest)
    skipped = sorted(set(manifest) - set(usable))
    for fold, entries in usable.items():
        arrays, meta = embed_fold(
            entries, topologies, device=args.device, seed=args.seed,
            n_resets=args.n_resets, config_cache_dir=args.config_cache_dir,
        )
        output = args.output_dir / f"{fold}.npz"
        save_fold_embeddings(output, arrays, meta)
        dim = arrays[topologies[0]].shape[0]
        print(f"[{fold}] wrote {len(topologies)} topologies x d={dim} "
              f"from seeds {[e.seed for e in entries]} to {output}")
        for seed_meta in meta["seeds"]:
            noise = seed_meta["reset_noise"]
            print(
                f"    seed {seed_meta['seed']}: reset spread mean={noise['reset_spread_mean']:.4g} "
                f"max={noise['reset_spread_max']:.4g} ({noise['reset_spread_max_topology']}); "
                f"between-topology min={noise['between_topology_min']:.4g} "
                f"median={noise['between_topology_median']:.4g}; "
                f"max-spread/min-between={noise['spread_max_over_between_min']:.3g}"
            )
    if skipped:
        print(f"Skipped folds with no available checkpoint: {skipped}")


if __name__ == "__main__":
    main()
