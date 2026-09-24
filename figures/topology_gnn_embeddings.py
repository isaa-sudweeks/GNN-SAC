"""Extract a pooled GNN-policy embedding per topology from a trained checkpoint.

Builds the same Hydra config a normal ``python sac/gnn_infer.py model=...``
invocation would, one topology at a time, resets the environment once (no
rollout, no domain randomization), and pools the actor GNN's pre-action-head
node embeddings over every physical node. The checkpoint reference is
whatever the caller supplies -- a local path, a ``wandb-artifact://`` ref, or
a W&B run/artifact URL -- exactly what ``sac/gnn_infer.py``'s
``resolve_checkpoint`` already accepts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

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

DEFAULT_OUTPUT = Path(__file__).with_name("topology_gnn_embeddings.npz")


def _load_training_config(path: Path) -> dict:
    """Flatten a W&B-exported config.yaml (each top-level key wrapped as
    ``{value: ...}``) into a plain dict of resolved hyperparameters."""
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    return {
        key: entry["value"]
        for key, entry in raw.items()
        if isinstance(entry, dict) and "value" in entry and not key.startswith("_")
    }


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
def embed_topology(
    checkpoint: str,
    topology: str,
    *,
    device: str = "cpu",
    seed: int = 0,
    training_config: Path | None = None,
) -> np.ndarray:
    """Return the mean-pooled, pre-action-head actor embedding for one topology."""
    cfg = _build_cfg(checkpoint, topology, device, seed, training_config=training_config)
    set_seed(cfg.seed)
    env = make_env(cfg)
    try:
        agent = _make_agent(cfg)
        load_agent_checkpoint(agent, resolve_checkpoint(str(cfg.model), cfg))
        agent.model.eval()

        obs = env.reset()
        # Mirrors GNNSAC.act_batch(): the raw env observation must be run
        # through prepare_graph() (role/virtual-node augmentation) before it
        # matches the shape the actor's GNN was built for.
        use_virtual_node = bool(getattr(cfg, "use_virtual_node", False))
        feature_flags = graph_feature_flags(cfg)
        prepared = prepare_graph(obs, use_virtual_node=use_virtual_node, **feature_flags)
        obs_batch = Batch.from_data_list([prepared]).to(cfg.device)

        node_embeddings = agent.model._pi(
            obs_batch.x, obs_batch.edge_index, getattr(obs_batch, "edge_attr", None)
        )
        mask = physical_node_mask(obs_batch)
        pooled = node_embeddings[mask].mean(dim=0)
        return pooled.cpu().numpy()
    finally:
        env.close()


def embed_all_topologies(
    checkpoint: str,
    topologies: list[str],
    *,
    device: str = "cpu",
    seed: int = 0,
    training_config: Path | None = None,
) -> dict[str, np.ndarray]:
    return {
        topology: embed_topology(
            checkpoint, topology, device=device, seed=seed, training_config=training_config
        )
        for topology in topologies
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", required=True,
        help="Local path, wandb-artifact:// ref, or W&B run/artifact URL.",
    )
    parser.add_argument(
        "--training-config", type=Path, default=None,
        help=(
            "W&B config.yaml from the run that produced --checkpoint, so the model "
            "is rebuilt with its actual architecture instead of inference/gnn.yaml's "
            "defaults. Required unless the checkpoint happens to match those defaults."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topologies = _topology_catalog.all_topologies()
    embeddings = embed_all_topologies(
        args.checkpoint, topologies, device=args.device, seed=args.seed,
        training_config=args.training_config,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **embeddings)
    print(f"Wrote {len(embeddings)} topology embeddings to {args.output}")


if __name__ == "__main__":
    main()
