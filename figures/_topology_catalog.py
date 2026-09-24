"""Shared topology/fold catalog derived from the cross-validation config.

Single source of truth for "which named topologies exist, and which
leave-one-group-out fold each belongs to," parsed straight from
``config/cross_validation/node_count_loso.yaml`` so the analysis scripts in
this directory never drift out of sync with the actual cross-validation
definition.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

DEFAULT_CV_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "cross_validation"
    / "node_count_loso.yaml"
)


def _load_cross_validation(path: Path) -> dict:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return raw["cross_validation"]


def groups(path: Path = DEFAULT_CV_CONFIG) -> dict[str, list[str]]:
    """Return the ``{group_name: [topology, ...]}`` mapping."""
    return _load_cross_validation(path)["groups"]


def final_test_topologies(path: Path = DEFAULT_CV_CONFIG) -> list[str]:
    """Topologies reserved for final-test and excluded from every CV fold."""
    return list(_load_cross_validation(path).get("final_test") or [])


def all_topologies(path: Path = DEFAULT_CV_CONFIG) -> list[str]:
    """Every named topology: CV-fold groups plus the final-test set."""
    cv = _load_cross_validation(path)
    names = [name for topologies in cv["groups"].values() for name in topologies]
    names.extend(cv.get("final_test") or [])
    return names


def fold_membership(path: Path = DEFAULT_CV_CONFIG) -> dict[str, dict[str, list[str]]]:
    """Per leave-one-group-out fold: which topologies train, which are held out.

    One fold per group in ``groups``; the fold named after a group holds that
    group out and trains on every other group. ``final_test`` topologies are
    excluded from every fold, matching the cross-validation launcher.
    """
    group_map = groups(path)
    folds = {}
    for held_out_group, held_out_topologies in group_map.items():
        train_topologies = [
            name
            for group_name, topologies in group_map.items()
            if group_name != held_out_group
            for name in topologies
        ]
        folds[held_out_group] = {
            "held_out": list(held_out_topologies),
            "train": train_topologies,
        }
    return folds
