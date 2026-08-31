"""Leave-one-configuration-group-out cross-validation helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict


_SAFE_GROUP_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _plain(value: Any) -> Any:
    if isinstance(value, (DictConfig,)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _topology_list(value: Any, field: str) -> list[str]:
    value = _plain(value)
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"cross_validation.{field} must be a list of topology identifiers.")
    topologies = [str(item).strip() for item in value]
    if any(not topology for topology in topologies):
        raise ValueError(f"cross_validation.{field} cannot contain empty topology identifiers.")
    return topologies


def validate_cross_validation_spec(
    value: Any,
    *,
    require_held_out: bool = False,
) -> dict[str, Any] | None:
    """Validate and normalize a cross-validation definition."""
    value = _plain(value)
    if value in (None, "null"):
        return None
    if not isinstance(value, Mapping):
        raise ValueError("cross_validation must be a mapping.")
    if not bool(value.get("enabled", False)):
        return None

    name = str(value.get("name") or "").strip()
    if not name or not _SAFE_GROUP_NAME.fullmatch(name):
        raise ValueError(
            "cross_validation.name must contain only letters, numbers, underscores, or hyphens."
        )

    raw_groups = _plain(value.get("groups", {}))
    if not isinstance(raw_groups, Mapping) or len(raw_groups) < 2:
        raise ValueError("cross_validation.groups must contain at least two groups.")

    groups: dict[str, list[str]] = {}
    owners: dict[str, str] = {}
    for raw_group_name, raw_topologies in raw_groups.items():
        group_name = str(raw_group_name).strip()
        if not group_name or not _SAFE_GROUP_NAME.fullmatch(group_name):
            raise ValueError(
                "Cross-validation group names must contain only letters, numbers, "
                "underscores, or hyphens."
            )
        topologies = _topology_list(raw_topologies, f"groups.{group_name}")
        if not topologies:
            raise ValueError(f"cross_validation.groups.{group_name} cannot be empty.")
        for topology in topologies:
            previous_owner = owners.get(topology)
            if previous_owner is not None:
                raise ValueError(
                    f"Topology '{topology}' appears in both '{previous_owner}' and "
                    f"'{group_name}'. Cross-validation groups must be disjoint."
                )
            owners[topology] = group_name
        groups[group_name] = topologies

    final_test = _topology_list(value.get("final_test", []), "final_test")
    if len(set(final_test)) != len(final_test):
        raise ValueError("cross_validation.final_test cannot contain duplicate topologies.")
    for topology in final_test:
        previous_owner = owners.get(topology)
        if previous_owner is not None:
            raise ValueError(
                f"Final-test topology '{topology}' also appears in development group "
                f"'{previous_owner}'."
            )

    held_out = value.get("held_out_group")
    if held_out in (None, "", "null"):
        held_out = None
    else:
        held_out = str(held_out)
        if held_out not in groups:
            raise ValueError(
                f"Unknown cross_validation.held_out_group '{held_out}'; expected one of "
                f"{list(groups)}."
            )
    if require_held_out and held_out is None:
        raise ValueError(
            "cross_validation.held_out_group must be selected for a training run. "
            "Use scripts/launch_cross_validation.py to generate all folds."
        )

    return {
        "enabled": True,
        "name": name,
        "groups": groups,
        "final_test": final_test,
        "held_out_group": held_out,
    }


def resolve_cross_validation(cfg: Any) -> Any:
    """Resolve one selected fold into the existing train/evaluation topology fields."""
    spec = validate_cross_validation_spec(
        getattr(cfg, "cross_validation", None),
        require_held_out=True,
    )
    if spec is None:
        return cfg

    if str(getattr(cfg, "eval_backend", "mujoco")).lower() != "mujoco":
        raise ValueError("Cross-validation requires eval_backend=mujoco.")
    if getattr(cfg, "truss_topologies", None) not in (None, "null"):
        raise ValueError(
            "Cross-validation owns truss_topologies; do not override it separately."
        )
    if getattr(cfg, "eval_extra_topologies", None) not in (None, "null"):
        raise ValueError(
            "Cross-validation owns eval_extra_topologies; do not override it separately."
        )

    group_names = list(spec["groups"])
    held_out = spec["held_out_group"]
    training_groups = [name for name in group_names if name != held_out]
    training_topologies = [
        topology
        for group_name in training_groups
        for topology in spec["groups"][group_name]
    ]
    heldout_topologies = list(spec["groups"][held_out])
    if not training_topologies:
        raise ValueError("The selected cross-validation fold has no training topologies.")

    with open_dict(cfg):
        cfg.truss_topologies = training_topologies
        cfg.eval_extra_topologies = heldout_topologies
        with open_dict(cfg.cross_validation):
            cfg.cross_validation.fold_index = group_names.index(held_out)
            cfg.cross_validation.fold_name = f"holdout_{held_out}"
            cfg.cross_validation.training_groups = training_groups
            cfg.cross_validation.training_topologies = training_topologies
            cfg.cross_validation.heldout_topologies = heldout_topologies
    return cfg
