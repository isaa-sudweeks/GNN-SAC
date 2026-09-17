"""Fail-fast validation helpers for non-finite training state."""

from collections.abc import Mapping

import torch


class NonFiniteTrainingError(RuntimeError):
    """Raised before non-finite data can silently poison a training run."""


def _iter_tensors(value, path):
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_tensors(item, f"{path}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _iter_tensors(item, f"{path}[{index}]")
        return

    items = getattr(value, "items", None)
    if callable(items):
        try:
            entries = items()
        except (AttributeError, TypeError):
            return
        for key, item in entries:
            yield from _iter_tensors(item, f"{path}.{key}")


def require_finite(label, value):
    """Raise with the first offending tensor path when ``value`` is non-finite."""
    for path, tensor in _iter_tensors(value, label):
        if not (tensor.is_floating_point() or tensor.is_complex()):
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all().item()):
            continue
        invalid_count = int((~finite).sum().item())
        raise NonFiniteTrainingError(
            f"{path} contains {invalid_count} non-finite value(s) "
            f"with shape={tuple(tensor.shape)} and dtype={tensor.dtype}."
        )


def require_finite_gradients(label, named_parameters):
    """Validate every materialized gradient in a named parameter collection."""
    gradients = {
        name: parameter.grad
        for name, parameter in named_parameters
        if parameter.grad is not None
    }
    require_finite(label, gradients)


def require_finite_optimizer(label, optimizer):
    """Validate tensor-valued optimizer accumulators before saving or resuming."""
    for index, state in enumerate(optimizer.state.values()):
        require_finite(f"{label}.state[{index}]", state)
