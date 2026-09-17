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
    tensors = [
        (path, tensor)
        for path, tensor in _iter_tensors(value, label)
        if tensor.is_floating_point() or tensor.is_complex()
    ]
    if not tensors:
        return

    finite_flags = [torch.isfinite(tensor).all() for _, tensor in tensors]
    aggregate_device = next(
        (flag.device for flag in finite_flags if flag.device.type != "cpu"),
        finite_flags[0].device,
    )
    all_finite = torch.stack(
        [flag.to(device=aggregate_device) for flag in finite_flags]
    ).all()
    if bool(all_finite.item()):
        return

    # The healthy path above performs one device-to-host synchronization. Only
    # inspect tensors individually after the aggregate reports a failure.
    for path, tensor in tensors:
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
    require_finite(f"{label}.state", tuple(optimizer.state.values()))
