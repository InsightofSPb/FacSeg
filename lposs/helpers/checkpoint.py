"""Utility helpers for saving and loading torch checkpoints safely."""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Tuple

import torch.nn as nn


def unwrap_module(module: nn.Module) -> nn.Module:
    """Return the underlying module, unwrapping common wrappers.

    Handles wrappers introduced by ``torch.compile`` (``_orig_mod``), distributed
    containers (``.module``), and adapters that retain the original layer under an
    ``original_module``/``original_layer`` attribute. The function iteratively
    unwraps until no known wrapper attributes remain.
    """

    current = module
    visited: set[int] = set()
    while True:
        marker = id(current)
        if marker in visited:
            break
        visited.add(marker)

        if hasattr(current, "_orig_mod") and isinstance(current._orig_mod, nn.Module):  # type: ignore[attr-defined]
            current = current._orig_mod  # type: ignore[assignment]
            continue
        if hasattr(current, "module") and isinstance(current.module, nn.Module):
            current = current.module  # type: ignore[assignment]
            continue
        if hasattr(current, "original_module") and isinstance(current.original_module, nn.Module):
            current = current.original_module  # type: ignore[assignment]
            continue
        if hasattr(current, "original_layer") and isinstance(current.original_layer, nn.Module):
            current = current.original_layer  # type: ignore[assignment]
            continue
        break

    return current


def state_dict_from_module(module: nn.Module, *args, **kwargs) -> MutableMapping[str, Any]:
    """Return a state_dict for ``module`` while unwrapping any wrappers."""

    base = unwrap_module(module)
    return base.state_dict(*args, **kwargs)


def load_model_state(
    module: nn.Module,
    state_dict: Mapping[str, Any],
    *args,
    **kwargs,
) -> Tuple[list[str], list[str]]:
    """Load a ``state_dict`` into ``module`` after unwrapping wrappers."""

    base = unwrap_module(module)
    missing, unexpected = base.load_state_dict(state_dict, *args, **kwargs)
    return missing, unexpected
