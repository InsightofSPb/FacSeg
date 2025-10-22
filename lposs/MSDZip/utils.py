"""Utility helpers for loading MSDZip checkpoints.

The original MSDZip project stores model checkpoints as nested Python
``dict`` structures.  Depending on which training loop produced the
checkpoint, the actual model weights might live at different nesting
levels (e.g. ``state_dict``, ``model_state`` or directly at the root) and
occasionally contain prefixes such as ``module.`` that come from
``torch.nn.DataParallel``.

The upstream loader assumed that the checkpoint dictionary exposed the
``MixedModel`` parameters directly and therefore rejected checkpoints
produced by the LPOSS fine-tuning pipeline where the weights were stored
under ``root.model_state``.  The helper implemented here performs a
structured search through the checkpoint payload and normalises the key
prefixes so that both layouts are supported.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import torch


LOGGER = logging.getLogger(__name__)


class CheckpointBundle(dict):
    """Dictionary that also exposes keys as attributes."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - convenience wrapper
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - mirrors ``dict`` behaviour
            raise AttributeError(item) from exc



# Keys that should exist in a MixedModel checkpoint.  We only care about
# the weights, therefore the check is intentionally lightweight.
EXPECTED_STATE_DICT_KEYS = {"embedding.weight", "lin.weight"}

# Prefixes that commonly appear when a ``torch.nn.Module`` is wrapped in a
# container such as ``nn.DataParallel``.
COMMON_PREFIXES = (
    "module.",
    "model.",
    "state_dict.",
    "network.",
    "net.",
    "student.",
    "ema.",
)


def _ensure_ordered_dict(state_dict: Mapping[str, Any]) -> OrderedDict:
    """Return an :class:`OrderedDict` copy of *state_dict*.

    ``torch.nn.Module.load_state_dict`` expects an ``OrderedDict``.  The
    helper is careful to preserve the order of keys when possible.
    """

    if isinstance(state_dict, OrderedDict):
        return state_dict
    return OrderedDict(state_dict.items())


def _contains_expected_keys(state_dict: Mapping[str, Any]) -> bool:
    """Check whether the mapping looks like a MixedModel state dict."""

    keys = set(state_dict.keys())
    if EXPECTED_STATE_DICT_KEYS <= keys:
        return True

    # Some training loops prepend a scope (e.g. ``model.embedding``).  In
    # those situations we inspect the suffixes to see if the required keys
    # are present after the first component.
    suffixes = {key.split(".", 1)[-1] for key in keys if "." in key}
    return EXPECTED_STATE_DICT_KEYS <= suffixes


def _strip_known_prefixes(state_dict: Mapping[str, Any]) -> Tuple[Mapping[str, Any], bool, str | None]:
    """Try to drop prefixes that come from wrapper modules.

    Returns a tuple ``(normalised_state_dict, stripped, prefix)`` where
    ``stripped`` indicates whether the returned mapping has modified keys
    and ``prefix`` stores the prefix that was removed.
    """

    for prefix in COMMON_PREFIXES:
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            stripped_state = state_dict.__class__(
                (key[len(prefix) :], value) for key, value in state_dict.items()
            )
            return stripped_state, True, prefix

    # As a last resort we strip the first component from every key if all
    # of them contain a dot.  This mirrors the behaviour of many training
    # loops that nest the model under a named attribute (e.g. ``model``).
    keys = list(state_dict.keys())
    if keys and all("." in key for key in keys):
        suffixes = [key.split(".", 1)[1] for key in keys]
        if len(set(suffixes)) == len(suffixes):
            stripped_state = state_dict.__class__(
                (suffix, value) for suffix, value in zip(suffixes, state_dict.values())
            )
            return stripped_state, True, "<first-component>"

    return state_dict, False, None


def _iter_mappings(obj: Any, prefix: str = "") -> Iterator[Tuple[str, Mapping[str, Any]]]:
    """Yield all mappings contained in *obj* with their hierarchical key."""

    if isinstance(obj, Mapping):
        yield prefix, obj
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            yield from _iter_mappings(value, child_prefix)


def load_checkpoint_state(weights_path: str | Path, device: torch.device | str) -> Tuple[
    OrderedDict,
    str | None,
    bool,
    Dict[str, Any],
    Dict[str, Any],
]:
    """Load a checkpoint from *weights_path* and return the model weights.

    The function is intentionally permissive and searches through the
    checkpoint for a mapping that contains the parameters of the MSDZip
    ``MixedModel``.  When a match is found the keys are normalised so that
    they no longer include DataParallel prefixes.
    """

    weights_path = Path(weights_path).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint '{weights_path}' does not exist")

    checkpoint = torch.load(str(weights_path), map_location=device)
    metadata: Dict[str, Any] = {}
    if isinstance(checkpoint, Mapping):
        maybe_metadata = checkpoint.get("metadata")
        if isinstance(maybe_metadata, Mapping):
            metadata = dict(maybe_metadata)

    available_paths: list[str] = []
    state_dict: OrderedDict | None = None
    source_key: str | None = None
    stripped = False
    stripped_prefix: str | None = None

    # Walk through the checkpoint structure looking for a candidate mapping
    # that resembles a state dict.
    for path, mapping in _iter_mappings(checkpoint):
        # Record the path for debugging purposes.  We mimic the behaviour of
        # the previous implementation that exposed the candidates in the
        # error message.
        if path:
            available_paths.append(path)
        else:
            available_paths.append("<root>")

        if not mapping:
            continue

        candidate = mapping
        local_stripped = False
        local_prefix: str | None = None

        if not _contains_expected_keys(candidate):
            candidate, local_stripped, local_prefix = _strip_known_prefixes(candidate)

        if _contains_expected_keys(candidate):
            state_dict = _ensure_ordered_dict(candidate)
            source_key = path or None
            stripped = local_stripped
            stripped_prefix = local_prefix
            break

    if state_dict is None:
        available = ", ".join(available_paths) if available_paths else "<none>"
        raise ValueError(
            "Checkpoint '{}' does not appear to store MSDZip MixedModel weights "
            "(expected parameters such as 'embedding.weight' and 'lin.weight'). "
            "Available state_dict paths: {}".format(weights_path, available)
        )

    details: Dict[str, Any] = {
        "available_state_dict_paths": available_paths,
    }
    if stripped_prefix is not None:
        details["stripped_prefix"] = stripped_prefix

    return state_dict, source_key, stripped, metadata, details


def prepare_state_from_weights(
    args: Any,
    device: torch.device | str,
    logger: logging.Logger | None = None,
) -> Dict[str, Any]:
    """Load model weights using the :func:`load_checkpoint_state` helper.

    The original project exposed a helper that wrapped
    :func:`load_checkpoint_state` and returned a dictionary containing the
    state dict together with additional metadata.  Only a very small
    portion of that functionality is required for the unit tests, so we
    replicate the parts that are relevant to the compression script.
    """

    logger = logger or LOGGER

    state_dict, source_key, stripped, metadata, details = load_checkpoint_state(
        args.weights, device
    )

    logger.info(
        "Loaded checkpoint '%s' from %s (stripped_prefix=%s)",
        args.weights,
        source_key or "<root>",
        details.get("stripped_prefix"),
    )

    bundle = CheckpointBundle(
        state_dict=state_dict,
        source_key=source_key,
        stripped=stripped,
        metadata=metadata,
        details=details,
    )

    return bundle


__all__ = [
    "CheckpointBundle",
    "load_checkpoint_state",
    "prepare_state_from_weights",
]

