"""Top-level package for the Facade Segmentation project."""

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:  # pragma: no cover - convenience import
    """Lazy-load submodules to keep the top-level import lightweight.

    This mirrors the previous behaviour where subpackages were accessed
    via ``lposs.<module>`` without ``lposs`` being an installed package.
    ``import lposs.helpers`` will continue to work, but optional modules
    are only imported on demand.
    """

    try:
        return import_module(f"lposs.{name}")
    except ModuleNotFoundError as exc:  # pragma: no cover - passthrough
        raise AttributeError(f"module 'lposs' has no attribute {name!r}") from exc


__all__ = [
    "helpers",
    "metrics",
    "models",
    "mmseg",
    "segmentation",
    "tools",
]
