"""Utilities for working with MSDZip checkpoints.

This package only exposes helper functions that are needed by the
`compress.py` entry point that accompanies the LPOSS release.  The
original project stores the helpers under ``MSDZip/utils.py`` so we keep
the same public surface even though the surrounding repository layout is
trimmed down for the kata environment.
"""

from .utils import CheckpointBundle, load_checkpoint_state, prepare_state_from_weights

__all__ = [
    "CheckpointBundle",
    "load_checkpoint_state",
    "prepare_state_from_weights",
]

