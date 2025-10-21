"""Utility helpers shared across the FacSeg project."""

from __future__ import annotations

import torch


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    """Fill ``tensor`` with values drawn from a truncated normal distribution.

    This mirrors the helper provided by facebookresearch/dino so that when the
    Torch Hub loader imports ``trunc_normal_`` from ``utils`` the call resolves
    to an implementation compatible with DINO's expectations.
    """

    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


__all__ = ["trunc_normal_"]
