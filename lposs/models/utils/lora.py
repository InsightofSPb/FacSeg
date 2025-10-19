"""LoRA adapter injection utilities for CLIP-based modules."""

from __future__ import annotations

import fnmatch
import math
from typing import Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


class _LoRABase(nn.Module):
    """Base class shared by LoRA wrappers."""

    def __init__(self, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be a positive integer")
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _freeze_module(module: nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False


class LoRAInjectedLinear(_LoRABase):
    """Wrap an ``nn.Linear`` layer with learnable low-rank adapters."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__(rank, alpha, dropout)
        self.base_layer = base_layer
        self._freeze_module(self.base_layer)
        self.lora_down = nn.Parameter(torch.zeros(rank, base_layer.in_features))
        self.lora_up = nn.Parameter(torch.zeros(base_layer.out_features, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        base = self.base_layer(x)
        lora = F.linear(self.dropout(x), self.lora_down.t(), bias=None)
        lora = F.linear(lora, self.lora_up, bias=None)
        return base + lora * self.scaling


class LoRAInjectedConv2d(_LoRABase):
    """LoRA wrapper specialised for 1x1 convolutional projections."""

    def __init__(self, base_layer: nn.Conv2d, rank: int, alpha: float, dropout: float) -> None:
        if base_layer.kernel_size != (1, 1) or base_layer.stride != (1, 1) or base_layer.padding != (0, 0):
            raise ValueError("LoRAInjectedConv2d currently supports only 1x1 convolutions without stride/padding")
        super().__init__(rank, alpha, dropout)
        self.base_layer = base_layer
        self._freeze_module(self.base_layer)
        in_channels = base_layer.in_channels
        out_channels = base_layer.out_channels
        self.lora_down = nn.Parameter(torch.zeros(rank, in_channels))
        self.lora_up = nn.Parameter(torch.zeros(out_channels, rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_down, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        base = self.base_layer(x)
        b, c, h, w = x.shape
        flattened = x.permute(0, 2, 3, 1).reshape(-1, c)
        adapted = F.linear(self.dropout(flattened), self.lora_down.t(), bias=None)
        adapted = F.linear(adapted, self.lora_up, bias=None)
        adapted = adapted.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return base + adapted * self.scaling


def inject_lora_adapters(
    module: nn.Module,
    *,
    target_modules: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> List[str]:
    """Attach LoRA adapters to the selected submodules of ``module``.

    Args:
        module: Root module whose children may receive LoRA adapters.
        target_modules: Iterable of fnmatch-compatible patterns describing
            module paths to wrap. Patterns are evaluated against the dotted
            module hierarchy as returned by ``named_modules``.
        rank: Rank of the low-rank adaptation matrices.
        alpha: LoRA scaling factor.
        dropout: Dropout rate applied to the adapter inputs.

    Returns:
        A list with the fully-qualified module names that were wrapped.
    """

    patterns = list(target_modules)
    if not patterns:
        return []

    wrapped: List[str] = []

    def _matches(name: str) -> bool:
        return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)

    def _recurse(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if _matches(full_name):
                if isinstance(child, nn.Linear):
                    setattr(parent, child_name, LoRAInjectedLinear(child, rank, alpha, dropout))
                    wrapped.append(full_name)
                elif isinstance(child, nn.Conv2d):
                    setattr(parent, child_name, LoRAInjectedConv2d(child, rank, alpha, dropout))
                    wrapped.append(full_name)
                else:
                    # Unsupported module type for LoRA injection.
                    continue
            else:
                _recurse(child, full_name)

    _recurse(module)
    return wrapped
