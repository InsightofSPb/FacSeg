"""Facade-specific LPOSS adapter for facade damage research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

import torch
from torch import Tensor, nn

from models.builder import MODELS
from models.lposs.lposs import LPOSS


@dataclass
class FacadeOptimizationConfig:
    """Hyper-parameters for fine-tuning the CLIP decode head."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    ignore_index: int = 255
    class_weight: Optional[Iterable[float]] = None

    def build_loss(self) -> nn.Module:
        """Create a cross-entropy criterion using the stored parameters."""

        if self.class_weight is None:
            weight_tensor = None
        else:
            weight_tensor = torch.tensor(list(self.class_weight), dtype=torch.float32)
        return nn.NLLLoss(weight=weight_tensor, ignore_index=self.ignore_index)


@MODELS.register_module()
class FacadeLPOSS(LPOSS):
    """LPOSS variant that exposes a simple fine-tuning interface."""

    def __init__(
        self,
        clip_backbone: str,
        class_names,
        vit_arch: str = "vit_base",
        vit_patch_size: int = 16,
        enc_type_feats: str = "k",
        trainable_decode_head: bool = True,
        unfreeze_clip_backbone: bool = False,
        optimisation: Optional[Dict] = None,
    ) -> None:
        super().__init__(
            clip_backbone=clip_backbone,
            class_names=class_names,
            vit_arch=vit_arch,
            vit_patch_size=vit_patch_size,
            enc_type_feats=enc_type_feats,
        )

        if unfreeze_clip_backbone:
            for param in self.clip_backbone.parameters():
                param.requires_grad = True

        if not trainable_decode_head:
            for param in self.clip_backbone.decode_head.parameters():
                param.requires_grad = False

        self.optimisation = FacadeOptimizationConfig(**(optimisation or {}))
        self._loss = self.optimisation.build_loss()

    @torch.no_grad()
    def forward_train(self, inputs: Tensor) -> Tensor:
        """Return CLIP decode head logits for supervision."""

        logits = self.clip_backbone(inputs)
        return logits

    def get_trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield parameters that require gradients."""

        for param in self.parameters():
            if param.requires_grad:
                yield param

    def configure_optimiser(self) -> torch.optim.Optimizer:
        """Build an optimiser for the fine-tuning parameters."""

        return torch.optim.AdamW(
            self.get_trainable_parameters(),
            lr=self.optimisation.lr,
            weight_decay=self.optimisation.weight_decay,
        )

    def training_step(self, images: Tensor, masks: Tensor) -> Dict[str, Tensor]:
        """Compute loss for a batch of facade annotations."""

        logits = self.clip_backbone(images)
        log_probs = torch.log(torch.clamp(logits, min=1e-6))
        loss = self._loss(log_probs, masks.long())
        return {"loss": loss, "logits": logits}

    # Inference uses the base class ``forward`` implementation.
