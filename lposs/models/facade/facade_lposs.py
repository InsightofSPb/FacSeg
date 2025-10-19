"""Facade-specific LPOSS adapter for facade damage research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
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

        if trainable_decode_head:
            for param in self.clip_backbone.decode_head.parameters():
                param.requires_grad = True
        else:
            for param in self.clip_backbone.decode_head.parameters():
                param.requires_grad = False

        self.optimisation = FacadeOptimizationConfig(**(optimisation or {}))
        self._loss = self.optimisation.build_loss()
    
    @staticmethod
    def _ensure_training_shapes(
        images: Tensor,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Ensure tensors include an explicit batch dimension.

        MMCV datasets configured with ``samples_per_gpu=1`` often yield tensors
        without a leading batch axis. The MaskCLIP backbone, however, expects
        ``(N, C, H, W)`` image tensors. This helper adds the missing dimension
        for images (and optionally masks) while validating other inputs.
        """

        if images.dim() == 3:
            images = images.unsqueeze(0)
        elif images.dim() != 4:
            raise ValueError(
                "Expected image tensor with 3 or 4 dimensions, "
                f"but received shape {tuple(images.shape)}",
            )

        if masks is None:
            return images, None

        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        elif masks.dim() == 4 and masks.size(1) == 1:
            masks = masks.squeeze(1)
        elif masks.dim() != 3:
            raise ValueError(
                "Expected mask tensor with 2, 3, or 4 dimensions, "
                f"but received shape {tuple(masks.shape)}",
            )

        if masks.shape[0] != images.shape[0]:
            raise ValueError(
                "Image and mask batch dimensions do not match: "
                f"{images.shape[0]} vs {masks.shape[0]}",
            )

        return images, masks
    
    @torch.no_grad()
    def forward_train(self, inputs: Tensor) -> Tensor:
        """Return CLIP decode head logits for supervision."""
        
        inputs, _ = self._ensure_training_shapes(inputs)
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

    @staticmethod
    def _resize_logits(logits: Tensor, target_hw: Tuple[int, int]) -> Tensor:
        """Match prediction spatial size to supervision masks.

        The decode head operates at the CLIP patch resolution (e.g. 32x32 for
        512x512 inputs with a 16 px patch). Upsample the logits to the target
        mask resolution and renormalise the class probabilities per pixel so
        they remain a valid distribution after interpolation.
        """

        if logits.shape[-2:] == target_hw:
            return logits

        resized = F.interpolate(
            logits,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        resized = torch.clamp(resized, min=1e-8)
        normaliser = resized.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return resized / normaliser

    def training_step(self, images: Tensor, masks: Tensor) -> Dict[str, Tensor]:
        """Compute loss for a batch of facade annotations."""

        images, masks = self._ensure_training_shapes(images, masks)
        logits = self.clip_backbone(images)
        logits = self._resize_logits(logits, masks.shape[-2:])
        log_probs = torch.log(torch.clamp(logits, min=1e-6))
        loss = self._loss(log_probs, masks.long())
        return {"loss": loss, "logits": logits}

    # Inference uses the base class ``forward`` implementation.
