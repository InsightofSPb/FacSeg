"""Facade-specific LPOSS adapter for facade damage research."""

from __future__ import annotations

import fnmatch
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR

from lposs.models.builder import MODELS
from lposs.models.lposs.lposs import LPOSS
from lposs.models.utils import inject_lora_adapters


@dataclass
class SchedulerConfig:
    """Configuration for cosine learning-rate decay with optional warmup."""

    type: str = "cosine"
    min_lr: float = 0.0
    warmup_epochs: int = 0
    t_max: Optional[int] = None

    def build(self, optimiser: torch.optim.Optimizer, base_lr: float, total_epochs: int) -> LambdaLR:
        if total_epochs <= 0:
            raise ValueError("total_epochs must be a positive integer for cosine schedule")
        if self.type.lower() != "cosine":
            raise ValueError(f"Unsupported scheduler type '{self.type}'")

        warmup = max(0, int(self.warmup_epochs))
        t_max = self.t_max if self.t_max is not None else total_epochs
        t_max = max(1, t_max)
        min_lr_ratio = self.min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(epoch: int) -> float:
            if warmup > 0 and epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            progress = 0.0
            if t_max > warmup:
                progress = (epoch - warmup) / max(1, t_max - warmup)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        return LambdaLR(optimiser, lr_lambda=lr_lambda)


@dataclass
class FacadeOptimizationConfig:
    """Hyper-parameters for fine-tuning the CLIP decode head."""

    lr: float = 1e-4
    weight_decay: float = 0.0
    ignore_index: int = 255
    class_weight: Optional[Iterable[float]] = None
    scheduler: Optional[SchedulerConfig] = None

    def __post_init__(self) -> None:
        if self.class_weight is not None:
            self.class_weight = list(self.class_weight)
        if self.scheduler is not None and not isinstance(self.scheduler, SchedulerConfig):
            self.scheduler = SchedulerConfig(**self.scheduler)  # type: ignore[arg-type]

    def build_loss(self) -> nn.Module:
        """Create a cross-entropy criterion using the stored parameters."""

        if self.class_weight is None:
            weight_tensor = None
        else:
            weight_tensor = torch.tensor(list(self.class_weight), dtype=torch.float32)
        return nn.NLLLoss(weight=weight_tensor, ignore_index=self.ignore_index)

    def build_scheduler(
        self,
        optimiser: torch.optim.Optimizer,
        *,
        total_epochs: Optional[int],
    ) -> Optional[LambdaLR]:
        if self.scheduler is None or total_epochs is None:
            return None
        return self.scheduler.build(optimiser, self.lr, total_epochs)


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
        decode_head_dropout: float = 0.0,
        gradual_unfreeze: Optional[Dict[str, Any]] = None,
        lora: Optional[Dict[str, Any]] = None,
        prompt_templates: Optional[Iterable[str]] = None,
        prompt_descriptions: Optional[Dict[str, Iterable[str]]] = None,
    ) -> None:
        super().__init__(
            clip_backbone=clip_backbone,
            class_names=class_names,
            vit_arch=vit_arch,
            vit_patch_size=vit_patch_size,
            enc_type_feats=enc_type_feats,
        )

        self._gradual_unfreeze_stages = self._parse_gradual_unfreeze(gradual_unfreeze)
        self._applied_unfreeze_epochs: set[int] = set()
        self._lora_wrapped_modules: List[str] = []

        if lora and lora.get("enabled", True):
            self._lora_wrapped_modules = inject_lora_adapters(
                self.clip_backbone,
                target_modules=self._normalise_lora_targets(lora.get("target_modules", ())),
                rank=int(lora.get("rank", 4)),
                alpha=float(lora.get("alpha", 8.0)),
                dropout=float(lora.get("dropout", 0.0)),
            )

        if self._gradual_unfreeze_stages:
            self.update_backbone_trainable(0)
        elif unfreeze_clip_backbone:
            self._unfreeze_clip_backbone_layers()

        if trainable_decode_head:
            for param in self.clip_backbone.decode_head.parameters():
                param.requires_grad = True
        else:
            for param in self.clip_backbone.decode_head.parameters():
                param.requires_grad = False

        if hasattr(self.clip_backbone.decode_head, "set_dropout"):
            self.clip_backbone.decode_head.set_dropout(decode_head_dropout)
        elif decode_head_dropout > 0:
            raise AttributeError(
                "Decode head does not expose set_dropout; update MaskClipHead to support dropout"
            )

        if hasattr(self.clip_backbone.decode_head, "configure_prompts"):
            self.clip_backbone.decode_head.configure_prompts(
                prompt_descriptions=prompt_descriptions,
                prompt_templates=prompt_templates,
            )

        self.optimisation = FacadeOptimizationConfig(**(optimisation or {}))
        self._loss = self.optimisation.build_loss()

    def _unfreeze_clip_backbone_layers(self) -> None:
        """Enable gradients for a subset of CLIP backbone layers."""

        trainable_patterns = (
            "backbone.visual.transformer.resblocks.10",
            "backbone.visual.transformer.resblocks.11",
            "backbone.visual.ln_post",
            "backbone.visual.proj",
        )

        for name, param in self.clip_backbone.named_parameters():
            param.requires_grad = any(pattern in name for pattern in trainable_patterns)

    @staticmethod
    def _normalise_lora_targets(targets: Iterable[str]) -> List[str]:
        return [str(target) for target in targets]

    @staticmethod
    def _to_plain_data(config: Optional[Any]) -> Optional[Any]:
        if config is None:
            return None
        if isinstance(config, dict):
            return {k: FacadeLPOSS._to_plain_data(v) for k, v in config.items()}
        if isinstance(config, (list, tuple)):
            return [FacadeLPOSS._to_plain_data(v) for v in config]
        return config

    def _parse_gradual_unfreeze(self, config: Optional[Any]) -> List[Dict[str, Any]]:
        plain = self._to_plain_data(config)
        if not plain:
            return []
        stages = []
        for stage in plain.get("stages", []):
            epoch = int(stage.get("epoch", 0))
            patterns_raw = stage.get("patterns", [])
            if isinstance(patterns_raw, (str, bytes)):
                patterns = [str(patterns_raw)]
            else:
                patterns = [str(p) for p in patterns_raw]
            stages.append({"epoch": epoch, "patterns": patterns})
        stages.sort(key=lambda item: item["epoch"])
        return stages

    def _set_parameter_requires_grad(
        self, patterns: Sequence[str], requires_grad: bool
    ) -> int:
        if not patterns:
            return 0
        toggled = 0
        for name, param in self.clip_backbone.named_parameters():
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                if param.requires_grad != requires_grad:
                    param.requires_grad = requires_grad
                    toggled += 1
        return toggled
    
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

    def configure_optimiser(
        self,
        total_epochs: Optional[int] = None,
    ) -> Tuple[torch.optim.Optimizer, Optional[LambdaLR]]:
        """Build an optimiser (and optional scheduler) for fine-tuning."""

        optimiser = torch.optim.AdamW(
            self.get_trainable_parameters(),
            lr=self.optimisation.lr,
            weight_decay=self.optimisation.weight_decay,
        )
        scheduler = self.optimisation.build_scheduler(optimiser, total_epochs=total_epochs)
        return optimiser, scheduler

    def update_backbone_trainable(self, epoch: int, logger: Optional[Any] = None) -> None:
        """Apply gradual unfreezing schedule up to the specified epoch."""

        if not self._gradual_unfreeze_stages:
            return

        for stage in self._gradual_unfreeze_stages:
            stage_epoch = int(stage["epoch"])
            if stage_epoch in self._applied_unfreeze_epochs:
                continue
            if epoch < stage_epoch:
                continue
            patterns: Sequence[str] = stage.get("patterns", [])
            toggled = self._set_parameter_requires_grad(patterns, True)
            self._applied_unfreeze_epochs.add(stage_epoch)
            if logger is not None and toggled:
                logger.info(
                    "Gradual unfreezing activated for %d parameter tensors at epoch %d (patterns=%s)",
                    toggled,
                    epoch,
                    patterns,
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
