"""Utility metrics tailored for facade damage segmentation."""

from collections.abc import Sequence
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .boundary_iou import boundary_iou


class FacadeMetricLogger:
    """Aggregate IoU and boundary IoU over a validation epoch."""

    DAMAGE_CLASSES = {
        "CRACK",
        "SPALLING",
        "DELAMINATION",
        "MISSING_ELEMENT",
        "EFFLORESCENCE",
        "CORROSION",
    }
    DAMAGE_PLUS_EXTRA = {"WATER_STAIN"}

    def __init__(
        self,
        classes: Union[int, Sequence[str], None] = None,
        *,
        num_classes: Optional[int] = None,
        class_names: Optional[Sequence[str]] = None,
        ignore_index: int = 255,
    ) -> None:
        if classes is None and num_classes is None and class_names is None:
            raise ValueError("Either class names or num_classes must be provided")

        resolved_names: Optional[Sequence[str]] = None
        resolved_num: Optional[int] = None

        if isinstance(classes, int):
            resolved_num = classes
        elif classes is not None:
            if isinstance(classes, str):
                raise TypeError(
                    "classes must be a sequence of class names or an integer count"
                )
            resolved_names = classes

        if class_names is not None:
            resolved_names = class_names

        if num_classes is not None:
            resolved_num = num_classes if resolved_num is None else resolved_num

        if resolved_names is not None and resolved_num is not None:
            if resolved_num != len(resolved_names):
                raise ValueError(
                    "num_classes does not match length of provided class names"
                )

        if resolved_names is not None:
            self.display_class_names = list(resolved_names)
            self.class_names = [self._normalise_name(name) for name in resolved_names]
            self.num_classes = len(self.class_names)
        elif resolved_num is not None:
            if resolved_num <= 0:
                raise ValueError("num_classes must be a positive integer")
            self.display_class_names = None
            self.class_names = None
            self.num_classes = resolved_num
        else:
            raise ValueError("Unable to resolve class definitions for metric logger")

        self.ignore_index = ignore_index
        self.damage_indices = self._collect_indices(self.DAMAGE_CLASSES)
        self.damage_plus_indices = self._collect_indices(
            self.DAMAGE_CLASSES | self.DAMAGE_PLUS_EXTRA
        )
        self.reset()

    @staticmethod
    def _normalise_name(name: str) -> str:
        return name.strip().upper()

    def _collect_indices(self, names: Iterable[str]) -> Tuple[int, ...]:
        if not self.class_names:
            return ()

        lookup = {}
        for idx, label in enumerate(self.class_names):
            for part in label.split(";"):
                lookup[part.strip()] = idx
        indices = sorted({lookup[label] for label in names if label in lookup})
        return tuple(indices)

    def reset(self) -> None:
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
        self.pred_area = torch.zeros(self.num_classes)
        self.gt_area = torch.zeros(self.num_classes)
        self.boundary_inter = torch.zeros(self.num_classes)
        self.boundary_union = torch.zeros(self.num_classes)
        self.confusion = torch.zeros((self.num_classes, self.num_classes))
        self.damage_intersection = 0.0
        self.damage_union = 0.0
        self.damage_plus_intersection = 0.0
        self.damage_plus_union = 0.0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probs = F.softmax(logits, dim=1)
        if probs.shape[-2:] != target.shape[-2:]:
            probs = F.interpolate(
                probs,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.clamp(probs, min=1e-8)
            normaliser = probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
            probs = probs / normaliser
        preds = probs.argmax(dim=1)
        for pred_map, gt_map in zip(preds, target):
            valid_mask = gt_map != self.ignore_index
            pred_valid = pred_map[valid_mask]
            gt_valid = gt_map[valid_mask]
            if valid_mask.any():
                self._update_confusion(pred_valid, gt_valid)
                self._update_damage_metrics(pred_valid, gt_valid)
            for cls in range(self.num_classes):
                pred_cls = pred_valid == cls
                gt_cls = gt_valid == cls
                inter = torch.logical_and(pred_cls, gt_cls).sum().item()
                union = torch.logical_or(pred_cls, gt_cls).sum().item()
                self.intersection[cls] += inter
                self.union[cls] += union
                self.pred_area[cls] += pred_cls.sum().item()
                self.gt_area[cls] += gt_cls.sum().item()
            b_inter, b_union, _, _ = boundary_iou(
                gt_map.cpu().numpy(),
                pred_map.cpu().numpy(),
                self.num_classes,
                ignore_index=self.ignore_index,
            )
            self.boundary_inter += b_inter.float()
            self.boundary_union += b_union.float()

    def _update_damage_metrics(
        self, pred_valid: torch.Tensor, gt_valid: torch.Tensor
    ) -> None:
        if self.damage_indices:
            damage_pred = self._mask_for_indices(pred_valid, self.damage_indices)
            damage_gt = self._mask_for_indices(gt_valid, self.damage_indices)
            self.damage_intersection += (
                torch.logical_and(damage_pred, damage_gt).sum().item()
            )
            self.damage_union += (
                torch.logical_or(damage_pred, damage_gt).sum().item()
            )

        if self.damage_plus_indices:
            damage_plus_pred = self._mask_for_indices(
                pred_valid, self.damage_plus_indices
            )
            damage_plus_gt = self._mask_for_indices(gt_valid, self.damage_plus_indices)
            self.damage_plus_intersection += (
                torch.logical_and(damage_plus_pred, damage_plus_gt).sum().item()
            )
            self.damage_plus_union += (
                torch.logical_or(damage_plus_pred, damage_plus_gt).sum().item()
            )

    @staticmethod
    def _mask_for_indices(tensor: torch.Tensor, indices: Tuple[int, ...]) -> torch.Tensor:
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        for idx in indices:
            mask |= tensor == idx
        return mask

    def _update_confusion(self, pred_valid: torch.Tensor, gt_valid: torch.Tensor) -> None:
        if self.confusion.numel() == 0:
            return
        flat_pred = pred_valid.to(torch.int64)
        flat_gt = gt_valid.to(torch.int64)
        combined = flat_gt * self.num_classes + flat_pred
        counts = torch.bincount(
            combined,
            minlength=self.num_classes * self.num_classes,
        ).float()
        self.confusion += counts.view(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, float]:
        miou = (self.intersection / self.union.clamp(min=1)).mean().item()
        boundary = (self.boundary_inter / self.boundary_union.clamp(min=1)).mean().item()
        metrics = {"mIoU": miou, "boundaryIoU": boundary}
        if self.damage_indices:
            damage_iou = (
                self.damage_intersection / max(self.damage_union, 1.0)
                if self.damage_union
                else 0.0
            )
            metrics["damageIoU"] = damage_iou
        if self.damage_plus_indices:
            damage_plus_iou = (
                self.damage_plus_intersection / max(self.damage_plus_union, 1.0)
                if self.damage_plus_union
                else 0.0
            )
            metrics["damagePlusIoU"] = damage_plus_iou
        return metrics

    def confusion_matrix(self) -> torch.Tensor:
        """Return a copy of the accumulated confusion matrix."""

        return self.confusion.clone()
