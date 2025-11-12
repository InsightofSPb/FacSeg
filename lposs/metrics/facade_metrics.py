"""Utility metrics tailored for facade damage segmentation."""

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .boundary_iou import boundary_iou


class FacadeMetricLogger:
    """Aggregate IoU and boundary IoU over a validation epoch."""

    DAMAGE_CLASSES = {
        "DAMAGE",
        "CRACK",
        "SPALLING",
        "DELAMINATION",
        "MISSING_ELEMENT",
        "EFFLORESCENCE",
        "CORROSION",
    }
    DAMAGE_PLUS_EXTRA = DAMAGE_CLASSES | {"WATER_STAIN"}
    DEFAULT_CLASS_ORDER = (
        "background",
        "DAMAGE",
        "WATER_STAIN",
        "ORNAMENT_INTACT",
        "REPAIRS",
        "TEXT_OR_IMAGES",
    )

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
            if self.num_classes == len(self.DEFAULT_CLASS_ORDER):
                self.display_class_names = list(self.DEFAULT_CLASS_ORDER)
                self.class_names = [
                    self._normalise_name(name) for name in self.DEFAULT_CLASS_ORDER
                ]
        else:
            raise ValueError("Unable to resolve class definitions for metric logger")

        self.ignore_index = ignore_index
        self.damage_indices = self._collect_indices(self.DAMAGE_CLASSES)
        self.damage_plus_indices = self._collect_indices(
            self.DAMAGE_CLASSES | self.DAMAGE_PLUS_EXTRA
        )
        self._dtype = torch.float32
        self._device = torch.device("cpu")
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
        self._initialise_state(self._device, self._dtype)
        self.damage_intersection = 0.0
        self.damage_union = 0.0
        self.damage_plus_intersection = 0.0
        self.damage_plus_union = 0.0
        self.damage_hits = 0.0
        self.damage_gt_total = 0.0
        self.damage_pred_total = 0.0
        self.damage_mislabel = 0.0

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        self._ensure_state_device(logits.device, logits.dtype)
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
            self.boundary_inter += b_inter.to(
                self.boundary_inter.device, dtype=self.boundary_inter.dtype
            )
            self.boundary_union += b_union.to(
                self.boundary_union.device, dtype=self.boundary_union.dtype
            )

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
            hits = torch.logical_and(damage_pred, damage_gt)
            self.damage_hits += hits.sum().item()
            self.damage_gt_total += damage_gt.sum().item()
            self.damage_pred_total += damage_pred.sum().item()
            self.damage_mislabel += torch.logical_and(
                hits, pred_valid != gt_valid
            ).sum().item()

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

    def _initialise_state(self, device: torch.device, dtype: torch.dtype) -> None:
        self._device = torch.device(device)
        self._dtype = dtype
        self.intersection = torch.zeros(
            self.num_classes, device=self._device, dtype=self._dtype
        )
        self.union = torch.zeros(
            self.num_classes, device=self._device, dtype=self._dtype
        )
        self.pred_area = torch.zeros(
            self.num_classes, device=self._device, dtype=self._dtype
        )
        self.gt_area = torch.zeros(
            self.num_classes, device=self._device, dtype=self._dtype
        )
        self.boundary_inter = torch.zeros(
            self.num_classes, device=self._device, dtype=self._dtype
        )
        self.boundary_union = torch.zeros(
            self.num_classes, device=self._device, dtype=self._dtype
        )
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes),
            device=self._device,
            dtype=self._dtype,
        )

    def _ensure_state_device(self, device: torch.device, _: torch.dtype) -> None:
        resolved_device = torch.device(device)
        if resolved_device != self._device:
            self.intersection = self.intersection.to(resolved_device)
            self.union = self.union.to(resolved_device)
            self.pred_area = self.pred_area.to(resolved_device)
            self.gt_area = self.gt_area.to(resolved_device)
            self.boundary_inter = self.boundary_inter.to(resolved_device)
            self.boundary_union = self.boundary_union.to(resolved_device)
            self.confusion = self.confusion.to(resolved_device)
            self._device = resolved_device

    def compute(self) -> Dict[str, float]:
        unions = self.union
        valid = unions > 0
        ious = torch.empty_like(unions)
        ious.fill_(float("nan"))
        if valid.any():
            ious[valid] = self.intersection[valid] / unions[valid]
            miou = ious[valid].mean().item()
        else:
            miou = float("nan")

        boundary_unions = self.boundary_union
        boundary_valid = boundary_unions > 0
        boundary_ious = torch.empty_like(boundary_unions)
        boundary_ious.fill_(float("nan"))
        if boundary_valid.any():
            boundary_ious[boundary_valid] = (
                self.boundary_inter[boundary_valid] / boundary_unions[boundary_valid]
            )
            boundary = boundary_ious[boundary_valid].mean().item()
        else:
            boundary = float("nan")
        precision = torch.where(
            self.pred_area > 0,
            self.intersection / self.pred_area.clamp(min=1),
            torch.zeros_like(self.intersection),
        )
        recall = torch.where(
            self.gt_area > 0,
            self.intersection / self.gt_area.clamp(min=1),
            torch.zeros_like(self.intersection),
        )
        denom = precision + recall
        eps = torch.finfo(precision.dtype).eps
        f1 = torch.where(
            denom > 0,
            (2.0 * precision * recall) / denom.clamp(min=eps),
            torch.zeros_like(denom),
        )
        macro_f1 = f1.mean().item()
        metrics = {
            "mIoU": miou,
            "boundaryIoU": boundary,
            "macroF1": macro_f1,
            "f1": macro_f1,
            "damageIoU": 0.0,
            "damagePlusIoU": 0.0,
            "damageMIoU": 0.0,
            "damagePlusMIoU": 0.0,
            "damageDetectionRecall": 0.0,
            "damageDetectionPrecision": 0.0,
            "damageDetectionF1": 0.0,
            "damageMislabelRate": 0.0,
        }
        if self.damage_indices:
            damage_values = [
                ious[idx]
                for idx in self.damage_indices
                if 0 <= idx < ious.numel() and valid[idx]
            ]
            if damage_values:
                metrics["damageMIoU"] = torch.stack(damage_values).mean().item()
            if self.damage_gt_total:
                metrics["damageDetectionRecall"] = self.damage_hits / max(
                    self.damage_gt_total, 1.0
                )
                metrics["damageMislabelRate"] = self.damage_mislabel / max(
                    self.damage_gt_total, 1.0
                )
            if self.damage_pred_total:
                metrics["damageDetectionPrecision"] = self.damage_hits / max(
                    self.damage_pred_total, 1.0
                )
            precision = metrics["damageDetectionPrecision"]
            recall = metrics["damageDetectionRecall"]
            denom = precision + recall
            if denom > 0:
                metrics["damageDetectionF1"] = (2 * precision * recall) / denom
        if self.damage_plus_indices:
            damage_plus_values = [
                ious[idx]
                for idx in self.damage_plus_indices
                if 0 <= idx < ious.numel() and valid[idx]
            ]
            if damage_plus_values:
                metrics["damagePlusMIoU"] = torch.stack(damage_plus_values).mean().item()
        if self.damage_indices and self.damage_union:
            metrics["damageIoU"] = self.damage_intersection / max(self.damage_union, 1.0)
        if self.damage_plus_indices and self.damage_plus_union:
            metrics["damagePlusIoU"] = self.damage_plus_intersection / max(
                self.damage_plus_union, 1.0
            )
        return metrics

    def per_class_iou(self) -> Dict[str, float]:
        unions = self.union
        valid = unions > 0
        ious = torch.empty_like(unions)
        ious.fill_(float("nan"))
        if valid.any():
            ious[valid] = self.intersection[valid] / unions[valid]
        iou_list = ious.tolist()
        if self.display_class_names is not None:
            keys = self.display_class_names
        else:
            keys = [str(idx) for idx in range(self.num_classes)]
        return {key: float(value) for key, value in zip(keys, iou_list)}

    def confusion_matrix(self) -> torch.Tensor:
        """Return a CPU copy of the accumulated confusion matrix."""

        return self.confusion.to("cpu").clone()

    def export_confusion(
        self,
        path: Union[str, Path],
        *,
        class_names: Optional[Sequence[str]] = None,
    ) -> None:
        matrix = self.confusion_matrix().tolist()
        if class_names is None:
            if self.display_class_names is not None:
                class_names = self.display_class_names
            else:
                class_names = [str(idx) for idx in range(self.num_classes)]
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = ["class"] + list(class_names)
        with output_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for name, row in zip(class_names, matrix):
                writer.writerow([name] + [f"{value:.0f}" for value in row])
