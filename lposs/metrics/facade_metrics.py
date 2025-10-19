"""Utility metrics tailored for facade damage segmentation."""

from typing import Dict

import torch
import torch.nn.functional as F

from .boundary_iou import boundary_iou


class FacadeMetricLogger:
    """Aggregate IoU and boundary IoU over a validation epoch."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.intersection = torch.zeros(self.num_classes)
        self.union = torch.zeros(self.num_classes)
        self.pred_area = torch.zeros(self.num_classes)
        self.gt_area = torch.zeros(self.num_classes)
        self.boundary_inter = torch.zeros(self.num_classes)
        self.boundary_union = torch.zeros(self.num_classes)

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

    def compute(self) -> Dict[str, float]:
        miou = (self.intersection / self.union.clamp(min=1)).mean().item()
        boundary = (self.boundary_inter / self.boundary_union.clamp(min=1)).mean().item()
        return {"mIoU": miou, "boundaryIoU": boundary}
