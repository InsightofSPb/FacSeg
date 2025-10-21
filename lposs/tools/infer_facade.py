"""Inference utilities for the facade damage baseline."""

from __future__ import annotations
import importlib
import importlib.util
import os
import sys
import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from PIL import Image
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from helpers.logger import get_logger
from metrics.facade_metrics import FacadeMetricLogger
from models import build_model
from tools.train_facade_baseline import (
    CONFIG_DIR,
    _build_datasets,
    _tensor_from_batch,
    _unwrap_data_container,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for the facade baseline model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint produced by train_facade_baseline.py.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="facade_baseline.yaml",
        help="Hydra configuration name located in lposs/configs.",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(CONFIG_DIR),
        help="Directory that contains Hydra configs (defaults to lposs/configs).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store visualisations and predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (cpu, cuda, or cuda:N). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.1,
        help="Fraction of the training set to visualise (between 0 and 1].",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on the number of training samples to process.",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="Process every Nth sample on the validation set (defaults to all samples).",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Additionally save blended overlays alongside side-by-side visuals.",
    )
    return parser.parse_args()


def _load_config(config_name: str, config_dir: Path):
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name)


def _get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _format_param_count(total: int, trainable: int) -> str:
    return f"total={total / 1e6:.2f}M, trainable={trainable / 1e6:.2f}M ({trainable} / {total})"


def _extract_dataset_metadata(dataset) -> Tuple[Iterable[str], Iterable[Iterable[int]]]:
    base = dataset
    while hasattr(base, "dataset"):
        base = base.dataset
    classes = getattr(base, "CLASSES", None)
    palette = getattr(base, "PALETTE", None)
    if classes is None or palette is None:
        raise AttributeError("Dataset does not expose CLASSES/PALETTE metadata")
    return classes, palette


def _extract_meta(batch: Dict) -> Dict:
    if "img_metas" not in batch:
        return {}
    meta = _unwrap_data_container(batch["img_metas"])
    if isinstance(meta, (list, tuple)) and meta:
        meta = _unwrap_data_container(meta[0])
    if not isinstance(meta, dict):
        return {}
    return meta


def _denormalise_image(
    tensor: torch.Tensor,
    mean: Iterable[float],
    std: Iterable[float],
) -> np.ndarray:
    image = tensor.detach().cpu().numpy()
    image = image * np.asarray(std)[:, None, None] + np.asarray(mean)[:, None, None]
    image = np.clip(image.transpose(1, 2, 0), 0, 255).astype(np.uint8)
    return image


def _render_visualisation(
    image: np.ndarray,
    mask: np.ndarray,
    palette: np.ndarray,
    alpha: float = 0.6,
    save_overlay: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    colour_mask = palette[mask]
    overlay = (image.astype(np.float32) * (1 - alpha) + colour_mask.astype(np.float32) * alpha).astype(np.uint8)
    combined = np.concatenate([image, colour_mask, overlay], axis=1)
    extra = overlay if save_overlay else None
    return combined, extra


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    state_dict = state.get("model_state", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return state.get("epoch"), missing, unexpected


def _ensure_logits_shape(logits: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == target_hw:
        return logits
    resized = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    resized = torch.clamp(resized, min=1e-8)
    normaliser = resized.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return resized / normaliser


def _iterate_loader(loader, step: int = 1, max_samples: Optional[int] = None):
    processed = 0
    for idx, batch in enumerate(loader):
        if step > 1 and idx % step != 0:
            continue
        yield idx, batch
        processed += 1
        if max_samples is not None and processed >= max_samples:
            break


def _format_confusion_matrix(
    confusion: np.ndarray, class_names: Optional[Sequence[str]] = None
) -> str:
    names = list(class_names) if class_names is not None else [str(i) for i in range(confusion.shape[0])]
    if not names:
        return ""
    max_name = max(len(name) for name in names)
    cell_width = max(max(len(str(int(val))) for val in confusion.flatten()), 1)
    header = " " * (max_name + 4) + " ".join(f"{name:>{cell_width}}" for name in names)
    lines = [header]
    for name, row in zip(names, confusion):
        row_str = " ".join(f"{int(val):>{cell_width}}" for val in row)
        lines.append(f"{name:>{max_name}} -> {row_str}")
    return "\n".join(lines)


def _run_split(
    split_name: str,
    loader,
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
    palette: np.ndarray,
    mean: Iterable[float],
    std: Iterable[float],
    step: int,
    max_samples: Optional[int],
    metric_logger: Optional[FacadeMetricLogger],
    save_overlay: bool,
    logger=None,
    class_names: Optional[Sequence[str]] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = output_dir / "overlay"
    if save_overlay:
        overlay_dir.mkdir(exist_ok=True)

    processed = 0
    if logger is not None:
        logger.info("Starting %s inference (step=%d, limit=%s)", split_name, step, max_samples)
    for _, batch in _iterate_loader(loader, step=step, max_samples=max_samples):
        img = _tensor_from_batch(batch, "img", device)
        logits = model.clip_backbone(img)
        logits = _ensure_logits_shape(logits, img.shape[-2:])
        preds = logits.argmax(dim=1)

        if metric_logger is not None and "gt_semantic_seg" in batch:
            target = _tensor_from_batch(batch, "gt_semantic_seg", device).squeeze(1).long()
            metric_logger.update(logits, target)

        meta = _extract_meta(batch)
        img_tensor = img[0].cpu()
        image = _denormalise_image(img_tensor, mean, std)
        mask = preds[0].detach().cpu().numpy().astype(np.int64)

        combined, overlay = _render_visualisation(image, mask, palette, save_overlay=save_overlay)

        filename = meta.get("ori_filename") or meta.get("filename") or f"{processed:06d}.png"
        stem = Path(filename).stem
        Image.fromarray(combined).save(output_dir / f"{stem}.png")
        if overlay is not None:
            Image.fromarray(overlay).save(overlay_dir / f"{stem}.png")

        processed += 1

    if logger is not None:
        logger.info("Finished %s inference on %d samples", split_name, processed)

    if metric_logger is not None:
        metrics = metric_logger.compute()
        per_class_iou = metric_logger.per_class_iou()
        confusion = metric_logger.confusion_matrix().cpu().numpy()
        label_names = class_names
        if label_names is None:
            label_names = getattr(metric_logger, "display_class_names", None)
        if logger is not None and confusion.size:
            logger.info(
                "%s confusion matrix:\n%s",
                split_name.capitalize(),
                _format_confusion_matrix(confusion, label_names),
            )
        return metrics, confusion, per_class_iou
    return None, None, None


def main() -> None:
    args = _parse_args()
    config_dir = Path(args.config_dir).resolve()
    cfg = _load_config(args.config_name, config_dir)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path(cfg.output) / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device(args.device)

    train_loader, val_loader, mmseg_cfg = _build_datasets(
        cfg,
        include_repeats=False,
        train_shuffle=False,
    )
    class_names, palette = _extract_dataset_metadata(train_loader.dataset)
    palette_arr = np.asarray(palette, dtype=np.uint8)
    img_norm_cfg = mmseg_cfg.get("img_norm_cfg", {})
    mean = img_norm_cfg.get("mean", [123.675, 116.28, 103.53])
    std = img_norm_cfg.get("std", [58.395, 57.12, 57.375])

    model = build_model(cfg.model, class_names=class_names)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger = get_logger(cfg)
    logger.info("Running inference on device %s", device)
    logger.info("Model parameters: %s", _format_param_count(total_params, trainable_params))

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    epoch, missing, unexpected = _load_checkpoint(model, checkpoint_path, device)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", sorted(missing))
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", sorted(unexpected))
    if epoch is not None:
        logger.info("Loaded checkpoint from epoch %s", epoch)

    train_samples = len(train_loader.dataset)
    desired_train = max(1, int(round(train_samples * min(max(args.train_ratio, 0.0), 1.0))))
    if args.max_train_samples is not None:
        desired_train = min(desired_train, args.max_train_samples)
    step = max(1, math.ceil(train_samples / desired_train)) if desired_train else 1

    logger.info(
        "Processing approximately %d/%d training samples (step=%d)",
        math.ceil(train_samples / step),
        train_samples,
        step,
    )

    train_limit = None if step == 1 else math.ceil(train_samples / step)
    train_metrics, _, train_per_class = _run_split(
        "train",
        train_loader,
        model,
        device,
        output_dir / "train",
        palette_arr,
        mean,
        std,
        step,
        train_limit,
        FacadeMetricLogger(class_names),
        args.save_overlay,
        logger=logger,
        class_names=class_names,
    )

    if train_metrics:
        logger.info("Train metrics: %s", train_metrics)
    if train_per_class:
        rounded_train = {k: round(v, 4) for k, v in train_per_class.items()}
        logger.info("Train per-class IoU: %s", rounded_train)

    val_step = max(1, args.val_interval)
    logger.info("Processing validation set with step=%d", val_step)

    val_metrics, _, val_per_class = _run_split(
        "val",
        val_loader,
        model,
        device,
        output_dir / "val",
        palette_arr,
        mean,
        std,
        val_step,
        None,
        FacadeMetricLogger(class_names),
        args.save_overlay,
        logger=logger,
        class_names=class_names,
    )

    if val_metrics:
        logger.info("Validation metrics: %s", val_metrics)
    if val_per_class:
        rounded_val = {k: round(v, 4) for k, v in val_per_class.items()}
        logger.info("Validation per-class IoU: %s", rounded_val)


if __name__ == "__main__":
    main()

