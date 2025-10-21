"""Quantitative evaluation utilities for facade damage models."""

from __future__ import annotations
import os
import sys
import argparse
import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING


# NOTE: When this script is executed directly (``python lposs/tools/...py``) the
# working directory inserted into ``sys.path`` is ``.../lposs/tools``. That path
# does not contain the ``lposs`` package itself, so we explicitly add the
# repository root to the module search path to keep ``import lposs`` statements
# working even without ``pip install -e``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:  # pragma: no cover - typing aid
    import numpy as np

try:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
from hydra import compose, initialize_config_dir

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - hydra dependency provides it at runtime
    OmegaConf = None  # type: ignore

from tqdm.auto import tqdm

from lposs.helpers.logger import get_logger
from lposs.metrics.facade_metrics import FacadeMetricLogger
from lposs.models import build_model
from lposs.tools.train_facade_baseline import (
    CONFIG_DIR as TRAIN_CONFIG_DIR,
    _build_datasets,
    _tensor_from_batch,
)
from lposs.tools.infer_facade import (
    _ensure_logits_shape,
    _extract_dataset_metadata,
    _get_device,
    _load_checkpoint,
)


DEFAULT_CONFIG_DIR = Path(TRAIN_CONFIG_DIR)


@dataclass
class ModelSpec:
    """Definition of a model variant to evaluate."""

    label: str
    checkpoint: Optional[Path]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned and stock facade models on segmentation metrics.",
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
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory that contains Hydra configs (defaults to lposs/configs).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (cpu, cuda, or cuda:N). Defaults to auto-detection.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where metrics, confusion matrices, and figures will be written.",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        type=str,
        default=None,
        help="Path to a fine-tuned checkpoint (evaluates alongside the stock model).",
    )
    parser.add_argument(
        "--stock-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint for the stock model. If omitted the freshly initialised model is used.",
    )
    parser.add_argument(
        "--skip-stock",
        action="store_true",
        help="Skip evaluating the stock (pre-finetuning) model variant.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=("val",),
        choices=("train", "val"),
        help="Dataset splits to evaluate. Supported values: train, val.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of samples processed per split (useful for smoke-tests).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help=(
            "Override the dataset root directory used for the Hydra configuration. "
            "Equivalent to setting the FACADE_DATA_ROOT environment variable."
        ),
    )
    return parser.parse_args()


def _resolve_models(args: argparse.Namespace) -> List[ModelSpec]:
    models: List[ModelSpec] = []
    if not args.skip_stock:
        checkpoint = Path(args.stock_checkpoint).expanduser().resolve() if args.stock_checkpoint else None
        models.append(ModelSpec(label="stock", checkpoint=checkpoint))
    if args.finetuned_checkpoint:
        checkpoint = Path(args.finetuned_checkpoint).expanduser().resolve()
        models.append(ModelSpec(label="finetuned", checkpoint=checkpoint))
    if not models:
        raise ValueError("No models selected for evaluation. Provide --finetuned-checkpoint and/or disable --skip-stock.")
    return models


def _load_config(config_name: str, config_dir: Path):
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name)


def _build_output_dirs(base: Path, splits: Iterable[str]) -> Dict[str, Path]:
    base.mkdir(parents=True, exist_ok=True)
    split_dirs = {}
    for split in splits:
        split_dir = base / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = split_dir
    return split_dirs


def _save_metrics(path: Path, metrics: Dict[str, float]) -> None:
    with path.open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)


def _save_per_class(path: Path, values: Dict[str, float]) -> None:
    with path.open("w") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)


def _save_confusion_csv(path: Path, confusion: "np.ndarray", class_names: Sequence[str]) -> None:
    header = ["class"] + list(class_names)
    with path.open("w") as handle:
        handle.write(",".join(header) + "\n")
        for name, row in zip(class_names, confusion):
            values = ",".join(str(int(value)) for value in row)
            handle.write(f"{name},{values}\n")


def _normalise_confusion(confusion: "np.ndarray") -> "np.ndarray":
    row_sums = confusion.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = confusion / np.maximum(row_sums, 1)
    normalised[row_sums.squeeze(1) == 0] = 0
    return normalised


def _save_confusion_figure(
    path: Path,
    confusion: "np.ndarray",
    class_names: Sequence[str],
    *,
    normalised: bool = False,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to render confusion matrix figures. Install it or rerun without figure generation."
        )
    title = "Confusion matrix"
    if normalised:
        title = "Normalised confusion matrix"
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.6), max(5, len(class_names) * 0.5)))
    cmap = plt.get_cmap("Blues")
    im = ax.imshow(confusion, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_ylabel("Ground truth")
    ax.set_xlabel("Predicted")
    ax.set_title(title)

    max_val = confusion.max() if confusion.size else 0
    thresh = max_val / 2.0 if max_val else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = confusion[i, j]
            if normalised:
                text = f"{value:.2f}"
            else:
                text = f"{int(round(value))}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _evaluate_split(
    loader,
    model,
    device: "torch.device",
    class_names: Sequence[str],
    max_samples: Optional[int] = None,
    *,
    progress=None,
):
    metric_logger = FacadeMetricLogger(class_names)
    processed = 0
    amp_enabled = device.type.startswith("cuda") and torch.cuda.is_available()
    if amp_enabled and hasattr(torch.cuda, "amp"):
        autocast_ctx = torch.cuda.amp.autocast
    else:
        autocast_ctx = nullcontext
    with torch.no_grad():
        for batch in loader:
            img = _tensor_from_batch(batch, "img", device)
            with autocast_ctx():
                logits = model.clip_backbone(img)
            logits = logits.float()
            logits = _ensure_logits_shape(logits, img.shape[-2:])
            if "gt_semantic_seg" not in batch:
                continue
            target = _tensor_from_batch(batch, "gt_semantic_seg", device).squeeze(1).long()
            metric_logger.update(logits, target)
            batch_size = img.shape[0]
            if progress is not None:
                if max_samples is not None:
                    remaining = max(max_samples - processed, 0)
                    increment = min(batch_size, remaining)
                else:
                    increment = batch_size
                if increment > 0:
                    progress.update(increment)
            processed += batch_size
            if max_samples is not None and processed >= max_samples:
                break
    return metric_logger


def _summarise_results(
    model_spec: ModelSpec,
    split: str,
    metric_logger: FacadeMetricLogger,
    output_dir: Path,
    class_names: Sequence[str],
    logger,
) -> Dict[str, float]:
    metrics = metric_logger.compute()
    per_class = metric_logger.per_class_iou()
    confusion = metric_logger.confusion_matrix().cpu().numpy()

    split_dir = output_dir / model_spec.label
    split_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = split_dir / f"{split}_metrics.json"
    per_class_path = split_dir / f"{split}_per_class_iou.json"
    confusion_csv = split_dir / f"{split}_confusion.csv"
    confusion_png = split_dir / f"{split}_confusion.png"
    confusion_norm_png = split_dir / f"{split}_confusion_normalised.png"

    _save_metrics(metrics_path, metrics)
    _save_per_class(per_class_path, per_class)
    if confusion.size:
        _save_confusion_csv(confusion_csv, confusion, class_names)
        try:
            _save_confusion_figure(confusion_png, confusion, class_names, normalised=False)
            normalised = _normalise_confusion(confusion)
            _save_confusion_figure(confusion_norm_png, normalised, class_names, normalised=True)
        except RuntimeError as exc:
            logger.warning(
                "[%s][%s] %s", model_spec.label, split, exc
            )

    rounded = {k: round(v, 4) for k, v in metrics.items()}
    logger.info("[%s][%s] metrics: %s", model_spec.label, split, rounded)
    logger.info("[%s][%s] per-class IoU: %s", model_spec.label, split, per_class)
    if confusion.size:
        logger.info("[%s][%s] confusion matrix saved to %s", model_spec.label, split, confusion_png)

    return metrics


def main() -> None:
    args = _parse_args()
    if np is None:
        raise RuntimeError("numpy is required for evaluation. Install it with 'pip install numpy'.")
    if torch is None:
        raise RuntimeError("PyTorch is required for evaluation. Install it following the repository instructions.")
    config_dir = Path(args.config_dir).resolve()
    cfg = _load_config(args.config_name, config_dir)
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        OmegaConf.set_struct(cfg, False)
    models = _resolve_models(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        cfg.output = str(output_dir)
    except AttributeError:
        if isinstance(cfg, dict):
            cfg["output"] = str(output_dir)
        else:
            raise
    split_dirs = _build_output_dirs(output_dir, args.splits)

    device = _get_device(args.device)
    logger = get_logger(cfg)
    logger.info("Evaluating models on device %s", device)

    if args.dataset_root:
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        def _assign_dataset_root(target):
            if target is None:
                return False
            if hasattr(target, "dataset"):
                dataset_cfg = target.dataset
            elif isinstance(target, dict):
                dataset_cfg = target.get("dataset")
            else:
                dataset_cfg = None
            if dataset_cfg is None:
                return False
            if hasattr(dataset_cfg, "data_root"):
                dataset_cfg.data_root = str(dataset_root)
                return True
            if isinstance(dataset_cfg, dict):
                dataset_cfg["data_root"] = str(dataset_root)
                return True
            return False

        assigned = False
        training_cfg = getattr(cfg, "training", None)
        if training_cfg is None and isinstance(cfg, dict):
            training_cfg = cfg.get("training")
        if training_cfg is not None:
            assigned = _assign_dataset_root(training_cfg)
        if not assigned:
            raise AttributeError(
                "Unable to override dataset root: configuration is missing training.dataset section."
            )
        os.environ["FACADE_DATA_ROOT"] = str(dataset_root)
        logger.info("Overriding dataset root to %s", dataset_root)

    train_loader, val_loader, mmseg_cfg = _build_datasets(
        cfg,
        include_repeats=False,
        train_shuffle=False,
    )
    class_names, _ = _extract_dataset_metadata(train_loader.dataset)

    loaders = {"train": train_loader, "val": val_loader}

    dataset_sizes = {
        "train": len(train_loader.dataset),
        "val": len(val_loader.dataset),
    }
    for split, size in dataset_sizes.items():
        if size == 0:
            data_cfg = getattr(mmseg_cfg.data, split, None)
            img_dir = None
            if data_cfg is not None:
                data_root = getattr(data_cfg, "data_root", None)
                img_subdir = getattr(data_cfg, "img_dir", None)
                if data_root and img_subdir:
                    img_dir = Path(data_root) / img_subdir
            message = f"No samples found for '{split}' split."
            if img_dir is not None:
                message += f" Expected images under {img_dir}."
            if split in args.splits:
                raise RuntimeError(message)
            logger.warning(message)

    summary: Dict[str, Dict[str, float]] = {}

    for model_spec in models:
        logger.info("Preparing model '%s'", model_spec.label)
        model = build_model(cfg.model, class_names=class_names)
        model.to(device)
        model.eval()

        if model_spec.checkpoint is not None:
            if not model_spec.checkpoint.is_file():
                raise FileNotFoundError(f"Checkpoint not found: {model_spec.checkpoint}")
            epoch, missing, unexpected = _load_checkpoint(model, model_spec.checkpoint, device)
            if missing:
                logger.warning("[%s] Missing keys when loading checkpoint: %s", model_spec.label, sorted(missing))
            if unexpected:
                logger.warning("[%s] Unexpected keys when loading checkpoint: %s", model_spec.label, sorted(unexpected))
            if epoch is not None:
                logger.info("[%s] Loaded checkpoint from epoch %s", model_spec.label, epoch)
        else:
            logger.info("[%s] Using freshly initialised weights (stock model).", model_spec.label)

        torch.set_grad_enabled(False)

        for split in args.splits:
            loader = loaders[split]
            logger.info(
                "[%s] Evaluating %s split with %d samples", model_spec.label, split, len(loader.dataset)
            )
            total_samples = len(loader.dataset)
            if args.max_samples is not None:
                total_samples = min(total_samples, args.max_samples)
            progress_bar = None
            if total_samples > 0:
                progress_bar = tqdm(
                    total=total_samples,
                    desc=f"[{model_spec.label}] {split}",
                    unit="sample",
                    dynamic_ncols=True,
                )
            try:
                metric_logger = _evaluate_split(
                    loader,
                    model,
                    device,
                    class_names,
                    max_samples=args.max_samples,
                    progress=progress_bar,
                )
            finally:
                if progress_bar is not None:
                    progress_bar.close()
            metrics = _summarise_results(
                model_spec,
                split,
                metric_logger,
                split_dirs[split],
                class_names,
                logger,
            )
            summary.setdefault(model_spec.label, {})[split] = metrics

        del model
        torch.cuda.empty_cache()

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    logger.info("Written aggregate summary to %s", summary_path)


if __name__ == "__main__":
    main()

