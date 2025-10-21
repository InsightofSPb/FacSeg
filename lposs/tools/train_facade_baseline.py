"""Baseline fine-tuning loop for facade damage segmentation."""

import copy
import importlib
import importlib.util
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from hydra import compose, initialize_config_dir
from mmcv import Config
from mmcv.parallel import DataContainer
from mmcv.utils import ConfigDict
from mmcv.runner import set_random_seed
from mmseg.datasets import build_dataloader, build_dataset
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
CONFIG_DIR = PROJECT_ROOT / "configs"

for path in (REPO_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from lposs.helpers.logger import get_logger
from lposs.metrics.facade_metrics import FacadeMetricLogger
from lposs.models import build_model

def _import_module_from_path(qualified_name: str, path: Path) -> None:
    """Import a module from a specific file path under the given name."""

    if qualified_name in sys.modules:
        return

    if not path.is_file():  # pragma: no cover - configuration issue
        raise FileNotFoundError(f"Module source not found at {path}")

    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - configuration issue
        raise ImportError(f"Unable to load spec for module {qualified_name} from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)

def _ensure_custom_modules(modules: Iterable[str]) -> None:
    """Import project-specific modules so MMCV registries discover them."""

    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:  # pragma: no cover - configuration issue
            raise ImportError(
                f"Failed to import required module '{module}' for facade training"
            ) from exc
def _resolve_config_path(config_path: Union[str, Path]) -> Path:
    """Resolve dataset config path allowing repo-relative references."""
    path = Path(config_path)
    if path.is_file():
        return path

    candidates = (
        PROJECT_ROOT / path,
        PROJECT_ROOT.parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Dataset config file not found for '{config_path}'. Checked: {[str(p) for p in candidates]}"
    )

def _build_datasets(
    cfg: Dict,
    dist: bool = False,
    include_repeats: bool = True,
    train_shuffle: bool = True,
):

    _ensure_custom_modules(("segmentation.datasets.facade_damage",))
    _import_module_from_path(
        "mmseg.datasets.pipelines.facade_augment",
        PROJECT_ROOT / "mmseg" / "datasets" / "pipelines" / "facade_augment.py",
    )
    dataset_cfg = cfg.training.dataset

    def _as_bool(value, *, default: Optional[bool] = None) -> Optional[bool]:
        if value is None:
            return default
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return bool(value)

    tile_mode = bool(_as_bool(dataset_cfg.get("tile_mode"), default=False))
    recursive_override = dataset_cfg.get("recursive", None)

    dataset_config_path = _resolve_config_path(dataset_cfg.config)
    dataset_cfg.config = str(dataset_config_path)
    mmseg_cfg = Config.fromfile(str(dataset_config_path))

    if hasattr(cfg, "evaluate") and cfg.evaluate is not None:
        for task in cfg.evaluate.get("task", []):
            task_cfg = cfg.evaluate.get(task)
            if task_cfg and task_cfg.get("config"):
                task_cfg["config"] = str(_resolve_config_path(task_cfg["config"]))
    if dataset_cfg.get('data_root'):
        mmseg_cfg.data.train.data_root = dataset_cfg.data_root
        mmseg_cfg.data.val.data_root = dataset_cfg.data_root
    if dataset_cfg.get('classes'):
        mmseg_cfg.data.train.classes = dataset_cfg.classes
        mmseg_cfg.data.val.classes = dataset_cfg.classes

    if tile_mode:
        default_norm = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        norm_cfg = mmseg_cfg.get("img_norm_cfg", None) or default_norm
        norm_cfg = copy.deepcopy(norm_cfg)
        default_meta_keys = [
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "flip_direction",
        ]
        meta_keys = mmseg_cfg.get("meta_keys", None) or default_meta_keys
        meta_keys = list(meta_keys)

        def _make_pipeline(include_annotations: bool):
            steps = [dict(type="LoadImageFromFile")]
            if include_annotations:
                steps.append(dict(type="LoadAnnotations"))
            steps.append(dict(type="Normalize", **copy.deepcopy(norm_cfg)))
            steps.append(dict(type="ImageToTensor", keys=["img"]))
            collect_meta_keys = [
                key for key in meta_keys if key not in {"flip", "flip_direction"}
            ]
            if include_annotations:
                steps.append(dict(type="ToTensor", keys=["gt_semantic_seg"]))
                steps.append(
                    dict(
                        type="Collect",
                        keys=["img", "gt_semantic_seg"],
                        meta_keys=list(collect_meta_keys),
                    )
                )
            else:
                steps.append(
                    dict(
                        type="Collect",
                        keys=["img"],
                        meta_keys=list(collect_meta_keys),
                    )
                )
            return steps

        mmseg_cfg.data.train.pipeline = _make_pipeline(include_annotations=True)
        mmseg_cfg.data.val.pipeline = _make_pipeline(include_annotations=True)
        if hasattr(mmseg_cfg.data, "test"):
            mmseg_cfg.data.test.pipeline = _make_pipeline(include_annotations=False)

    recursive_flag = recursive_override
    if recursive_flag is None and tile_mode:
        recursive_flag = True

    if recursive_flag is not None:
        recursive_value = bool(_as_bool(recursive_flag, default=True))

        def _apply_recursive(cfg_node):
            if cfg_node is None:
                return
            if isinstance(cfg_node, dict):
                cfg_node["recursive"] = recursive_value

        _apply_recursive(mmseg_cfg.data.train)
        _apply_recursive(mmseg_cfg.data.val)
        if hasattr(mmseg_cfg.data, "test"):
            _apply_recursive(mmseg_cfg.data.test)
    train_cfg = mmseg_cfg.data.train
    repeat_times = dataset_cfg.get('repeat_times', 1)
    if include_repeats and repeat_times > 1:
        train_cfg = ConfigDict(dict(type='RepeatDataset', times=repeat_times, dataset=train_cfg))
    train_dataset = build_dataset(train_cfg)
    val_dataset = build_dataset(mmseg_cfg.data.val)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.training.samples_per_gpu,
        workers_per_gpu=cfg.training.workers_per_gpu,
        shuffle=train_shuffle,
        dist=dist,
    )
    val_samples_per_gpu = getattr(
        cfg.training, "val_samples_per_gpu", cfg.training.samples_per_gpu
    )
    val_workers_per_gpu = getattr(cfg.training, "val_workers_per_gpu", cfg.training.workers_per_gpu)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=val_samples_per_gpu,
        workers_per_gpu=val_workers_per_gpu,
        shuffle=False,
        dist=dist,
    )
    return train_loader, val_loader, mmseg_cfg


def save_checkpoint(
    model,
    optimiser,
    epoch: int,
    path: Path,
    *,
    metadata: Optional[Dict[str, Union[str, float, int]]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimiser.state_dict(),
    }
    if metadata:
        payload["metadata"] = metadata
    torch.save(payload, path)


class BestCheckpointManager:
    """Track and persist the top-N checkpoints per split/metric."""

    def __init__(
        self,
        root: Union[str, Path],
        *,
        max_keep: int = 5,
        logger=None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_keep = max(1, int(max_keep))
        self.logger = logger
        self._criteria: Dict[Tuple[str, str], str] = {
            ("train", "loss"): "min",
            ("train", "mIoU"): "max",
            ("val", "loss"): "min",
            ("val", "mIoU"): "max",
        }
        self._records: Dict[Tuple[str, str], List[Dict[str, Union[float, Path, int]]]] = {
            key: [] for key in self._criteria
        }

    def maybe_save(
        self,
        *,
        split: str,
        metric: str,
        value: float,
        epoch: int,
        model,
        optimiser,
    ) -> bool:
        key = (split, metric)
        if key not in self._criteria:
            raise KeyError(f"Unsupported checkpoint key {key}")
        if value is None or not math.isfinite(value):
            return False

        criterion = self._criteria[key]
        records = self._records[key]
        directory = self.root / split / metric
        directory.mkdir(parents=True, exist_ok=True)
        filename = f"epoch_{epoch:03d}_{value:.4f}.pth"
        path = directory / filename

        save_checkpoint(
            model,
            optimiser,
            epoch,
            path,
            metadata={
                "split": split,
                "metric": metric,
                "value": float(value),
                "timestamp": time.time(),
            },
        )

        entry: Dict[str, Union[float, Path, int]] = {
            "value": float(value),
            "path": path,
            "epoch": epoch,
        }
        records.append(entry)
        records.sort(key=lambda item: item["value"], reverse=(criterion == "max"))

        removed: List[Dict[str, Union[float, Path, int]]] = []
        while len(records) > self.max_keep:
            removed.append(records.pop(-1))

        for item in removed:
            ckpt_path = item["path"]
            if isinstance(ckpt_path, Path) and ckpt_path.exists():
                ckpt_path.unlink()
                if self.logger is not None:
                    self.logger.info(
                        "Removed checkpoint %s to maintain top %d for %s/%s",
                        ckpt_path,
                        self.max_keep,
                        split,
                        metric,
                    )

        if entry in records:
            if self.logger is not None:
                self.logger.info(
                    "Saved checkpoint for %s %s (value=%.4f) at %s",
                    split,
                    metric,
                    value,
                    path,
                )
            return True

        if path.exists():
            path.unlink()
        return False

def _unwrap_data_container(value):
    """Recursively unwrap MMCV DataContainer/list wrappers to a tensor."""

    if isinstance(value, DataContainer):
        return _unwrap_data_container(value.data)
    if isinstance(value, (list, tuple)):
        if not value:  # pragma: no cover - defensive programming
            raise ValueError("Received an empty sequence when unwrapping data container")
        return _unwrap_data_container(value[0])
    return value


def _tensor_from_batch(batch, key: str, device: torch.device) -> torch.Tensor:
    """Extract the first tensor for ``key`` from MMCV batch output and move it."""

    tensor = _unwrap_data_container(batch[key])
    if not isinstance(tensor, torch.Tensor):  # pragma: no cover - configuration issue
        raise TypeError(f"Expected tensor for key '{key}', received {type(tensor)!r}")
    return tensor.to(device)


def _run_lposs_sanity_check(
    cfg,
    logger,
    device: torch.device,
    sample_limit: int,
) -> None:
    """Execute a quick training/validation cycle to verify the pipeline."""

    try:
        limit = int(sample_limit)
    except (TypeError, ValueError):
        limit = 20
    limit = max(2, limit)
    logger.info("Running LPOSS sanity check (limit ≈ %d samples).", limit)

    sanity_train_loader, sanity_val_loader, _ = _build_datasets(
        cfg,
        include_repeats=False,
        train_shuffle=False,
    )

    class_names = list(getattr(sanity_train_loader.dataset, "CLASSES", []))
    if not class_names and hasattr(sanity_val_loader.dataset, "CLASSES"):
        class_names = list(sanity_val_loader.dataset.CLASSES)

    model = build_model(cfg.model, class_names=class_names)
    model.to(device)
    optimiser, _ = model.configure_optimiser(total_epochs=1)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    train_metric_logger = FacadeMetricLogger(class_names)
    val_metric_logger = FacadeMetricLogger(class_names)

    train_target = max(1, limit // 2)
    val_target = max(1, limit - train_target)
    train_processed = 0
    val_processed = 0
    train_batches = 0
    val_batches = 0
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    model.update_backbone_trainable(1, logger)
    for data in sanity_train_loader:
        img = _tensor_from_batch(data, "img", device)
        target = _tensor_from_batch(data, "gt_semantic_seg", device).squeeze(1).long()
        optimiser.zero_grad()
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model.training_step(img, target)
            loss = outputs["loss"]
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        logits = outputs.get("logits")
        if logits is not None:
            train_metric_logger.update(logits.detach(), target)
        batch_size = int(img.shape[0])
        train_processed += batch_size
        train_batches += 1
        train_loss += float(loss.item())
        if train_processed >= train_target:
            break

    if train_processed == 0:
        logger.warning("Sanity check: no training samples were processed.")

    train_metrics = train_metric_logger.compute()
    if train_batches > 0:
        train_metrics["loss"] = train_loss / train_batches
    else:
        train_metrics["loss"] = float("nan")

    model.eval()
    with torch.no_grad():
        for data in sanity_val_loader:
            img = _tensor_from_batch(data, "img", device)
            target = _tensor_from_batch(data, "gt_semantic_seg", device).squeeze(1).long()
            outputs = model.training_step(img, target)
            loss = outputs.get("loss")
            if loss is not None:
                val_loss += float(loss.item())
                val_batches += 1
            logits = outputs.get("logits")
            if logits is not None:
                val_metric_logger.update(logits.detach(), target)
            batch_size = int(img.shape[0])
            val_processed += batch_size
            if val_processed >= val_target:
                break

    if val_processed == 0:
        logger.warning("Sanity check: no validation samples were processed.")

    val_metrics = val_metric_logger.compute()
    if val_batches > 0:
        val_metrics["loss"] = val_loss / val_batches
    else:
        val_metrics["loss"] = float("nan")

    def _round_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        rounded: Dict[str, float] = {}
        for key, value in metrics.items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                rounded[key] = round(numeric, 4)
        return rounded

    train_summary = _round_metrics(train_metrics)
    val_summary = _round_metrics(val_metrics)

    sanity_dir = Path(cfg.training.checkpoint_dir) / "sanity_check"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = sanity_dir / "sanity_checkpoint.pth"
    save_checkpoint(
        model,
        optimiser,
        epoch=0,
        path=ckpt_path,
        metadata={
            "stage": "sanity_check",
            "train_samples": train_processed,
            "val_samples": val_processed,
            "train_metrics": train_summary,
            "val_metrics": val_summary,
        },
    )

    logger.info(
        "Sanity check metrics (train=%d, val=%d): train=%s, val=%s",
        train_processed,
        val_processed,
        train_summary,
        val_summary,
    )
    logger.info("Sanity check checkpoint saved to %s", ckpt_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    del sanity_train_loader
    del sanity_val_loader

def validate(
    model,
    data_loader,
    metric_logger: FacadeMetricLogger,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metric_logger.reset()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Validating", leave=False):
            img = _tensor_from_batch(data, 'img', device)
            target = _tensor_from_batch(data, 'gt_semantic_seg', device).squeeze(1).long()
            outputs = model.training_step(img, target)
            loss = outputs.get("loss")
            logits = outputs.get("logits")
            if loss is not None:
                total_loss += loss.item()
                num_batches += 1
            if logits is not None:
                metric_logger.update(logits.detach(), target)
    metrics = metric_logger.compute()
    if num_batches > 0:
        metrics["loss"] = total_loss / num_batches
    else:
        metrics["loss"] = float("nan")
    return metrics


def train(cfg, *, sanity_check: bool = True, sanity_sample_limit: int = 20) -> None:
    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seed(cfg.seed)

    if sanity_check:
        try:
            _run_lposs_sanity_check(
                cfg,
                logger,
                device,
                sanity_sample_limit,
            )
        except Exception:  # pragma: no cover - sanity check must surface errors
            logger.exception("LPOSS sanity check failed")
            raise
        set_random_seed(cfg.seed)
    else:
        logger.info("Sanity check disabled; proceeding directly to training.")

    train_loader, val_loader, _ = _build_datasets(cfg)
    repeat_times = 1
    dataset_cfg = getattr(cfg.training, "dataset", None)
    if dataset_cfg is not None:
        if hasattr(dataset_cfg, "get"):
            repeat_times = dataset_cfg.get("repeat_times", repeat_times)
        else:
            repeat_times = getattr(dataset_cfg, "repeat_times", repeat_times)
    total_train_images = len(train_loader.dataset)
    if repeat_times and repeat_times > 1:
        logger.info(
            "Train dataset contains %d images in total after applying repeat factor x%d.",
            total_train_images,
            repeat_times,
        )
    else:
        logger.info("Train dataset contains %d images in total.", total_train_images)
    class_names = list(train_loader.dataset.CLASSES)
    model = build_model(cfg.model, class_names=class_names)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model parameters: total=%.2fM, trainable=%.2fM (%d / %d)",
        total_params / 1e6,
        trainable_params / 1e6,
        trainable_params,
        total_params,
    )
    optimiser, scheduler = model.configure_optimiser(total_epochs=cfg.training.max_epochs)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    train_metric_logger = FacadeMetricLogger(class_names)
    val_metric_logger = FacadeMetricLogger(class_names)
    confusion_dir = getattr(cfg.training, "confusion_dir", None)
    if confusion_dir:
        confusion_dir = Path(confusion_dir)
        confusion_dir.mkdir(parents=True, exist_ok=True)

    max_best_checkpoints = getattr(cfg.training, "max_best_checkpoints", 5)
    checkpoint_manager = BestCheckpointManager(
        Path(cfg.training.checkpoint_dir),
        max_keep=max_best_checkpoints,
        logger=logger,
    )

    for epoch in range(1, cfg.training.max_epochs + 1):
        model.train()
        train_metric_logger.reset()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        model.update_backbone_trainable(epoch, logger)
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.training.max_epochs}",
            leave=False,
        )
        for idx, data in enumerate(progress, start=1):
            img = _tensor_from_batch(data, 'img', device)
            target = _tensor_from_batch(data, 'gt_semantic_seg', device).squeeze(1).long()
            optimiser.zero_grad()
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model.training_step(img, target)
                loss = outputs['loss']
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            epoch_loss += loss.item()
            num_batches += 1
            logits = outputs.get('logits')
            if logits is not None:
                train_metric_logger.update(logits.detach(), target)
            if idx % cfg.training.log_interval == 0:
                progress.set_postfix(loss=loss.item(), avg_loss=epoch_loss / num_batches)

        elapsed = time.time() - start_time
        average_train_loss = epoch_loss / max(num_batches, 1)
        logger.info(
            f"Epoch {epoch} finished: loss={average_train_loss:.4f} time={elapsed:.1f}s"
        )

        train_metrics = train_metric_logger.compute()
        train_metrics["loss"] = average_train_loss
        logger.info(f"Training metrics at epoch {epoch}: {train_metrics}")
        logger.info(
            "Per-class IoU (train) at epoch %d: %s",
            epoch,
            {k: round(v, 4) for k, v in train_metric_logger.per_class_iou().items()},
        )

        checkpoint_manager.maybe_save(
            split="train",
            metric="loss",
            value=average_train_loss,
            epoch=epoch,
            model=model,
            optimiser=optimiser,
        )
        train_miou = train_metrics.get("mIoU")
        if train_miou is not None:
            checkpoint_manager.maybe_save(
                split="train",
                metric="mIoU",
                value=float(train_miou),
                epoch=epoch,
                model=model,
                optimiser=optimiser,
            )

        if epoch % cfg.training.val_interval == 0:
            metrics = validate(model, val_loader, val_metric_logger, device)
            logger.info(f"Validation metrics at epoch {epoch}: {metrics}")
            logger.info(
                "Per-class IoU (val) at epoch %d: %s",
                epoch,
                {k: round(v, 4) for k, v in val_metric_logger.per_class_iou().items()},
            )
            if confusion_dir:
                val_metric_logger.export_confusion(
                    confusion_dir / f"val_epoch_{epoch:03d}.csv",
                    class_names=class_names,
                )

            val_loss = metrics.get("loss")
            if val_loss is not None:
                checkpoint_manager.maybe_save(
                    split="val",
                    metric="loss",
                    value=float(val_loss),
                    epoch=epoch,
                    model=model,
                    optimiser=optimiser,
                )
            val_miou = metrics.get("mIoU")
            if val_miou is not None:
                checkpoint_manager.maybe_save(
                    split="val",
                    metric="mIoU",
                    value=float(val_miou),
                    epoch=epoch,
                    model=model,
                    optimiser=optimiser,
                )

        if confusion_dir:
            train_metric_logger.export_confusion(
                confusion_dir / f"train_epoch_{epoch:03d}.csv",
                class_names=class_names,
            )

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimiser.param_groups[0]["lr"]
        logger.info("Learning rate after epoch %d: %.6e", epoch, current_lr)


if __name__ == '__main__':
    if not CONFIG_DIR.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {CONFIG_DIR}")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="facade_baseline.yaml")

    train(cfg)
