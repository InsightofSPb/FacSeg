"""Baseline fine-tuning loop for facade damage segmentation."""

import importlib
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Union

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
CONFIG_DIR = PROJECT_ROOT / "configs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from helpers.logger import get_logger
from metrics.facade_metrics import FacadeMetricLogger
from models import build_model

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

def _build_datasets(cfg: Dict, dist: bool = False):

    _ensure_custom_modules(("segmentation.datasets.facade_damage",))
    _import_module_from_path(
        "mmseg.datasets.pipelines.facade_augment",
        PROJECT_ROOT / "mmseg" / "datasets" / "pipelines" / "facade_augment.py",
    )
    dataset_cfg = cfg.training.dataset
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
    train_cfg = mmseg_cfg.data.train
    repeat_times = dataset_cfg.get('repeat_times', 1)
    if repeat_times > 1:
        train_cfg = ConfigDict(dict(type='RepeatDataset', times=repeat_times, dataset=train_cfg))
    train_dataset = build_dataset(train_cfg)
    val_dataset = build_dataset(mmseg_cfg.data.val)
    train_loader = build_dataloader(
        train_dataset,
        samples_per_gpu=cfg.training.samples_per_gpu,
        workers_per_gpu=cfg.training.workers_per_gpu,
        shuffle=True,
        dist=dist,
    )
    val_samples_per_gpu = getattr(cfg.training, "val_samples_per_gpu", 1)
    val_workers_per_gpu = getattr(cfg.training, "val_workers_per_gpu", cfg.training.workers_per_gpu)
    val_loader = build_dataloader(
        val_dataset,
        samples_per_gpu=val_samples_per_gpu,
        workers_per_gpu=val_workers_per_gpu,
        shuffle=False,
        dist=dist,
    )
    return train_loader, val_loader


def save_checkpoint(model, optimiser, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimiser.state_dict(),
    }, path)

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


def validate(model, data_loader, metric_logger: FacadeMetricLogger, device: torch.device) -> Dict[str, float]:
    model.eval()
    metric_logger.reset()
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Validating", leave=False):
            img = _tensor_from_batch(data, 'img', device)
            target = _tensor_from_batch(data, 'gt_semantic_seg', device).squeeze(1).long()
            logits = model.forward_train(img)
            metric_logger.update(logits, target)
    return metric_logger.compute()


def train(cfg) -> None:
    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seed(cfg.seed)

    train_loader, val_loader = _build_datasets(cfg)
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
    class_names = train_loader.dataset.CLASSES
    model = build_model(cfg.model, class_names=class_names)
    model.to(device)
    optimiser = model.configure_optimiser()
    scaler = GradScaler(enabled=torch.cuda.is_available())
    metric_logger = FacadeMetricLogger(len(class_names))

    for epoch in range(1, cfg.training.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
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
            if idx % cfg.training.log_interval == 0:
                progress.set_postfix(loss=loss.item(), avg_loss=epoch_loss / num_batches)

        elapsed = time.time() - start_time
        logger.info(f"Epoch {epoch} finished: loss={epoch_loss/num_batches:.4f} time={elapsed:.1f}s")

        if epoch % cfg.training.val_interval == 0:
            metrics = validate(model, val_loader, metric_logger, device)
            logger.info(f"Validation metrics at epoch {epoch}: {metrics}")

        ckpt_path = Path(cfg.training.checkpoint_dir) / f"epoch_{epoch:03d}.pth"
        save_checkpoint(model, optimiser, epoch, ckpt_path)


if __name__ == '__main__':
    if not CONFIG_DIR.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {CONFIG_DIR}")

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="facade_baseline.yaml")

    train(cfg)
