"""Baseline fine-tuning loop for facade damage segmentation."""

import os
import time
from pathlib import Path
from typing import Dict

import torch
from hydra import compose, initialize
from mmcv import Config
from mmcv.utils import ConfigDict
from mmcv.runner import set_random_seed
from mmseg.datasets import build_dataloader, build_dataset
from torch.cuda.amp import GradScaler, autocast

from helpers.logger import get_logger
from metrics.facade_metrics import FacadeMetricLogger
from models import build_model


def _build_datasets(cfg: Dict, dist: bool = False):
    dataset_cfg = cfg.training.dataset
    mmseg_cfg = Config.fromfile(dataset_cfg.config)
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
    train_loader = build_dataloader(train_dataset, samples_per_gpu=cfg.training.samples_per_gpu,
                                    workers_per_gpu=cfg.training.workers_per_gpu, shuffle=True, dist=dist)
    val_loader = build_dataloader(val_dataset, samples_per_gpu=2, workers_per_gpu=2, shuffle=False, dist=dist)
    return train_loader, val_loader


def save_checkpoint(model, optimiser, epoch: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optim_state': optimiser.state_dict(),
    }, path)


def validate(model, data_loader, metric_logger: FacadeMetricLogger, device: torch.device) -> Dict[str, float]:
    model.eval()
    metric_logger.reset()
    with torch.no_grad():
        for data in data_loader:
            img = data['img'].data[0].to(device)
            target = data['gt_semantic_seg'].data[0].squeeze(1).long().to(device)
            logits = model.forward_train(img)
            metric_logger.update(logits, target)
    return metric_logger.compute()


def train(cfg) -> None:
    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_random_seed(cfg.seed)

    train_loader, val_loader = _build_datasets(cfg)
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
        for idx, data in enumerate(train_loader, start=1):
            img = data['img'].data[0].to(device)
            target = data['gt_semantic_seg'].data[0].squeeze(1).long().to(device)
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
                logger.info(f"Epoch {epoch} Iter {idx}: loss={loss.item():.4f}")

        elapsed = time.time() - start_time
        logger.info(f"Epoch {epoch} finished: loss={epoch_loss/num_batches:.4f} time={elapsed:.1f}s")

        if epoch % cfg.training.val_interval == 0:
            metrics = validate(model, val_loader, metric_logger, device)
            logger.info(f"Validation metrics at epoch {epoch}: {metrics}")

        ckpt_path = Path(cfg.training.checkpoint_dir) / f"epoch_{epoch:03d}.pth"
        save_checkpoint(model, optimiser, epoch, ckpt_path)


if __name__ == '__main__':
    initialize(config_path="configs", version_base=None)
    cfg = compose(config_name="facade_baseline.yaml")
    train(cfg)
