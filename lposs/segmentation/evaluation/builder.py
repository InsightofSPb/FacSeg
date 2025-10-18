import os
from typing import Any, Mapping, MutableMapping
import mmcv
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
from mmseg.datasets import build_dataloader, build_dataset
from omegaconf import DictConfig, OmegaConf

MODELS = Registry('models', parent=MMCV_MODELS)

SEGMENTORS = MODELS
from .lposs_eval import LPOSS_Infrencer


def _resolve_config(config: Any) -> Any:
    """Resolve OmegaConf containers into plain Python objects."""

    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return config


def _override_test_cfg(test_cfg: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> None:
    """Apply user supplied overrides to the dataset test configuration."""

    for key, value in overrides.items():
        if key == "config":
            continue
        if key == "data_root" and isinstance(value, str):
            # Expand environment variables and user directory references.
            test_cfg[key] = os.path.expanduser(os.path.expandvars(value))
        else:
            test_cfg[key] = value


def build_seg_dataset(config: Any):
    """Build a dataset from config.

    The configuration can either be a path to an MMCV config file or a mapping
    with a ``config`` key pointing to that file and optional overrides for the
    ``data.test`` settings (e.g. ``data_root``).
    """

    config = _resolve_config(config)
    if isinstance(config, str):
        cfg = mmcv.Config.fromfile(config)
    elif isinstance(config, Mapping):
        config_path = config.get("config")
        if not config_path:
            raise ValueError("Dataset configuration dictionary must contain a 'config' key.")
        cfg = mmcv.Config.fromfile(config_path)
        _override_test_cfg(cfg.data.test, config)
    else:
        raise TypeError(
            "Dataset configuration must be a file path or a mapping with a 'config' key."
        )

    dataset = build_dataset(cfg.data.test)
    return dataset


def build_seg_dataloader(dataset, dist=True):
    # batch size is set to 1 to handle varying image size (due to different aspect ratio)
    if dist:
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=2,
            dist=dist,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
        )
    else:
        data_loader = build_dataloader(
            dataset=dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=dist,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False,
        )

    return data_loader


def build_seg_inference(
        model,
        dataset,
        config,
        seg_config,
):
    seg_config = _resolve_config(seg_config)
    if isinstance(seg_config, Mapping):
        seg_config_path = seg_config.get("config")
        if not seg_config_path:
            raise ValueError("Segmentation config mapping must include a 'config' entry")
        seg_config = seg_config_path
    elif not isinstance(seg_config, str):
        raise TypeError(
            "Segmentation config must be a file path or mapping with a 'config' entry"
        )

    dset_cfg = mmcv.Config.fromfile(seg_config)  # dataset config
    classnames = dataset.CLASSES
    kwargs = dict()
    if hasattr(dset_cfg, "test_cfg"):
        kwargs["test_cfg"] = dset_cfg.test_cfg

    seg_model = LPOSS_Infrencer(model, config, num_classes=len(classnames), **kwargs, **config.evaluate)
    seg_model.CLASSES = dataset.CLASSES
    seg_model.PALETTE = dataset.PALETTE

    return seg_model
