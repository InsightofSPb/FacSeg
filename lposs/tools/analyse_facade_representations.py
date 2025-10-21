"""Visualise internal representations for facade damage models."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing aid
    import numpy as np
    import torch

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
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore
    F = None  # type: ignore
from hydra import compose, initialize_config_dir

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - hydra dependency provides it at runtime
    OmegaConf = None  # type: ignore
from PIL import Image

from lposs.helpers.logger import get_logger
from lposs.metrics.facade_metrics import FacadeMetricLogger
from lposs.models import build_model
from lposs.tools.train_facade_baseline import (
    CONFIG_DIR as TRAIN_CONFIG_DIR,
    _build_datasets,
    _unwrap_data_container,
)
from lposs.tools.infer_facade import (
    _denormalise_image,
    _ensure_logits_shape,
    _extract_dataset_metadata,
    _extract_meta,
    _get_device,
    _load_checkpoint,
    _render_visualisation,
)


DEFAULT_CONFIG_DIR = Path(TRAIN_CONFIG_DIR)


@dataclass
class ModelSpec:
    label: str
    checkpoint: Optional[Path]


@dataclass
class PromptSet:
    name: str
    overrides: Optional[Dict[str, List[str]]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare internal representations of stock and fine-tuned facade models "
            "using Grad-CAM and Integrated Gradients."
        )
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
        help="Directory where visualisations and analysis artefacts will be stored.",
    )
    parser.add_argument(
        "--finetuned-checkpoint",
        type=str,
        default=None,
        help="Path to a fine-tuned checkpoint (evaluated alongside the stock model).",
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
        help="Skip analysing the stock (pre-finetuning) model variant.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=("train", "val"),
        help="Dataset split to sample from.",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit dataset indices to analyse. Overrides --num-samples if provided.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="If --indices is not provided, analyse the first N samples of the selected split.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=("gradcam", "integrated_gradients"),
        help="Attribution methods to compute (choices: gradcam, integrated_gradients).",
    )
    parser.add_argument(
        "--ig-steps",
        type=int,
        default=24,
        help="Number of interpolation steps for Integrated Gradients.",
    )
    parser.add_argument(
        "--focus-classes",
        type=str,
        nargs="*",
        default=None,
        help="Class names to analyse. Defaults to facade damage categories.",
    )
    parser.add_argument(
        "--prompt-set",
        type=str,
        action="append",
        default=None,
        help=(
            "Additional prompt description sets to evaluate, specified as name=path_to_json. "
            "The JSON should map class labels to either a string or a list of strings."
        ),
    )
    parser.add_argument(
        "--prompt",
        type=str,
        action="append",
        default=None,
        help="Inline prompt override in the form CLASS=description. Can be passed multiple times.",
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
        raise ValueError("No models selected. Provide --finetuned-checkpoint and/or disable --skip-stock.")
    return models


def _load_config(config_name: str, config_dir: Path):
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Hydra config directory not found: {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name)


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name.lower())


def _ensure_list(value: Optional[Iterable[str]]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    return [str(v) for v in value]


def _load_prompt_file(path: Path) -> Dict[str, List[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Prompt description file not found: {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PyYAML is required to parse YAML prompt files") from exc
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("Prompt file must contain a mapping from class names to prompts")
    resolved: Dict[str, List[str]] = {}
    for key, value in data.items():
        resolved[str(key)] = _ensure_list(value)
    return resolved


def _resolve_prompt_sets(args: argparse.Namespace) -> List[PromptSet]:
    prompt_sets: List[PromptSet] = [PromptSet(name="default", overrides=None)]
    if args.prompt_set:
        for entry in args.prompt_set:
            if "=" not in entry:
                raise ValueError("Prompt sets must be specified as name=path")
            name, path_str = entry.split("=", 1)
            prompt_sets.append(
                PromptSet(name=name.strip(), overrides=_load_prompt_file(Path(path_str).expanduser().resolve()))
            )
    if args.prompt:
        overrides: Dict[str, List[str]] = {}
        for entry in args.prompt:
            if "=" not in entry:
                raise ValueError("Inline prompts must be specified as CLASS=description")
            cls, text = entry.split("=", 1)
            overrides.setdefault(cls.strip(), []).append(text.strip())
        prompt_sets.append(PromptSet(name="inline", overrides=overrides))
    return prompt_sets


def _resolve_focus_classes(
    focus_classes: Optional[Iterable[str]],
    class_names: Sequence[str],
) -> Tuple[List[str], List[int]]:
    lookup = {name.upper(): idx for idx, name in enumerate(class_names)}
    if focus_classes is None:
        focus_classes = FacadeMetricLogger.DAMAGE_CLASSES
    resolved_labels: List[str] = []
    resolved_indices: List[int] = []
    for label in focus_classes:
        key = label.upper()
        if key not in lookup:
            raise KeyError(f"Focus class '{label}' not present in dataset classes")
        resolved_labels.append(class_names[lookup[key]])
        resolved_indices.append(lookup[key])
    return resolved_labels, resolved_indices


def _gather_samples(
    loader,
    indices: Optional[Sequence[int]],
    limit: Optional[int],
) -> List[Dict]:
    desired = None if indices is None else sorted(set(int(i) for i in indices))
    collected: List[Dict] = []
    target_count = len(desired) if desired is not None else (limit or 0)
    current_index = 0
    for batch in loader:
        if desired is not None and current_index not in desired:
            current_index += 1
            continue

        img_tensor = _unwrap_data_container(batch["img"])
        if isinstance(img_tensor, torch.Tensor) and img_tensor.ndim == 4:
            if img_tensor.shape[0] != 1:
                raise ValueError(
                    "Analysis pipeline expects DataLoader batches of size 1; received batch with "
                    f"shape {tuple(img_tensor.shape)}"
                )
            img_tensor = img_tensor[0]
        img_tensor = img_tensor.detach().cpu()

        target = None
        if "gt_semantic_seg" in batch:
            target_tensor = _unwrap_data_container(batch["gt_semantic_seg"])
            if isinstance(target_tensor, torch.Tensor) and target_tensor.ndim == 4:
                if target_tensor.shape[0] != 1:
                    raise ValueError(
                        "Analysis pipeline expects DataLoader batches of size 1; received target with "
                        f"shape {tuple(target_tensor.shape)}"
                    )
                target_tensor = target_tensor[0]
            target = target_tensor.detach().cpu()

        meta = _extract_meta(batch)

        collected.append({"index": current_index, "img": img_tensor, "target": target, "meta": meta})
        current_index += 1

        if desired is not None and len(collected) >= len(desired):
            break
        if desired is None and limit is not None and len(collected) >= limit:
            break
    return collected


def _build_model(
    cfg,
    class_names: Sequence[str],
    device: "torch.device",
    model_spec: ModelSpec,
    prompt_overrides: Optional[Dict[str, List[str]]],
    logger,
):
    model = build_model(cfg.model, class_names=class_names)
    decode_head = getattr(model.clip_backbone, "decode_head", None)
    if prompt_overrides:
        if decode_head is None or not hasattr(decode_head, "configure_prompts"):
            raise AttributeError("Model decode head does not support prompt configuration")
        decode_head.configure_prompts(prompt_descriptions=prompt_overrides)
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
        logger.info("[%s] Using freshly initialised weights", model_spec.label)
    return model


def _class_logits_from_feat(model, feat: "torch.Tensor") -> "torch.Tensor":
    embeddings = model.clip_backbone.decode_head.class_embeddings
    return F.conv2d(feat, embeddings[:, :, None, None])


def _forward_decode_head_features(
    model,
    image: "torch.Tensor",
) -> "torch.Tensor":
    """Run the decode head while keeping gradients w.r.t. the input image.

    Some MaskCLIP variants decorate ``extract_feat`` with ``torch.no_grad`` which
    detaches the autograd graph when calling ``model.clip_backbone`` directly.
    Integrated Gradients needs the gradient to flow back to the input image, so
    we try to call the undecorated implementation when available.

    Args:
        model: The high-level model that exposes ``clip_backbone``.
        image: A 4D tensor (N, C, H, W) whose gradients should be tracked.

    Returns:
        The projected feature map produced by the decode head.
    """

    clip_backbone = getattr(model, "clip_backbone", None)
    if clip_backbone is None:
        raise AttributeError("Model does not expose a clip_backbone attribute")

    extract_feat = getattr(clip_backbone, "extract_feat", None)
    decode_head = getattr(clip_backbone, "decode_head", None)
    clip_transform = getattr(clip_backbone, "clip_T", None)

    if (
        extract_feat is not None
        and hasattr(extract_feat, "__wrapped__")
        and decode_head is not None
        and clip_transform is not None
    ):
        raw_extract = extract_feat.__wrapped__
        normalised = clip_transform(image)
        features = raw_extract(clip_backbone, normalised)
        try:
            outputs = decode_head(features, return_feat=True)
        except TypeError:
            outputs = decode_head(features)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            _, feat = outputs
        elif isinstance(outputs, tuple) and outputs:
            feat = outputs[-1]
        elif isinstance(outputs, torch.Tensor):
            feat = outputs
        else:
            raise RuntimeError(
                "Decode head did not return feature maps required for attribution"
            )
        return feat

    outputs = clip_backbone(image, return_feat=True)
    if isinstance(outputs, tuple) and outputs:
        return outputs[-1]
    raise RuntimeError(
        "clip_backbone did not return feature maps required for attribution"
    )


def _compute_gradcam(
    model,
    image: "torch.Tensor",
    feat: "torch.Tensor",
    logits: "torch.Tensor",
    class_idx: int,
) -> "np.ndarray":
    model.zero_grad()
    score = logits[:, class_idx].mean()
    grads = torch.autograd.grad(score, feat, retain_graph=True)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feat).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    heatmap = cam.squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0.0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def _compute_integrated_gradients(
    model,
    image: "torch.Tensor",
    class_idx: int,
    steps: int,
) -> "np.ndarray":
    model.zero_grad(set_to_none=True)
    baseline = torch.zeros_like(image)
    scaled_inputs = torch.linspace(0.0, 1.0, steps + 1, device=image.device, dtype=image.dtype)
    total_grad = torch.zeros_like(image)

    with torch.enable_grad():
        for alpha in scaled_inputs:
            interpolated = baseline + alpha * (image - baseline)
            interpolated = interpolated.unsqueeze(0)
            interpolated.requires_grad_(True)
            feat = _forward_decode_head_features(model, interpolated)
            logits = _class_logits_from_feat(model, feat)
            score = logits[:, class_idx].mean()
            try:
                grad = torch.autograd.grad(score, interpolated, retain_graph=False)[0]
            except RuntimeError as exc:
                raise RuntimeError(
                    "Integrated Gradients requires the model to propagate gradients to the input. "
                    "Ensure the backbone does not disable autograd or drop integrated gradients from --methods."
                ) from exc
            total_grad += grad.squeeze(0)

    avg_grad = total_grad / (steps + 1)
    attribution = (image - baseline) * avg_grad
    heatmap = attribution.abs().sum(dim=0).detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0.0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def _save_heatmap_overlay(
    image: "np.ndarray",
    heatmap: "np.ndarray",
    output_path: Path,
    *,
    title: Optional[str] = None,
    cmap: str = "magma",
    alpha: float = 0.6,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required to render attribution overlays. Please install it to run the analysis script."
        )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    ax.imshow(heatmap, cmap=cmap, alpha=alpha)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _save_segmentation(
    image: "np.ndarray",
    mask: "np.ndarray",
    palette: "np.ndarray",
    output_path: Path,
) -> None:
    combined, overlay = _render_visualisation(image, mask, palette, save_overlay=True)
    base_name = output_path.name
    if output_path.suffix:
        base_name = output_path.stem
    Image.fromarray(image).save(output_path.parent / f"{base_name}_input.png")
    Image.fromarray(combined).save(output_path.parent / f"{base_name}_prediction.png")
    if overlay is not None:
        Image.fromarray(overlay).save(output_path.parent / f"{base_name}_overlay.png")


def main() -> None:
    args = _parse_args()
    config_dir = Path(args.config_dir).resolve()
    cfg = _load_config(args.config_name, config_dir)
    if OmegaConf is not None and OmegaConf.is_config(cfg):
        OmegaConf.set_struct(cfg, False)
    models = _resolve_models(args)
    prompt_sets = _resolve_prompt_sets(args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        cfg.output = str(output_dir)
    except AttributeError:
        if isinstance(cfg, dict):
            cfg["output"] = str(output_dir)
        else:
            raise

    if plt is None:
        raise RuntimeError(
            "matplotlib is required for attribution visualisation. Install it with 'pip install matplotlib'."
        )

    if np is None:
        raise RuntimeError("numpy is required for representation analysis. Install it with 'pip install numpy'.")
    if torch is None or F is None:
        raise RuntimeError("PyTorch is required for representation analysis. Install it before running this script.")

    device = _get_device(args.device)
    logger = get_logger(cfg)
    logger.info("Analysing models on device %s", device)

    train_loader, val_loader, mmseg_cfg = _build_datasets(
        cfg,
        include_repeats=False,
        train_shuffle=False,
    )

    loaders = {"train": train_loader, "val": val_loader}
    loader = loaders[args.split]
    class_names, palette = _extract_dataset_metadata(train_loader.dataset)
    focus_labels, focus_indices = _resolve_focus_classes(args.focus_classes, class_names)
    palette_arr = np.asarray(palette, dtype=np.uint8)

    img_norm_cfg = mmseg_cfg.get("img_norm_cfg", {})
    mean = img_norm_cfg.get("mean", [123.675, 116.28, 103.53])
    std = img_norm_cfg.get("std", [58.395, 57.12, 57.375])

    if args.indices:
        selected_indices = args.indices
        logger.info("Collecting %d explicitly requested samples", len(selected_indices))
    else:
        selected_indices = None
        logger.info("Collecting the first %d samples from the %s split", args.num_samples, args.split)

    samples = _gather_samples(loader, selected_indices, args.num_samples if selected_indices is None else None)
    if not samples:
        raise RuntimeError("No samples collected for analysis")

    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for prompt_set in prompt_sets:
        logger.info("Evaluating prompt set '%s'", prompt_set.name)
        prompt_dir = output_dir / _sanitize_name(prompt_set.name)
        prompt_dir.mkdir(parents=True, exist_ok=True)

        models_for_prompt: Dict[str, torch.nn.Module] = {}
        for spec in models:
            models_for_prompt[spec.label] = _build_model(
                cfg,
                class_names,
                device,
                spec,
                prompt_set.overrides,
                logger,
            )

        for sample in samples:
            sample_dir = prompt_dir / f"sample_{sample['index']:05d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            image_np = _denormalise_image(sample["img"], mean, std)
            Image.fromarray(image_np).save(sample_dir / "input.png")

            target = sample.get("target")
            prompt_summary = summary.setdefault(prompt_set.name, {})
            sample_summary = prompt_summary.setdefault(str(sample["index"]), {})
            meta = sample.get("meta") or {}
            if "metadata" not in sample_summary:
                sample_summary["metadata"] = {
                    "filename": meta.get("ori_filename") or meta.get("filename"),
                    "dataset_index": int(sample["index"]),
                }

            for label, model in models_for_prompt.items():
                model_summary = sample_summary.setdefault(label, {})

                image_tensor = sample["img"].unsqueeze(0).to(device)

                with torch.no_grad():
                    seg_probs, _ = model.clip_backbone(image_tensor, return_feat=True)
                    seg_probs = _ensure_logits_shape(seg_probs, image_tensor.shape[-2:])
                    mask = seg_probs.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.int64)
                    probs_np = seg_probs.squeeze(0).detach().cpu().numpy()

                pred_counts = {
                    class_names[idx]: int((mask == idx).sum()) for idx in range(len(class_names))
                }
                model_summary.update({f"pixels_{name}": count for name, count in pred_counts.items()})
                for label_name, class_idx in zip(focus_labels, focus_indices):
                    model_summary[f"prob_{label_name}_mean"] = float(probs_np[class_idx].mean())
                    model_summary[f"prob_{label_name}_max"] = float(probs_np[class_idx].max())

                seg_path = sample_dir / f"{label}_segmentation"
                _save_segmentation(image_np, mask, palette_arr, seg_path)

                if target is not None:
                    target_path = sample_dir / "ground_truth.png"
                    if not target_path.exists():
                        if hasattr(target, "detach"):
                            target_np = target.detach().cpu().numpy()
                        else:
                            target_np = np.asarray(target)
                        target_np = target_np.astype(np.int64, copy=False)
                        if target_np.ndim == 0:
                            raise ValueError(
                                "Expected target mask to have at least one spatial dimension, "
                                f"got shape {target_np.shape}"
                            )
                        if target_np.ndim == 1:
                            mask_np = target_np.reshape(1, target_np.shape[0])
                        else:
                            height, width = target_np.shape[-2:]
                            mask_np = target_np.reshape(height, width)
                        colour_mask = palette_arr[mask_np]
                        Image.fromarray(colour_mask).save(target_path)

                methods = {m.lower() for m in args.methods}

                if "gradcam" in methods:
                    image_tensor_gc = sample["img"].unsqueeze(0).to(device)
                    image_tensor_gc.requires_grad_(True)
                    _, feat_gc = model.clip_backbone(image_tensor_gc, return_feat=True)
                    logits_gc = _class_logits_from_feat(model, feat_gc)
                    for label_name, class_idx in zip(focus_labels, focus_indices):
                        heatmap = _compute_gradcam(model, image_tensor_gc, feat_gc, logits_gc, class_idx)
                        heat_path = sample_dir / f"{label}_gradcam_{_sanitize_name(label_name)}.png"
                        _save_heatmap_overlay(
                            image_np,
                            heatmap,
                            heat_path,
                            title=f"{label} Grad-CAM: {label_name}",
                        )
                        model_summary[f"gradcam_{label_name}_mean"] = float(np.mean(heatmap))

                if "integrated_gradients" in methods or "ig" in methods:
                    ig_steps = max(1, int(args.ig_steps))
                    image_tensor_ig = sample["img"].to(device)
                    for label_name, class_idx in zip(focus_labels, focus_indices):
                        heatmap = _compute_integrated_gradients(
                            model,
                            image_tensor_ig,
                            class_idx,
                            steps=ig_steps,
                        )
                        heat_path = sample_dir / f"{label}_ig_{_sanitize_name(label_name)}.png"
                        _save_heatmap_overlay(
                            image_np,
                            heatmap,
                            heat_path,
                            title=f"{label} Integrated Gradients: {label_name}",
                            cmap="viridis",
                        )
                        model_summary[f"ig_{label_name}_mean"] = float(np.mean(heatmap))

            # avoid holding tensors on GPU between samples
            torch.cuda.empty_cache()

        for model in models_for_prompt.values():
            del model
        torch.cuda.empty_cache()

    summary_path = output_dir / "analysis_summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    logger.info("Written analysis summary to %s", summary_path)


if __name__ == "__main__":
    main()

