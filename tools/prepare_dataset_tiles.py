#!/usr/bin/env python3
"""Unified dataset tiler that supports multiple annotation formats and augmentations.

This helper script bridges disparate facade datasets by generating
segmentation-ready crops from polygon JSON annotations (e.g. DACL10K), Pascal
VOC XML bounding boxes (e.g. Prova), or paired mask PNG datasets (e.g.
portrait_spalling_cracks). Beyond tiling it can create QA overlays, collect per
class statistics, and synthesise offline augmentations such as CutMix.

Typical workflow::

    python tools/prepare_dataset_tiles.py \
        path/to/images path/to/output \
        --dataset-type json-polygons \
        --annotations-dir path/to/annotations \
        --class-mapping tools/mappings/dacl10k_to_facseg.json

The script writes crops into ``output/images`` and masks into ``output/masks``.
If ``--overlay-dir`` is provided, blended QA previews are stored there. A JSON
manifest with aggregate statistics is written when ``--metadata`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
import sys

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "OpenCV (cv2) is required for prepare_dataset_tiles.py. Install it via 'pip install opencv-python'."
    ) from exc

import numpy as np

try:  # Pillow is optional – used for palette-based masks.
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore[assignment]

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable
# Ensure the repository root (parent of tools/) is importable when the script is
# executed directly (``python tools/prepare_dataset_tiles.py``).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.utils import _norm_name


DATASET_TYPES = {"json-polygons", "xml-bboxes", "mask-png"}
DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_MASK_EXTS = (".png", ".tif", ".tiff")


@dataclass
class TileSpec:
    height: int
    width: int
    stride_y: int
    stride_x: int
    pad: bool


@dataclass
class TileRecord:
    image: np.ndarray
    mask: np.ndarray


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile raw facade datasets into segmentation-ready crops. "
            "Supports JSON polygons (e.g. DACL10K), Pascal VOC XML bboxes (e.g. Prova), "
            "and paired mask PNGs (e.g. portrait_spalling_cracks)."
        )
    )
    parser.add_argument("images_dir", type=Path, help="Directory containing source images")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory that will contain images/ and masks/ with generated tiles",
    )
    parser.add_argument(
        "--dataset-type",
        choices=sorted(DATASET_TYPES),
        required=True,
        help="Type of annotations to consume.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        help="Directory with annotation files (.json or .xml depending on dataset type).",
    )
    parser.add_argument(
        "--annotation-extensions",
        nargs="+",
        help="Annotation extensions to search for (defaults depend on dataset type). Ignored for mask-png datasets.",
    )
    parser.add_argument(
        "--class-mapping",
        type=Path,
        help=(
            "JSON mapping from raw labels to target labels. Keys are raw class names; "
            "values are final labels. Empty strings or null values will be ignored."
        ),
    )
    parser.add_argument(
        "--mask-value-mapping",
        type=Path,
        help=(
            "JSON mapping for mask-png datasets. Keys are integer pixel values, values are target labels. "
            "Values mapped to empty strings are ignored."
        ),
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        help="Directory containing raw mask images (required for mask-png datasets).",
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTS),
        help="Candidate image extensions when resolving image files (default: common image types).",
    )
    parser.add_argument(
        "--mask-extensions",
        nargs="+",
        default=list(DEFAULT_MASK_EXTS),
        help="Candidate mask extensions when resolving mask files (default: .png .tif .tiff).",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=(1024, 1024),
        help="Tile size (height width). Default: 1024 1024",
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=2,
        metavar=("SY", "SX"),
        default=None,
        help="Stride (vertical horizontal). Defaults to tile size (no overlap).",
    )
    parser.add_argument(
        "--pad",
        action="store_true",
        help="Pad images smaller than the tile size instead of skipping them.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="Minimal fraction (0..1) of foreground pixels required to keep a tile (default: 0).",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep tiles that fail the --min-coverage requirement.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional path to write a JSON manifest with tiling statistics.",
    )
    parser.add_argument(
        "--image-extension",
        default=".jpg",
        help="Extension to use when writing tiles (default: .jpg).",
    )
    parser.add_argument(
        "--mask-extension",
        default=".png",
        help="Extension to use for mask tiles (default: .png).",
    )
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        help="Optional directory for visual overlays (image blended with mask).",
    )
    parser.add_argument(
        "--augmentations",
        nargs="*",
        default=[],
        help="Augmentations to apply per tile (choices: cutout, cutblur, cutmix).",
    )
    parser.add_argument(
        "--augmentations-per-tile",
        type=int,
        default=1,
        help="How many augmented variants to produce per tile for each augmentation type (default: 1).",
    )
    parser.add_argument(
        "--cutmix-buffer",
        type=int,
        default=32,
        help="Number of recent tiles kept in memory for CutMix partner sampling (default: 32).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory (existing files may be overwritten).",
    )
    return parser.parse_args(argv)


def ensure_output_dirs(output_dir: Path, overlay_dir: Optional[Path], overwrite: bool) -> None:
    if output_dir.exists() and not overwrite and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory '{output_dir}' is not empty. Use --overwrite to proceed."
        )
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    if overlay_dir:
        overlay_dir.mkdir(parents=True, exist_ok=True)


def _normalize_exts(exts: Iterable[str]) -> Tuple[str, ...]:
    result: List[str] = []
    for ext in exts:
        if not ext:
            continue
        result.append(ext if ext.startswith(".") else f".{ext}")
    return tuple(dict.fromkeys(result))


def _load_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_class_mapping(path: Path) -> Dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    for raw_name, target in data.items():
        if target is None:
            continue
        target_name = str(target).strip()
        if not target_name:
            continue
        mapping[_norm_name(raw_name)] = target_name
    if not mapping:
        raise ValueError(
            f"Class mapping from '{path}' is empty after filtering; provide at least one target class."
        )
    return mapping


def infer_identity_class_mapping(
    annotation_paths: Sequence[Path], dataset_type: str
) -> Dict[str, str]:
    classes: Set[str] = set()
    if dataset_type == "json-polygons":
        for ann_path in annotation_paths:
            try:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON annotation: {ann_path}") from exc
            objects = data.get("objects", []) or []
            for obj in objects:
                if not isinstance(obj, Mapping):
                    continue
                class_title = obj.get("classTitle")
                if not class_title:
                    continue
                classes.add(str(class_title).strip())
    elif dataset_type == "xml-bboxes":
        import xml.etree.ElementTree as ET

        for ann_path in annotation_paths:
            try:
                tree = ET.parse(ann_path)
            except ET.ParseError as exc:
                raise ValueError(f"Failed to parse XML annotation: {ann_path}") from exc
            root = tree.getroot()
            for obj in root.findall("object"):
                name_tag = obj.find("name")
                if name_tag is None or not name_tag.text:
                    continue
                classes.add(name_tag.text.strip())
    else:
        raise ValueError(f"Unsupported dataset type for identity mapping: {dataset_type}")

    classes = {cls for cls in classes if cls}
    if not classes:
        raise ValueError(
            "Could not infer any classes from annotations; specify --class-mapping to continue."
        )
    return {_norm_name(name): name for name in sorted(classes)}

def load_mask_value_mapping(path: Path) -> Tuple[Dict[int, str], Set[int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[int, str] = {}
    ignored_values: Set[int] = set()
    for raw_value, target in data.items():
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Mask value mapping keys must be integers, got '{raw_value}'.") from exc
        if target is None:
            ignored_values.add(value)
            continue
        target_name = str(target).strip()
        if not target_name:
            ignored_values.add(value)
            continue
        mapping[value] = target_name
    if not mapping:
        raise ValueError("Mask value mapping is empty after filtering.")
    return mapping, ignored_values


def resolve_companion_file(
    source_path: Path,
    root_dir: Path,
    extensions: Sequence[str],
    relative_to: Optional[Path] = None,
) -> Optional[Path]:
    candidates: List[Path] = []
    if relative_to is not None:
        try:
            rel = source_path.relative_to(relative_to)
            rel_stem = rel.with_suffix("")
            for ext in extensions:
                candidates.append(root_dir / rel_stem.with_suffix(ext))
        except ValueError:
            pass
    stem = source_path.stem
    candidates.extend(root_dir / f"{stem}{ext}" for ext in extensions)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for ext in extensions:
        matches = list(root_dir.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return None


def polygon_to_mask(
    height: int, width: int, exterior: Iterable[Iterable[float]], interior: Iterable[Iterable[Iterable[float]]]
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    exterior_pts = np.asarray(list(exterior), dtype=np.float32)
    if exterior_pts.size < 6:
        return mask
    exterior_pts = np.round(exterior_pts).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [exterior_pts], 1)
    for hole in interior or []:
        hole_pts = np.asarray(list(hole), dtype=np.float32)
        if hole_pts.size < 6:
            continue
        hole_pts = np.round(hole_pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [hole_pts], 0)
    return mask


def build_mask_from_json(
    ann: Mapping[str, object],
    height: int,
    width: int,
    mapping: Mapping[str, str],
    class_to_index: Mapping[str, int],
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    objects = ann.get("objects", []) or []
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        raw_label = str(obj.get("classTitle", ""))
        target_label = mapping.get(_norm_name(raw_label))
        if target_label is None:
            continue
        geom_type = str(obj.get("geometryType", "")).lower()
        points = obj.get("points", {}) or {}
        if geom_type == "polygon":
            exterior = points.get("exterior") or []
            interior = points.get("interior") or []
            poly_mask = polygon_to_mask(height, width, exterior, interior)
            if not np.any(poly_mask):
                continue
            cls_idx = class_to_index[target_label]
            mask[poly_mask > 0] = cls_idx
        elif geom_type == "rectangle":
            exterior = points.get("exterior") or []
            xs = [float(pt[0]) for pt in exterior]
            ys = [float(pt[1]) for pt in exterior]
            if not xs or not ys:
                continue
            xmin, ymin, xmax, ymax = int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))
            cls_idx = class_to_index[target_label]
            obj_mask = np.zeros_like(mask, dtype=np.uint8)
            cv2.rectangle(obj_mask, (xmin, ymin), (xmax, ymax), 1, thickness=-1)
            mask[obj_mask > 0] = cls_idx
    return mask


def build_mask_from_xml(
    ann_path: Path,
    height: int,
    width: int,
    mapping: Mapping[str, str],
    class_to_index: Mapping[str, int],
) -> np.ndarray:
    import xml.etree.ElementTree as ET

    mask = np.zeros((height, width), dtype=np.uint8)

    try:
        tree = ET.parse(ann_path)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse XML annotation: {ann_path}") from exc

    root = tree.getroot()
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None or not name_tag.text:
            continue
        raw_label = name_tag.text.strip()
        target_label = mapping.get(_norm_name(raw_label))
        if target_label is None:
            continue
        bbox_tag = obj.find("bndbox")
        if bbox_tag is None:
            continue
        try:
            xmin = int(round(float(bbox_tag.findtext("xmin", "0"))))
            ymin = int(round(float(bbox_tag.findtext("ymin", "0"))))
            xmax = int(round(float(bbox_tag.findtext("xmax", "0"))))
            ymax = int(round(float(bbox_tag.findtext("ymax", "0"))))
        except ValueError:
            continue
        cls_idx = class_to_index[target_label]
        obj_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.rectangle(obj_mask, (xmin, ymin), (xmax, ymax), 1, thickness=-1)
        mask[obj_mask > 0] = cls_idx
    return mask


def build_mask_from_png(
    mask_path: Path,
    value_to_label: Mapping[int, str],
    class_to_index: Mapping[str, int],
    ignored_values: Set[int],
) -> Tuple[np.ndarray, Set[int]]:
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if raw_mask is None:
        raise FileNotFoundError(f"Failed to read mask '{mask_path}'.")

    processed = raw_mask

    # Palette-based PNGs (P mode) are decoded by OpenCV into colourised BGR images.  That breaks
    # mappings that expect the palette indices (0..N).  When Pillow is available we can detect such
    # masks and retrieve the palette indices directly, restoring compatibility with value mappings
    # that use the encoded indices generated by ``convert_labelstudio_facade.py``.
    if Image is not None:
        try:
            with Image.open(mask_path) as pil_image:
                if pil_image.mode == "P":
                    processed = np.asarray(pil_image)
        except Exception:
            # Fall back to the OpenCV-decoded array – diagnostics later on will report unmapped values
            # if the conversion produced unexpected colours.
            processed = raw_mask

    if processed.ndim == 3:
        # Drop alpha channel if present
        if processed.shape[2] == 4:
            processed = processed[:, :, :3]
        if processed.shape[2] == 3:
            # Some datasets store masks as RGB even though they contain a single channel value.
            # If all channels match we can safely collapse to one; otherwise encode the RGB triplet
            # (re-ordered from OpenCV's BGR) as a single integer (R << 16 | G << 8 | B) so that mappings
            # can specify composite values.
            if np.array_equal(processed[:, :, 0], processed[:, :, 1]) and np.array_equal(
                processed[:, :, 0], processed[:, :, 2]
            ):
                processed = processed[:, :, 0]
            else:
                rgb = processed.astype(np.int64)
                processed = (
                    (rgb[:, :, 2] << 16)
                    | (rgb[:, :, 1] << 8)
                    | rgb[:, :, 0]
                )
        elif processed.shape[2] == 1:
            processed = processed[:, :, 0]
        else:
            raise ValueError(
                f"Unsupported mask shape {processed.shape} for '{mask_path}'. Expected 1, 3, or 4 channels."
            )

    processed = processed.astype(np.int64, copy=False)
    unique_values = np.unique(processed)
    unmatched_values: Set[int] = {
        int(value)
        for value in unique_values
        if int(value) not in value_to_label and int(value) not in ignored_values
    }

    mask = np.zeros(processed.shape, dtype=np.uint8)
    for value, target in value_to_label.items():
        cls_idx = class_to_index[target]
        matched = processed == value
        mask[matched] = cls_idx
    return mask, unmatched_values


def compute_starts(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def pad_if_needed(arr: np.ndarray, tile_h: int, tile_w: int) -> np.ndarray:
    pad_h = max(0, tile_h - arr.shape[0])
    pad_w = max(0, tile_w - arr.shape[1])
    if pad_h == 0 and pad_w == 0:
        return arr
    if arr.ndim == 3:
        return np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
    return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")


def tile_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    tile_spec: TileSpec,
    min_coverage: float,
    keep_empty: bool,
) -> Iterable[Tuple[int, int, np.ndarray, np.ndarray]]:
    tile_h, tile_w = tile_spec.height, tile_spec.width
    stride_y, stride_x = tile_spec.stride_y, tile_spec.stride_x
    if tile_spec.pad:
        image = pad_if_needed(image, tile_h, tile_w)
        mask = pad_if_needed(mask, tile_h, tile_w)
    H, W = mask.shape
    if not tile_spec.pad and (H < tile_h or W < tile_w):
        yield 0, 0, image, mask
        return
    y_starts = compute_starts(H, tile_h, stride_y)
    x_starts = compute_starts(W, tile_w, stride_x)
    for y0 in y_starts:
        for x0 in x_starts:
            tile_img = image[y0 : y0 + tile_h, x0 : x0 + tile_w]
            tile_mask = mask[y0 : y0 + tile_h, x0 : x0 + tile_w]
            total_pixels = tile_mask.size
            fg_pixels = int(np.count_nonzero(tile_mask))
            coverage = fg_pixels / float(total_pixels) if total_pixels else 0.0
            if coverage < min_coverage and not keep_empty:
                continue
            yield x0, y0, tile_img, tile_mask


def overlay_from_mask(
    image: np.ndarray,
    mask: np.ndarray,
    index_to_class: Mapping[int, str],
    alpha: float = 0.45,
) -> np.ndarray:
    colors = {}
    for idx, name in index_to_class.items():
        if idx == 0:
            continue
        rng = np.random.default_rng(abs(hash(name)) & 0xFFFFFFFF)
        colors[idx] = rng.integers(low=64, high=256, size=3, dtype=np.uint8)
    result = image.copy()
    for idx, color in colors.items():
        mask_idx = mask == idx
        if not np.any(mask_idx):
            continue
        result[mask_idx] = (1 - alpha) * result[mask_idx] + alpha * color
    return result.astype(np.uint8)


def apply_cutout(tile: TileRecord) -> TileRecord:
    image, mask = tile.image.copy(), tile.mask.copy()
    h, w = image.shape[:2]
    hole_ratio = random.uniform(0.15, 0.35)
    hole_h = max(1, int(h * hole_ratio))
    hole_w = max(1, int(w * hole_ratio))
    y0 = random.randint(0, max(0, h - hole_h))
    x0 = random.randint(0, max(0, w - hole_w))
    image[y0 : y0 + hole_h, x0 : x0 + hole_w] = 0
    return TileRecord(image=image, mask=mask)


def apply_cutblur(tile: TileRecord) -> TileRecord:
    image, mask = tile.image.copy(), tile.mask.copy()
    h, w = image.shape[:2]
    ratio = random.uniform(0.2, 0.5)
    patch_h = max(1, int(h * ratio))
    patch_w = max(1, int(w * ratio))
    y0 = random.randint(0, max(0, h - patch_h))
    x0 = random.randint(0, max(0, w - patch_w))
    patch = image[y0 : y0 + patch_h, x0 : x0 + patch_w]
    blurred = cv2.GaussianBlur(patch, (0, 0), sigmaX=3)
    image[y0 : y0 + patch_h, x0 : x0 + patch_w] = blurred
    return TileRecord(image=image, mask=mask)


def apply_cutmix(tile: TileRecord, buffer: Deque[TileRecord]) -> Optional[TileRecord]:
    candidates = [candidate for candidate in buffer if candidate.image.shape == tile.image.shape]
    if not candidates:
        return None
    partner = random.choice(candidates)
    image, mask = tile.image.copy(), tile.mask.copy()
    h, w = image.shape[:2]
    ratio = random.uniform(0.2, 0.5)
    patch_h = max(1, int(h * ratio))
    patch_w = max(1, int(w * ratio))
    y0 = random.randint(0, max(0, h - patch_h))
    x0 = random.randint(0, max(0, w - patch_w))
    image[y0 : y0 + patch_h, x0 : x0 + patch_w] = partner.image[y0 : y0 + patch_h, x0 : x0 + patch_w]
    mask[y0 : y0 + patch_h, x0 : x0 + patch_w] = partner.mask[y0 : y0 + patch_h, x0 : x0 + patch_w]
    return TileRecord(image=image, mask=mask)


def class_pixel_summary(mask: np.ndarray, class_to_index: Mapping[str, int]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for cls, idx in class_to_index.items():
        summary[cls] = int(np.count_nonzero(mask == idx))
    return summary


def update_metadata(
    metadata: MutableMapping[str, object],
    mask: np.ndarray,
    class_to_index: Mapping[str, int],
) -> None:
    metadata["total_tiles"] += 1
    per_tile_pixels = class_pixel_summary(mask, class_to_index)
    total_pixels = mask.size
    for cls, pixels in per_tile_pixels.items():
        metadata["per_class_pixels"][cls] += pixels
        if pixels > 0:
            metadata["per_class_tiles"][cls] += 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    dataset_type = args.dataset_type
    image_exts = _normalize_exts(args.image_extensions)
    mask_exts = _normalize_exts(args.mask_extensions)
    image_ext = args.image_extension if args.image_extension.startswith(".") else f".{args.image_extension}"
    mask_ext = args.mask_extension if args.mask_extension.startswith(".") else f".{args.mask_extension}"

    tile_h, tile_w = args.tile_size
    stride_y, stride_x = args.stride if args.stride else (tile_h, tile_w)
    tile_spec = TileSpec(height=tile_h, width=tile_w, stride_y=stride_y, stride_x=stride_x, pad=args.pad)

    annotations_dir: Optional[Path] = None

    if dataset_type in {"json-polygons", "xml-bboxes"}:
        if not args.annotations_dir:
            raise ValueError("--annotations-dir is required for json-polygons and xml-bboxes datasets.")
        annotations_dir = args.annotations_dir
        if not annotations_dir.exists():
            raise FileNotFoundError(f"Annotations directory '{annotations_dir}' does not exist")
        if args.annotation_extensions:
            annotation_exts = _normalize_exts(args.annotation_extensions)
        else:
            default_ext = [".json"] if dataset_type == "json-polygons" else [".xml"]
            annotation_exts = _normalize_exts(default_ext)
        ann_paths: List[Path] = []
        normalized_exts = tuple(ext.lower() for ext in annotation_exts)
        ann_paths = [
            path
            for path in sorted(annotations_dir.rglob("*"))
            if path.is_file() and any(path.name.lower().endswith(ext) for ext in normalized_exts)
        ]
        if not ann_paths:
            raise FileNotFoundError(
                f"No annotation files with extensions {annotation_exts} found under {annotations_dir}"
            )
        if args.class_mapping:
            class_mapping = load_class_mapping(args.class_mapping)
        else:
            class_mapping = infer_identity_class_mapping(ann_paths, dataset_type)
    else:  # mask-png
        if not args.masks_dir:
            raise ValueError("--masks-dir is required for mask-png datasets.")
        if not args.mask_value_mapping:
            raise ValueError("--mask-value-mapping is required for mask-png datasets.")
        ann_paths = []
        class_mapping = {}

    ensure_output_dirs(args.output_dir, args.overlay_dir, args.overwrite)

    augmentations = set(name.lower() for name in args.augmentations)
    valid_augs = {"cutout", "cutblur", "cutmix"}
    invalid = augmentations - valid_augs
    if invalid:
        raise ValueError(f"Unsupported augmentations requested: {sorted(invalid)}")

    if dataset_type == "mask-png":
        value_mapping, ignored_values = load_mask_value_mapping(args.mask_value_mapping)
        unique_targets = sorted(set(value_mapping.values()))
    else:
        unique_targets = sorted(set(class_mapping.values()))
        value_mapping = {}
        ignored_values: Set[int] = set()

    class_to_index = {name: idx for idx, name in enumerate(unique_targets, start=1)}
    index_to_class = {idx: name for name, idx in class_to_index.items()}
    metadata: MutableMapping[str, object] = {
        "dataset_type": dataset_type,
        "tile_size": [tile_h, tile_w],
        "stride": [stride_y, stride_x],
        "min_coverage": args.min_coverage,
        "keep_empty": bool(args.keep_empty),
        "pad": bool(args.pad),
        "classes": [{"name": name, "index": class_to_index[name]} for name in unique_targets],
        "total_tiles": 0,
        "per_class_tiles": {name: 0 for name in unique_targets},
        "per_class_pixels": {name: 0 for name in unique_targets},
    }

    if dataset_type == "mask-png":
        # Collect image list from images_dir
        image_paths = []
        for ext in image_exts:
            image_paths.extend(sorted(args.images_dir.rglob(f"*{ext}")))
        if not image_paths:
            raise FileNotFoundError(
                f"No image files with extensions {image_exts} found under {args.images_dir}"
            )
        unmatched_values_seen: Set[int] = set()
        unmatched_examples: Dict[str, List[int]] = {}
        empty_masks: List[str] = []
    else:
        image_paths = []

    buffer: Deque[TileRecord] = deque(maxlen=args.cutmix_buffer)
    augment_per_tile = max(1, int(args.augmentations_per_tile))

    print("[prepare_dataset_tiles] Configuration summary:")
    print(f"  dataset_type: {dataset_type}")
    print(f"  images_dir:   {args.images_dir}")
    if annotations_dir:
        print(f"  annotations:  {annotations_dir}")
    if args.masks_dir:
        print(f"  masks_dir:    {args.masks_dir}")
    print(f"  output_dir:   {args.output_dir}")
    print(f"  tile_size:    {tile_spec.height}x{tile_spec.width}")
    print(f"  stride:       {tile_spec.stride_y}x{tile_spec.stride_x}")
    print(f"  pad:          {tile_spec.pad}")
    print(f"  min_coverage: {args.min_coverage}")
    print(f"  keep_empty:   {bool(args.keep_empty)}")
    if dataset_type != "mask-png":
        if args.class_mapping:
            print(f"  class_mapping: {args.class_mapping}")
        else:
            print("  class_mapping: (identity mapping inferred from annotations)")
    if augmentations:
        print(
            "  augmentations: "
            + ", ".join(
                f"{name}×{augment_per_tile}" for name in sorted(augmentations)
            )
        )
    else:
        print("  augmentations: (none)")
    if args.overlay_dir:
        print(f"  overlays -> {args.overlay_dir}")
    if args.metadata:
        print(f"  manifest -> {args.metadata}")

    if dataset_type == "mask-png":
        iterator = tqdm(image_paths, desc="Tiling", unit="img")
        for image_path in iterator:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Failed to read image '{image_path}'.")
            mask_path = resolve_companion_file(image_path, args.masks_dir, mask_exts, relative_to=args.images_dir)
            if mask_path is None:
                raise FileNotFoundError(f"Mask for image '{image_path.name}' not found in {args.masks_dir}.")
            mask, unmatched_values = build_mask_from_png(
                mask_path,
                value_mapping,
                class_to_index,
                ignored_values,
            )
            if unmatched_values:
                unmatched_values_seen.update(unmatched_values)
                if len(unmatched_examples) < 5:
                    try:
                        rel_mask = mask_path.relative_to(args.masks_dir)
                    except ValueError:
                        rel_mask = mask_path.name
                    unmatched_examples[str(rel_mask)] = sorted(unmatched_values)[:10]
            if not np.any(mask):
                if len(empty_masks) < 5:
                    try:
                        rel_mask = mask_path.relative_to(args.masks_dir)
                    except ValueError:
                        rel_mask = mask_path.name
                    empty_masks.append(str(rel_mask))

            tiles = tile_image_and_mask(
                image,
                mask,
                tile_spec,
                min_coverage=args.min_coverage,
                keep_empty=args.keep_empty,
            )

            image_stem = image_path.stem
            for x0, y0, tile_img, tile_mask in tiles:
                tile_name = f"{image_stem}_y{y0:04d}_x{x0:04d}"
                tile_record = TileRecord(image=tile_img.copy(), mask=tile_mask.copy())
                write_tile(
                    tile_name,
                    tile_record,
                    args.output_dir,
                    args.overlay_dir,
                    image_ext,
                    mask_ext,
                    index_to_class,
                )
                update_metadata(metadata, tile_mask, class_to_index)
                buffer.append(tile_record)
                for aug_name in augmentations:
                    for aug_idx in range(augment_per_tile):
                        aug_tile = apply_augmentation(aug_name, tile_record, buffer)
                        if aug_tile is None:
                            continue
                        aug_tile_name = f"{tile_name}_{aug_name}{aug_idx}"
                        write_tile(
                            aug_tile_name,
                            aug_tile,
                            args.output_dir,
                            args.overlay_dir,
                            image_ext,
                            mask_ext,
                            index_to_class,
                        )
                        update_metadata(metadata, aug_tile.mask, class_to_index)
                        buffer.append(aug_tile)
    else:
        iterator = tqdm(ann_paths, desc="Tiling", unit="ann")
        for ann_path in iterator:
            image_path = resolve_companion_file(ann_path, args.images_dir, image_exts, relative_to=annotations_dir)
            if image_path is None:
                raise FileNotFoundError(
                    f"Image for annotation '{ann_path.name}' not found under {args.images_dir}."
                )
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Failed to read image '{image_path}'.")
            height, width = image.shape[:2]
            if dataset_type == "json-polygons":
                ann = _load_json(ann_path)
                mask = build_mask_from_json(ann, height, width, class_mapping, class_to_index)
            else:
                mask = build_mask_from_xml(ann_path, height, width, class_mapping, class_to_index)

            tiles = tile_image_and_mask(
                image,
                mask,
                tile_spec,
                min_coverage=args.min_coverage,
                keep_empty=args.keep_empty,
            )

            image_stem = image_path.stem
            for x0, y0, tile_img, tile_mask in tiles:
                tile_name = f"{image_stem}_y{y0:04d}_x{x0:04d}"
                tile_record = TileRecord(image=tile_img.copy(), mask=tile_mask.copy())
                write_tile(
                    tile_name,
                    tile_record,
                    args.output_dir,
                    args.overlay_dir,
                    image_ext,
                    mask_ext,
                    index_to_class,
                )
                update_metadata(metadata, tile_mask, class_to_index)
                buffer.append(tile_record)
                for aug_name in augmentations:
                    for aug_idx in range(augment_per_tile):
                        aug_tile = apply_augmentation(aug_name, tile_record, buffer)
                        if aug_tile is None:
                            continue
                        aug_tile_name = f"{tile_name}_{aug_name}{aug_idx}"
                        write_tile(
                            aug_tile_name,
                            aug_tile,
                            args.output_dir,
                            args.overlay_dir,
                            image_ext,
                            mask_ext,
                            index_to_class,
                        )
                        update_metadata(metadata, aug_tile.mask, class_to_index)
                        buffer.append(aug_tile)

    if args.metadata:
        args.metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print("[prepare_dataset_tiles] Finished:")
    print(f"  tiles saved:     {metadata['total_tiles']}")
    for name in unique_targets:
        print(
            f"    {name}: {metadata['per_class_tiles'][name]} tiles, "
            f"{metadata['per_class_pixels'][name]} foreground pixels"
        )
    print(f"  images -> {args.output_dir / 'images'}")
    print(f"  masks  -> {args.output_dir / 'masks'}")
    if args.overlay_dir:
        print(f"  overlays -> {args.overlay_dir}")
    if args.metadata:
        print(f"  manifest -> {args.metadata}")

    if dataset_type == "mask-png":
        if unmatched_values_seen:
            unmatched_preview = ", ".join(str(v) for v in sorted(unmatched_values_seen)[:10])
            print(
                "[prepare_dataset_tiles] Warning: encountered mask pixel values without a mapping: "
                f"{unmatched_preview}",
                file=sys.stderr,
            )
            if unmatched_examples:
                print(
                    "[prepare_dataset_tiles] Example masks with unmapped values:",
                    file=sys.stderr,
                )
                for mask_name, values in unmatched_examples.items():
                    print(
                        "    "
                        + mask_name
                        + ": "
                        + ", ".join(str(v) for v in values),
                        file=sys.stderr,
                    )
        if empty_masks:
            print(
                "[prepare_dataset_tiles] Warning: some masks contained no mapped foreground pixels:",
                file=sys.stderr,
            )
            for mask_name in dict.fromkeys(empty_masks):
                print(f"    {mask_name}", file=sys.stderr)

    empty_classes = [name for name in unique_targets if metadata["per_class_pixels"][name] == 0]
    if empty_classes:
        print(
            "[prepare_dataset_tiles] Warning: no foreground pixels were written for "
            + ", ".join(empty_classes),
            file=sys.stderr,
        )

    if metadata["total_tiles"] == 0:
        message_parts = [
            "No tiles were generated. Check that your masks contain foreground pixels,",
            "verify the --mask-value-mapping/--class-mapping, and consider lowering",
            "--min-coverage or using --keep-empty.",
        ]
        if dataset_type == "mask-png":
            if unmatched_values_seen:
                preview = ", ".join(str(v) for v in sorted(unmatched_values_seen)[:10])
                message_parts.append(
                    f"Unmapped mask pixel values were encountered: {preview}. Update --mask-value-mapping to cover them."
                )
                if unmatched_examples:
                    mask_name, values = next(iter(unmatched_examples.items()))
                    message_parts.append(
                        f"Example: {mask_name} contained values {', '.join(str(v) for v in values)}."
                    )
            elif empty_masks:
                unique_empty = list(dict.fromkeys(empty_masks))[:3]
                sample = ", ".join(unique_empty)
                message_parts.append(
                    "Masks with only background after applying the mapping (showing up to 3): "
                    + sample
                )
        raise RuntimeError(" ".join(message_parts))

    return 0


def write_tile(
    tile_name: str,
    tile: TileRecord,
    output_dir: Path,
    overlay_dir: Optional[Path],
    image_ext: str,
    mask_ext: str,
    index_to_class: Mapping[int, str],
) -> None:
    image_out = output_dir / "images" / f"{tile_name}{image_ext}"
    mask_out = output_dir / "masks" / f"{tile_name}{mask_ext}"
    cv2.imwrite(str(image_out), tile.image)
    cv2.imwrite(str(mask_out), tile.mask)
    if overlay_dir:
        overlay_img = overlay_from_mask(tile.image, tile.mask, index_to_class)
        overlay_out = overlay_dir / f"{tile_name}.jpg"
        cv2.imwrite(str(overlay_out), overlay_img)


def apply_augmentation(name: str, tile: TileRecord, buffer: Deque[TileRecord]) -> Optional[TileRecord]:
    if name == "cutout":
        return apply_cutout(tile)
    if name == "cutblur":
        return apply_cutblur(tile)
    if name == "cutmix":
        return apply_cutmix(tile, buffer)
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
