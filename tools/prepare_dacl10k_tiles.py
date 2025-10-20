#!/usr/bin/env python3
"""Convert raw DACL10K annotations into tiled segmentation patches."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import cv2
except ImportError as exc:  # pragma: no cover - convenience guard
    raise ImportError(
        "OpenCV (cv2) is required for prepare_dacl10k_tiles.py. Install it via 'pip install opencv-python'."
    ) from exc

import numpy as np

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(iterable, **kwargs):
        return iterable

from utils.utils import _norm_name


@dataclass
class TileSpec:
    height: int
    width: int
    stride_y: int
    stride_x: int
    pad: bool


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create segmentation-ready tiles from raw DACL10K images and JSON annotations."
        )
    )
    parser.add_argument("images_dir", type=Path, help="Directory with original DACL10K images")
    parser.add_argument(
        "annotations_dir", type=Path, help="Directory with raw JSON annotations (per image)"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Destination directory that will contain images/ and masks/ with generated tiles",
    )
    parser.add_argument(
        "--class-mapping",
        type=Path,
        required=True,
        help=(
            "JSON mapping from original DACL10K labels to target labels."
            " Keys are raw labels, values are target class names."
            " Empty strings/null values will be ignored."
        ),
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
        help="Stride (vertical horizontal). Defaults to the tile size (no overlap).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help=(
            "Minimal fraction (0..1) of foreground pixels required to keep a tile."
            " Default: 0 (keep all tiles)."
        ),
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep tiles even if they do not meet the --min-coverage threshold.",
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
        "--pad",
        action="store_true",
        help="Pad images smaller than the tile size instead of skipping them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory (files may be overwritten).",
    )
    return parser.parse_args(argv)


def load_class_mapping(path: Path) -> Mapping[str, str]:
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
        raise ValueError("Class mapping is empty after filtering; provide at least one target class.")
    return mapping


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite and any(path.iterdir()):
            raise FileExistsError(
                f"Output directory '{path}' is not empty. Use --overwrite to write anyway."
            )
    path.mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "masks").mkdir(parents=True, exist_ok=True)


def load_annotation(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def resolve_image_path(json_path: Path, images_dir: Path) -> Path:
    base = json_path.stem
    forced_ext = None
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            forced_ext = ext
            break
    candidates: List[Path] = []
    if forced_ext:
        candidates.append(images_dir / f"{base}{forced_ext}")
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        candidates.append(images_dir / f"{base}{ext}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image for annotation '{json_path.name}' not found. Tried: {candidates[:5]}")


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


def build_mask(
    ann: Mapping[str, object],
    height: int,
    width: int,
    mapping: Mapping[str, str],
    class_to_index: Mapping[str, int],
) -> Tuple[np.ndarray, Dict[str, int]]:
    mask = np.zeros((height, width), dtype=np.uint8)
    per_class_pixels: Dict[str, int] = {cls: 0 for cls in class_to_index}
    objects = ann.get("objects", []) or []
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        if obj.get("geometryType") != "polygon":
            continue
        raw_label = str(obj.get("classTitle", ""))
        target_label = mapping.get(_norm_name(raw_label))
        if target_label is None:
            continue
        points = obj.get("points", {}) or {}
        exterior = points.get("exterior") or []
        interior = points.get("interior") or []
        poly_mask = polygon_to_mask(height, width, exterior, interior)
        if not np.any(poly_mask):
            continue
        cls_idx = class_to_index[target_label]
        mask[poly_mask > 0] = cls_idx
        per_class_pixels[target_label] += int(np.count_nonzero(poly_mask))
    return mask, per_class_pixels


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
    class_to_index: Mapping[str, int],
) -> Iterable[Tuple[int, int, np.ndarray, np.ndarray, Dict[str, float]]]:
    tile_h, tile_w = tile_spec.height, tile_spec.width
    stride_y, stride_x = tile_spec.stride_y, tile_spec.stride_x
    if tile_spec.pad:
        image = pad_if_needed(image, tile_h, tile_w)
        mask = pad_if_needed(mask, tile_h, tile_w)
    H, W = mask.shape
    if not tile_spec.pad and (H < tile_h or W < tile_w):
        return
    y_starts = compute_starts(H, tile_h, stride_y)
    x_starts = compute_starts(W, tile_w, stride_x)
    total_pixels = tile_h * tile_w
    for y0 in y_starts:
        for x0 in x_starts:
            tile_img = image[y0 : y0 + tile_h, x0 : x0 + tile_w]
            tile_mask = mask[y0 : y0 + tile_h, x0 : x0 + tile_w]
            fg_pixels = int(np.count_nonzero(tile_mask))
            coverage = fg_pixels / float(total_pixels)
            if coverage < min_coverage and not keep_empty:
                continue
            per_class_fraction: Dict[str, float] = {}
            if fg_pixels > 0:
                for cls, idx in class_to_index.items():
                    cls_pixels = int(np.count_nonzero(tile_mask == idx))
                    if cls_pixels:
                        per_class_fraction[cls] = cls_pixels / float(total_pixels)
            yield x0, y0, tile_img, tile_mask, per_class_fraction


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    tile_h, tile_w = args.tile_size
    stride_y, stride_x = args.stride if args.stride else (tile_h, tile_w)
    tile_spec = TileSpec(
        height=tile_h,
        width=tile_w,
        stride_y=stride_y,
        stride_x=stride_x,
        pad=args.pad,
    )

    class_mapping = load_class_mapping(args.class_mapping)
    unique_targets = sorted(set(class_mapping.values()))
    class_to_index = {name: idx for idx, name in enumerate(unique_targets, start=1)}

    ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    metadata: MutableMapping[str, object] = {
        "source": "DACL10K",
        "tile_size": [tile_h, tile_w],
        "stride": [stride_y, stride_x],
        "min_coverage": args.min_coverage,
        "keep_empty": bool(args.keep_empty),
        "classes": [{"name": name, "index": class_to_index[name]} for name in unique_targets],
        "total_tiles": 0,
        "per_class_tiles": {name: 0 for name in unique_targets},
        "per_class_pixels": {name: 0 for name in unique_targets},
    }

    ann_paths = sorted(p for p in args.annotations_dir.rglob("*.json"))
    if not ann_paths:
        raise FileNotFoundError(f"No JSON annotations found under {args.annotations_dir}")

    progress = tqdm(ann_paths, desc="Tiling", unit="img")
    for ann_path in progress:
        ann = load_annotation(ann_path)
        size = ann.get("size") or {}
        height = int(size.get("height") or 0)
        width = int(size.get("width") or 0)
        if height <= 0 or width <= 0:
            raise ValueError(f"Annotation '{ann_path}' lacks valid size information.")

        image_path = resolve_image_path(ann_path, args.images_dir)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image '{image_path}'.")

        mask, per_class_pixels = build_mask(ann, height, width, class_mapping, class_to_index)
        for cls_name, pixels in per_class_pixels.items():
            metadata["per_class_pixels"][cls_name] += int(pixels)

        tiles = tile_image_and_mask(
            image,
            mask,
            tile_spec,
            min_coverage=args.min_coverage,
            keep_empty=args.keep_empty,
            class_to_index=class_to_index,
        )

        image_stem = image_path.stem
        for x0, y0, tile_img, tile_mask, per_class_fraction in tiles:
            tile_name = f"{image_stem}_y{y0:04d}_x{x0:04d}"
            img_out = args.output_dir / "images" / f"{tile_name}{args.image_extension}"
            mask_out = args.output_dir / "masks" / f"{tile_name}{args.mask_extension}"
            cv2.imwrite(str(img_out), tile_img)
            cv2.imwrite(str(mask_out), tile_mask)
            metadata["total_tiles"] += 1
            for cls in per_class_fraction.keys():
                metadata["per_class_tiles"][cls] += 1

    if args.metadata:
        args.metadata.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
