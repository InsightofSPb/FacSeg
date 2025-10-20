#!/usr/bin/env python3
"""Visualise random samples from one or more tiled datasets.

The helper walks every ``images/`` and ``masks/`` pair inside the provided
directories, picks a subset of examples, and writes RGB images, raw masks, and
mask overlays to the requested output directory. The goal is to offer a quick
“sanity check” before launching training.

Example::

    python tools/preview_tiled_dataset.py \
        tiles/dacl10k/train tiles/facade_damage/train \
        --num-samples 10 --output previews/combined

The script assumes that mask files share the same stem as the corresponding
image (e.g. ``tile_0001.png`` <-> ``tile_0001.png``). Supported image formats
match those accepted by :mod:`prepare_dataset_tiles.py`.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.utils import DEFAULT_CATEGORIES, LS_PALETTE, hex_to_bgr, save_overlay


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MASK_EXTS = (".png", ".tif", ".tiff")


def _load_manifest_classes(root: Path) -> Tuple[List[str], Dict[int, str]]:
    """Return class ordering and index→name mapping for a tiles root."""

    manifest_path = root / "manifest.json"
    index_to_name: Dict[int, str] = {}

    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        for entry in data.get("classes", []) or []:
            try:
                idx = int(entry.get("index", 0))
            except (TypeError, ValueError):
                continue
            if idx <= 0:
                continue
            name = str(entry.get("name", f"class_{idx}")).strip()
            if not name:
                continue
            index_to_name[idx] = name

    if not index_to_name:
        index_to_name = {idx: name for idx, name in DEFAULT_CATEGORIES.items()}

    ordered_names = [index_to_name[idx] for idx in sorted(index_to_name.keys())]
    return ordered_names, index_to_name


def _resolve_pairs(root: Path) -> List[Tuple[Path, Path]]:
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        raise FileNotFoundError(
            f"Expected '{root}' to contain 'images/' and 'masks/' subdirectories."
        )

    mask_lookup: Dict[str, List[Path]] = defaultdict(list)
    for mask_path in masks_dir.rglob("*"):
        if mask_path.is_file() and mask_path.suffix.lower() in MASK_EXTS:
            mask_lookup[mask_path.stem].append(mask_path)

    pairs: List[Tuple[Path, Path]] = []
    for image_path in images_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        candidates = mask_lookup.get(image_path.stem, [])
        if not candidates:
            continue
        # Prefer a mask with the same relative directory if multiple exist.
        if len(candidates) == 1:
            mask_path = candidates[0]
        else:
            rel = image_path.relative_to(images_dir)
            candidate = masks_dir / rel.with_suffix("")
            match = next((m for m in candidates if m.parent == candidate.parent), None)
            mask_path = match or candidates[0]
        pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found under {root}.")
    return pairs


def _mask_to_color(mask: np.ndarray, index_to_name: Mapping[int, str]) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for idx in sorted(index_to_name.keys()):
        name = index_to_name[idx]
        color[mask == idx] = hex_to_bgr(LS_PALETTE.get(name, "#FF00FF"))
    return color


def _choose_samples(
    by_root: Mapping[str, Sequence[Tuple[Path, Path]]],
    num_samples: int,
    rng: random.Random,
) -> List[Tuple[str, Path, Path]]:
    selected: List[Tuple[str, Path, Path]] = []

    # Ensure each dataset contributes at least one example when possible.
    for root_name, pairs in by_root.items():
        if not pairs:
            continue
        sample_path, sample_mask = rng.choice(list(pairs))
        selected.append((root_name, sample_path, sample_mask))

    remaining = num_samples - len(selected)
    if remaining <= 0:
        return selected[:num_samples]

    pool: List[Tuple[str, Path, Path]] = []
    for root_name, pairs in by_root.items():
        pool.extend((root_name, img, mask) for img, mask in pairs)

    rng.shuffle(pool)
    seen = {(root, img, mask) for root, img, mask in selected}
    for entry in pool:
        if len(selected) >= num_samples:
            break
        if entry in seen:
            continue
        selected.append(entry)
    return selected


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create visual previews from tiled datasets (images/ + masks/)."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Dataset roots that contain 'images/' and 'masks/' subdirectories.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="How many samples to export in total (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("previews"),
        help="Directory to store the preview images (default: ./previews).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection (default: 42).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Overlay blending factor passed to save_overlay (default: 0.6).",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Draw a class legend on top of the overlay previews.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)

    rng = random.Random(args.seed)
    pairs_by_root: Dict[str, List[Tuple[Path, Path]]] = {}
    classes_by_root: Dict[str, List[str]] = {}
    value_maps: Dict[str, Dict[int, str]] = {}
    for root in args.roots:
        base_name = root.name or root.as_posix().replace("/", "_")
        root_name = base_name
        suffix = 2
        while root_name in pairs_by_root:
            root_name = f"{base_name}_{suffix}"
            suffix += 1

        pairs_by_root[root_name] = _resolve_pairs(root)
        class_list, index_to_name = _load_manifest_classes(root)
        classes_by_root[root_name] = class_list
        value_maps[root_name] = index_to_name

    for root_name, pairs in pairs_by_root.items():
        print(f"[preview] {root_name}: {len(pairs)} image/mask pairs found")

    total_available = sum(len(pairs) for pairs in pairs_by_root.values())
    if total_available == 0:
        raise RuntimeError("No image/mask pairs discovered across the provided roots.")

    num_samples = min(int(args.num_samples), total_available)
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    selected = _choose_samples(pairs_by_root, num_samples, rng)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"[preview] exporting {len(selected)} samples to {args.output}")

    for idx, (root_name, image_path, mask_path) in enumerate(selected, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[preview] warning: unable to read image {image_path}")
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"[preview] warning: unable to read mask {mask_path}")
            continue
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        overlay_dir = args.output / root_name
        overlay_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{idx:03d}_{image_path.stem}"
        image_out = overlay_dir / f"{base_name}_image.jpg"
        mask_out = overlay_dir / f"{base_name}_mask.png"
        overlay_out = overlay_dir / f"{base_name}_overlay.jpg"

        cv2.imwrite(str(image_out), image)
        cv2.imwrite(str(mask_out), mask)

        index_to_name = value_maps.get(root_name) or {}
        class_names = classes_by_root.get(root_name) or [
            DEFAULT_CATEGORIES[idx] for idx in sorted(DEFAULT_CATEGORIES.keys())
        ]

        color_map = _mask_to_color(mask.astype(np.uint8), index_to_name)
        save_overlay(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            color_map,
            str(overlay_out),
            alpha=float(args.alpha),
            blur=0,
            add_legend=bool(args.legend),
            class_names=class_names,
        )

    print("[preview] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
