"""Convert Label Studio COCO export into mask maps for the facade dataset.

The script can either write a single split (default) or derive train/val/test
partitions from one annotation file via ``--auto-split``. When the latter is
used, each argument must follow ``name=ratio`` and the ratios need to sum (or
almost sum, accounting for floating point noise) to ``1.0``. Example:

```
python convert_labelstudio_facade.py annotations.json images/ \
    --output-root data/facade_damage --auto-split train=0.7 val=0.1 test=0.2
```

Set ``--seed`` for deterministic shuffling before the split is created.
"""

import argparse
import json
import math
import re
import os
import random
import shutil
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# Keep in sync with ``FacadeDamageDataset``.
CLASS_GROUPS = OrderedDict(
    (
        ("background", ("background",)),
        (
            "DAMAGE",
            (
                "CRACK",
                "SPALLING",
                "DELAMINATION",
                "MISSING_ELEMENT",
                "EFFLORESCENCE",
                "CORROSION",
            ),
        ),
        ("WATER_STAIN", ("WATER_STAIN",)),
        ("ORNAMENT_INTACT", ("ORNAMENT_INTACT",)),
        ("REPAIRS", ("REPAIRS",)),
        ("TEXT_OR_IMAGES", ("TEXT_OR_IMAGES",)),
    )
)

CLASS_NAMES = tuple(CLASS_GROUPS.keys())

PALETTE = [
    [0, 0, 0],
    [229, 57, 53],
    [142, 36, 170],
    [158, 158, 158],
    [78, 158, 158],
    [142, 126, 71],
]

CLASS_NAME_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
for canonical, aliases in CLASS_GROUPS.items():
    idx = CLASS_NAME_TO_INDEX[canonical]
    for alias in aliases:
        CLASS_NAME_TO_INDEX[alias] = idx


def _flatten_palette(palette: Iterable[Iterable[int]]) -> list:
    flat = []
    for color in palette:
        flat.extend(color)
    # Pad to 256 * 3 entries for P-mode images.
    flat.extend([0] * (768 - len(flat)))
    return flat

_LABEL_STUDIO_PREFIX = re.compile(r"^[0-9a-f]{8,}-")
def _sanitise_name(name: str) -> str:
    """Strip the Label Studio prefix (``<hash>-``) if present."""

    base = os.path.basename(name)
    if _LABEL_STUDIO_PREFIX.match(base):
        return base.split("-", 1)[1]
    return base


def _resolve_image_path(images_dir: Path, file_name: str, *, prefer_original: bool = True) -> Tuple[Path, str]:
    """Locate the image on disk and decide on an output file name."""

    candidates = [file_name]
    if prefer_original:
        candidates.append(_sanitise_name(file_name))

    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = images_dir / candidate
        if candidate_path.exists():
            output_name = _sanitise_name(candidate) if prefer_original else candidate
            return candidate_path, output_name

    # Fall back to a glob search – last resort.
    glob_target = _sanitise_name(file_name)
    matches = list(images_dir.glob(f"*{glob_target}"))
    if matches:
        return matches[0], _sanitise_name(matches[0].name)

    raise FileNotFoundError(f"Could not locate image for '{file_name}' in {images_dir}.")


def _build_masks(
    annotations: Iterable[dict],
    categories: Dict[int, str],
    height: int,
    width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        category_name = categories.get(ann["category_id"])
        if category_name is None:
            continue
        class_idx = CLASS_NAME_TO_INDEX[category_name]
        segmentation = ann.get("segmentation")
        if not segmentation:
            continue
        rles = mask_utils.frPyObjects(segmentation, height, width)
        if isinstance(rles, dict):
            rle = rles
        else:
            rle = mask_utils.merge(rles)
        binary_mask = mask_utils.decode(rle).astype(bool)
        mask[binary_mask] = class_idx
    return mask


def _ensure_categories(coco: MutableMapping) -> Dict[int, str]:
    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}
    missing = sorted(set(categories.values()) - set(CLASS_NAME_TO_INDEX))
    if missing:
        raise ValueError(f"Categories {missing} are not part of the predefined facade classes.")
    return categories


def _convert_single_split(
    *,
    images: Mapping[int, dict],
    anns_by_image: Mapping[int, Sequence[dict]],
    categories: Mapping[int, str],
    images_dir: Path,
    output_root: Path,
    split: str,
    selected_image_ids: Iterable[int],
) -> None:
    image_out_dir = output_root / "images" / split
    mask_out_dir = output_root / "masks" / split
    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    palette = _flatten_palette(PALETTE)

    for image_id in selected_image_ids:
        meta = images[image_id]
        src_path, output_name = _resolve_image_path(images_dir, meta["file_name"], prefer_original=True)

        dest_image = image_out_dir / output_name
        if not dest_image.exists():
            shutil.copy2(src_path, dest_image)

        width = meta.get("width")
        height = meta.get("height")
        if not width or not height:
            with Image.open(src_path) as img:
                width, height = img.size

        mask = _build_masks(anns_by_image.get(image_id, []), categories, height, width)
        mask_path = mask_out_dir / Path(output_name).with_suffix(".png")
        mask_img = Image.fromarray(mask, mode="P")
        mask_img.putpalette(palette)
        mask_img.save(mask_path)


def _parse_auto_split(values: Sequence[str]) -> List[Tuple[str, float]]:
    if not values:
        raise ValueError("--auto-split requires at least one 'name=ratio' entry.")

    splits: List[Tuple[str, float]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid auto-split spec '{value}', expected name=ratio.")
        name, ratio_str = value.split("=", 1)
        try:
            ratio = float(ratio_str)
        except ValueError as exc:  # pragma: no cover - defensive.
            raise ValueError(f"Ratio '{ratio_str}' for split '{name}' is not a float.") from exc
        if ratio <= 0:
            raise ValueError(f"Ratio for split '{name}' must be positive, got {ratio}.")
        splits.append((name, ratio))

    total_ratio = sum(r for _, r in splits)
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError(
            "Ratios in --auto-split must sum to 1.0 (±1e-3). Got " f"{total_ratio:.6f}."
        )
    return splits


def _split_indices(total: int, specs: Sequence[Tuple[str, float]], rng: random.Random) -> Dict[str, List[int]]:
    indices = list(range(total))
    rng.shuffle(indices)

    raw_counts = [ratio * total for _, ratio in specs]
    counts = [math.floor(value) for value in raw_counts]
    remainder = total - sum(counts)
    if remainder > 0:
        fractional = [value - math.floor(value) for value in raw_counts]
        order = sorted(range(len(specs)), key=lambda idx: fractional[idx], reverse=True)
        for idx in order[:remainder]:
            counts[idx] += 1

    offsets = [0]
    for count in counts[:-1]:
        offsets.append(offsets[-1] + count)

    split_mapping: Dict[str, List[int]] = {}
    for (name, _), count, offset in zip(specs, counts, offsets):
        split_mapping[name] = indices[offset : offset + count]

    return split_mapping


def convert_dataset(
    annotations: Path,
    images_dir: Path,
    output_root: Path,
    split: str,
    *,
    auto_split: Optional[Sequence[str]] = None,
    seed: Optional[int] = None,
) -> None:
    with annotations.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = _ensure_categories(coco)

    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    if auto_split:
        specs = _parse_auto_split(auto_split)
        rng = random.Random(seed)
        split_indices = _split_indices(len(images), specs, rng)
        image_ids = list(images.keys())

        for split_name, idxs in split_indices.items():
            selected_ids = [image_ids[idx] for idx in idxs]
            _convert_single_split(
                images=images,
                anns_by_image=anns_by_image,
                categories=categories,
                images_dir=images_dir,
                output_root=output_root,
                split=split_name,
                selected_image_ids=selected_ids,
            )
    else:
        _convert_single_split(
            images=images,
            anns_by_image=anns_by_image,
            categories=categories,
            images_dir=images_dir,
            output_root=output_root,
            split=split,
            selected_image_ids=images.keys(),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="Path to the Label Studio COCO JSON export")
    parser.add_argument("images", type=Path, help="Directory that contains the source images")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./data/facade_damage"),
        help="Root directory for the converted dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (e.g. train, val, test). Ignored when --auto-split is used.",
    )
    parser.add_argument(
        "--auto-split",
        nargs="+",
        help=(
            "Optional list of name=ratio pairs to derive multiple splits from one export. "
            "Ratios must sum to 1.0."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for automatic split shuffling (only used with --auto-split)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(
        args.annotations,
        args.images,
        args.output_root,
        args.split,
        auto_split=args.auto_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
