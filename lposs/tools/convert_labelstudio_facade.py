"""Convert Label Studio COCO export into mask maps for the facade dataset."""

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# Keep in sync with ``FacadeDamageDataset``.
CLASS_NAMES = (
    "background",
    "CRACK",
    "SPALLING",
    "DELAMINATION",
    "MISSING_ELEMENT",
    "WATER_STAIN",
    "EFFLORESCENCE",
    "CORROSION",
    "ORNAMENT_INTACT",
    "REPAIRS",
    "TEXT_OR_IMAGES",
)

PALETTE = [
    [0, 0, 0],
    [229, 57, 53],
    [30, 136, 229],
    [67, 160, 71],
    [251, 140, 0],
    [142, 36, 170],
    [253, 216, 53],
    [0, 172, 193],
    [158, 158, 158],
    [78, 158, 158],
    [142, 126, 71],
]

CLASS_NAME_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _flatten_palette(palette: Iterable[Iterable[int]]) -> list:
    flat = []
    for color in palette:
        flat.extend(color)
    # Pad to 256 * 3 entries for P-mode images.
    flat.extend([0] * (768 - len(flat)))
    return flat


def _sanitise_name(name: str) -> str:
    """Strip the Label Studio prefix (``<hash>-``) if present."""

    base = os.path.basename(name)
    if "-" in base:
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


def convert_dataset(annotations: Path, images_dir: Path, output_root: Path, split: str) -> None:
    with annotations.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco.get("categories", [])}

    missing = sorted(set(categories.values()) - set(CLASS_NAMES))
    if missing:
        raise ValueError(f"Categories {missing} are not part of the predefined facade classes.")

    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    image_out_dir = output_root / "images" / split
    mask_out_dir = output_root / "masks" / split
    image_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    palette = _flatten_palette(PALETTE)

    for image_id, meta in images.items():
        src_path, output_name = _resolve_image_path(images_dir, meta["file_name"], prefer_original=True)

        # Copy image, only if it isn't already there.
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
        help="Dataset split name (e.g. train, val, test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_dataset(args.annotations, args.images, args.output_root, args.split)


if __name__ == "__main__":
    main()
