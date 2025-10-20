#!/usr/bin/env python3
"""Filter raw facade datasets to only keep annotations/images with mapped classes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from tqdm.auto import tqdm

from utils.utils import _norm_name


DATASET_TYPES = {"json-polygons", "xml-bboxes", "mask-png"}
COPY_MODES = {"copy", "symlink", "hardlink"}
DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
DEFAULT_MASK_EXTS = (".png", ".tif", ".tiff")


@dataclass
class FilterStats:
    matched: int = 0
    total: int = 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter facade datasets so only annotations/images that contain mapped "
            "classes are kept. Supports JSON polygons (e.g. DACL10K), Pascal VOC "
            "XML bounding boxes (Prova) and mask PNG datasets (portrait_spalling_cracks)."
        )
    )
    parser.add_argument("images_dir", type=Path, help="Directory containing source images")
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Destination directory. The script writes filtered copies into images/ and "
            "annotations/ (plus masks/ for mask datasets)."
        ),
    )
    parser.add_argument(
        "--dataset-type",
        choices=sorted(DATASET_TYPES),
        required=True,
        help="Dataset flavour to parse",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        help="Folder with JSON or XML annotation files (required for json-polygons/xml-bboxes)",
    )
    parser.add_argument(
        "--annotation-extensions",
        nargs="+",
        help="Annotation extensions to consider (defaults depend on dataset type)",
    )
    parser.add_argument(
        "--class-mapping",
        type=Path,
        help="JSON mapping from raw labels to target labels (required for json/xml datasets)",
    )
    parser.add_argument(
        "--mask-value-mapping",
        type=Path,
        help="JSON mapping from mask pixel values to target labels (required for mask datasets)",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        help="Folder with raw masks (required for mask datasets)",
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=list(DEFAULT_IMAGE_EXTS),
        help="Candidate extensions when resolving images",
    )
    parser.add_argument(
        "--mask-extensions",
        nargs="+",
        default=list(DEFAULT_MASK_EXTS),
        help="Mask extensions to check when resolving masks",
    )
    parser.add_argument(
        "--copy-mode",
        choices=sorted(COPY_MODES),
        default="copy",
        help="How to duplicate files into the output directory (default: copy)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional path to write a JSON summary with counts",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report how many files would be copied without writing anything",
    )
    return parser.parse_args(argv)


def _normalize_exts(exts: Iterable[str]) -> Tuple[str, ...]:
    result = []
    for ext in exts:
        if not ext:
            continue
        result.append(ext if ext.startswith(".") else f".{ext}")
    return tuple(dict.fromkeys(result))


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


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
        raise ValueError("Class mapping is empty after filtering; provide at least one target class")
    return mapping


def load_mask_value_mapping(path: Path) -> Dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[int, str] = {}
    for raw_value, target in data.items():
        if target is None:
            continue
        target_name = str(target).strip()
        if not target_name:
            continue
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Mask value mapping keys must be integers, got '{raw_value}'") from exc
        mapping[value] = target_name
    if not mapping:
        raise ValueError("Mask value mapping is empty after filtering")
    return mapping


def resolve_companion_file(
    source_path: Path,
    root_dir: Path,
    extensions: Sequence[str],
    relative_to: Optional[Path] = None,
) -> Optional[Path]:
    candidates = []
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


def annotation_contains_target_json(path: Path, mapping: Mapping[str, str]) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    for obj in data.get("objects", []) or []:
        if not isinstance(obj, Mapping):
            continue
        raw_label = _norm_name(str(obj.get("classTitle", "")))
        if mapping.get(raw_label):
            return True
    return False


def annotation_contains_target_xml(path: Path, mapping: Mapping[str, str]) -> bool:
    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None or not name_tag.text:
            continue
        raw_label = _norm_name(name_tag.text)
        if mapping.get(raw_label):
            return True
    return False


def mask_contains_target(path: Path, mapping: Mapping[int, str]) -> bool:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask '{path}'")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    for value in mapping:
        if np.any(mask == value):
            return True
    return False


def copy_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        if dst.exists():
            dst.unlink()
        os.symlink(src, dst)
    elif mode == "hardlink":
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported copy mode '{mode}'")


def filter_json_or_xml(
    *,
    annotations_dir: Path,
    annotation_exts: Sequence[str],
    images_dir: Path,
    image_exts: Sequence[str],
    output_dir: Path,
    mapping: Mapping[str, str],
    copy_mode: str,
    dataset_type: str,
    dry_run: bool,
) -> FilterStats:
    ann_files = [
        path
        for ext in annotation_exts
        for path in annotations_dir.rglob(f"*{ext}")
        if path.is_file()
    ]
    ann_files.sort()
    stats = FilterStats(total=len(ann_files))
    iterator = tqdm(ann_files, desc="Scanning annotations", unit="ann")
    for ann_path in iterator:
        try:
            if dataset_type == "json-polygons":
                keep = annotation_contains_target_json(ann_path, mapping)
            else:
                keep = annotation_contains_target_xml(ann_path, mapping)
        except Exception as exc:
            raise RuntimeError(f"Failed to inspect annotation '{ann_path}'") from exc
        if not keep:
            continue
        image_path = resolve_companion_file(
            ann_path,
            images_dir,
            extensions=image_exts,
            relative_to=annotations_dir,
        )
        if image_path is None:
            raise FileNotFoundError(
                f"Image for annotation '{ann_path}' was not found under '{images_dir}'"
            )
        stats.matched += 1
        if dry_run:
            continue
        ann_out = output_dir / "annotations" / _safe_relative(ann_path, annotations_dir)
        img_out = output_dir / "images" / _safe_relative(image_path, images_dir)
        copy_file(ann_path, ann_out, copy_mode)
        copy_file(image_path, img_out, copy_mode)
    return stats


def filter_masks(
    *,
    images_dir: Path,
    masks_dir: Path,
    mask_exts: Sequence[str],
    image_exts: Sequence[str],
    output_dir: Path,
    mapping: Mapping[int, str],
    copy_mode: str,
    dry_run: bool,
) -> FilterStats:
    mask_files = [
        path
        for ext in mask_exts
        for path in masks_dir.rglob(f"*{ext}")
        if path.is_file()
    ]
    mask_files.sort()
    stats = FilterStats(total=len(mask_files))
    iterator = tqdm(mask_files, desc="Scanning masks", unit="mask")
    for mask_path in iterator:
        try:
            keep = mask_contains_target(mask_path, mapping)
        except Exception as exc:
            raise RuntimeError(f"Failed to inspect mask '{mask_path}'") from exc
        if not keep:
            continue
        stats.matched += 1
        if dry_run:
            continue
        rel_mask = _safe_relative(mask_path, masks_dir)
        mask_out = output_dir / "masks" / rel_mask
        copy_file(mask_path, mask_out, copy_mode)
        image_path = resolve_companion_file(mask_path, images_dir, image_exts, relative_to=masks_dir)
        if image_path is not None:
            img_out = output_dir / "images" / _safe_relative(image_path, images_dir)
            copy_file(image_path, img_out, copy_mode)
    return stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    image_exts = _normalize_exts(args.image_extensions)
    mask_exts = _normalize_exts(args.mask_extensions)

    if args.dataset_type in {"json-polygons", "xml-bboxes"}:
        if not args.annotations_dir:
            raise ValueError("--annotations-dir is required for json-polygons and xml-bboxes datasets")
        if not args.class_mapping:
            raise ValueError("--class-mapping is required for json-polygons and xml-bboxes datasets")
        class_mapping = load_class_mapping(args.class_mapping)
        annotation_exts = args.annotation_extensions
        if annotation_exts is None:
            annotation_exts = [".json"] if args.dataset_type == "json-polygons" else [".xml"]
        annotation_exts = _normalize_exts(annotation_exts)
        stats = filter_json_or_xml(
            annotations_dir=args.annotations_dir,
            annotation_exts=annotation_exts,
            images_dir=args.images_dir,
            image_exts=image_exts,
            output_dir=args.output_dir,
            mapping=class_mapping,
            copy_mode=args.copy_mode,
            dataset_type=args.dataset_type,
            dry_run=args.dry_run,
        )
    else:
        if not args.masks_dir:
            raise ValueError("--masks-dir is required for mask-png datasets")
        if not args.mask_value_mapping:
            raise ValueError("--mask-value-mapping is required for mask-png datasets")
        value_mapping = load_mask_value_mapping(args.mask_value_mapping)
        stats = filter_masks(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            mask_exts=mask_exts,
            image_exts=image_exts,
            output_dir=args.output_dir,
            mapping=value_mapping,
            copy_mode=args.copy_mode,
            dry_run=args.dry_run,
        )

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps({"matched": stats.matched, "total": stats.total}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print(f"Matched {stats.matched} out of {stats.total} files.")
    if args.dry_run:
        print("Dry run enabled; no files were written.")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
