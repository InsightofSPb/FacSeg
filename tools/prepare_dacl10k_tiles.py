#!/usr/bin/env python3
"""Compatibility wrapper that tiles DACL10K via the unified dataset tiler."""

from __future__ import annotations
import sys
import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence


# Ensure the repository root (parent of tools/) is importable when the script is
# executed directly (``python tools/prepare_dacl10k_tiles.py``).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import prepare_dataset_tiles


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create segmentation-ready tiles from raw DACL10K polygons using the "
            "unified tiler (prepare_dataset_tiles.py)."
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
            "JSON mapping from original DACL10K labels to target labels. "
            "Keys are raw labels, values are target class names. Empty strings/null values will be ignored."
        ),
    )
    parser.add_argument(
        "--annotation-extensions",
        nargs="+",
        help="Optional list of annotation extensions to search for (default: .json)",
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        help="Override the set of image extensions when resolving source files",
    )
    parser.add_argument(
        "--mask-extensions",
        nargs="+",
        help="Mask extensions are ignored for polygon datasets but forwarded for completeness",
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
        help="Stride (vertical horizontal). Defaults to the tile size (no overlap).",
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
        help="Minimal fraction (0..1) of foreground pixels required to keep a tile. Default: 0.",
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
        "--overlay-dir",
        type=Path,
        help="Optional directory for visual overlays (image blended with mask).",
    )
    parser.add_argument(
        "--augmentations",
        nargs="*",
        default=(),
        help="Offline augmentations to synthesise per tile (choices: cutmix, cutout, cutblur).",
    )
    parser.add_argument(
        "--augmentations-per-tile",
        type=int,
        default=1,
        help="How many variants to generate for each requested augmentation (default: 1).",
    )
    parser.add_argument(
        "--cutmix-buffer",
        type=int,
        default=128,
        help="Sliding-window buffer size for CutMix donor tiles (default: 128).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory (files may be overwritten).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    cli_args = [
        str(args.images_dir),
        str(args.output_dir),
        "--dataset-type",
        "json-polygons",
        "--annotations-dir",
        str(args.annotations_dir),
        "--class-mapping",
        str(args.class_mapping),
        "--tile-size",
        str(args.tile_size[0]),
        str(args.tile_size[1]),
        "--min-coverage",
        str(args.min_coverage),
        "--image-extension",
        str(args.image_extension),
        "--mask-extension",
        str(args.mask_extension),
        "--augmentations-per-tile",
        str(args.augmentations_per_tile),
        "--cutmix-buffer",
        str(args.cutmix_buffer),
    ]

    if args.stride:
        cli_args.extend(["--stride", str(args.stride[0]), str(args.stride[1])])
    if args.pad:
        cli_args.append("--pad")
    if args.keep_empty:
        cli_args.append("--keep-empty")
    if args.metadata:
        cli_args.extend(["--metadata", str(args.metadata)])
    if args.overlay_dir:
        cli_args.extend(["--overlay-dir", str(args.overlay_dir)])
    if args.overwrite:
        cli_args.append("--overwrite")
    if args.annotation_extensions:
        cli_args.extend(["--annotation-extensions", *map(str, args.annotation_extensions)])
    if args.image_extensions:
        cli_args.extend(["--image-extensions", *map(str, args.image_extensions)])
    if args.mask_extensions:
        cli_args.extend(["--mask-extensions", *map(str, args.mask_extensions)])
    if args.augmentations:
        cli_args.extend(["--augmentations", *map(str, args.augmentations)])

    return prepare_dataset_tiles.main(cli_args)


if __name__ == "__main__":
    raise SystemExit(main())

