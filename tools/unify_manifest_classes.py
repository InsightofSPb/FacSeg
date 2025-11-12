"""Normalise tile dataset manifests to the canonical FacSeg class order."""

from __future__ import annotations
import sys
import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from utils.utils import DEFAULT_CATEGORIES


_CANON_ORDER: List[str] = [DEFAULT_CATEGORIES[idx] for idx in sorted(DEFAULT_CATEGORIES.keys())]
_BACKGROUND_NAMES = {"BACKGROUND", "BG"}


def _unique(sequence: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for name in sequence:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _split_background(names: Sequence[str]) -> Tuple[List[str], List[str]]:
    front: List[str] = []
    tail: List[str] = []
    for name in names:
        if name.upper() in _BACKGROUND_NAMES:
            tail.append(name)
        else:
            front.append(name)
    return front, tail


def normalise_manifest(manifest_path: Path, *, overwrite: bool = False, output_dir: Path | None = None) -> Path:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    class_entries = data.get("classes") or []
    existing_order = [entry.get("name", "") for entry in class_entries if entry.get("name")]

    per_class_tiles = data.get("per_class_tiles") or {}
    per_class_pixels = data.get("per_class_pixels") or {}

    # Pull in any additional names referenced in per-class statistics.
    extra_from_stats = [name for name in list(per_class_tiles.keys()) + list(per_class_pixels.keys())]
    merged_order = _unique(existing_order + extra_from_stats)

    # Compose final order: canonical classes first, then any extras (background last).
    extras = [name for name in merged_order if name not in _CANON_ORDER]
    extra_front, extra_tail = _split_background(extras)
    final_order = _CANON_ORDER + extra_front
    for name in extra_tail:
        if name not in final_order:
            final_order.append(name)

    # Ensure every canonical class is present even if absent in the source manifest.
    final_order = _unique(final_order)

    new_classes = []
    new_per_class_tiles = {}
    new_per_class_pixels = {}

    for index, name in enumerate(final_order, start=1):
        new_classes.append({"name": name, "index": index})
        new_per_class_tiles[name] = int(per_class_tiles.get(name, 0))
        new_per_class_pixels[name] = int(per_class_pixels.get(name, 0))

    data["classes"] = new_classes
    data["per_class_tiles"] = new_per_class_tiles
    data["per_class_pixels"] = new_per_class_pixels

    output_path: Path
    if overwrite:
        output_path = manifest_path
    else:
        target_dir = output_dir or manifest_path.parent
        suffix = manifest_path.suffix or ".json"
        output_path = target_dir / f"{manifest_path.stem}_canon{suffix}"

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite one or more tile dataset manifests so that the class order "
            "matches the canonical FacSeg categories (CRACK, SPALLING, …)."
        )
    )
    parser.add_argument("manifests", nargs="+", type=Path, help="Path(s) to manifest.json files to normalise")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the supplied manifest(s) instead of writing <name>_canon.json alongside them.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where rewritten manifests should be stored when not using --in-place.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir and not args.output_dir.exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for manifest in args.manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest}")
        result_path = normalise_manifest(manifest, overwrite=args.in_place, output_dir=args.output_dir)
        print(f"[OK] {manifest} -> {result_path}")


if __name__ == "__main__":
    main()