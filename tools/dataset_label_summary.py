"""Summarize class labels in annotation files and export visual examples.

This helper inspects a directory of annotation files (JSON for datasets such as
DACL10K or XML for Pascal VOC-like datasets such as Prova) and reports the set
of discovered class labels together with the number of annotation files each
label appears in.  Optionally the script can also export per-class
visualisations: for every class it will render up to *N* sample images with the
corresponding objects highlighted, greatly speeding up manual validation of the
raw labels.

An optional mapping template can be written to disk to help with constructing
class remapping dictionaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "Pillow is required for dataset visualisation. Install it via 'pip install pillow'."
    ) from exc


COLORS = [
    (230, 57, 70),
    (29, 53, 87),
    (69, 123, 157),
    (168, 218, 220),
    (241, 250, 238),
    (20, 33, 61),
    (252, 163, 17),
    (2, 195, 154),
    (214, 40, 40),
    (38, 70, 83),
]


@dataclass
class Geometry:
    """A lightweight representation of an annotation geometry."""

    type: str
    points: Sequence[Tuple[float, float]]
    interior: Sequence[Sequence[Tuple[float, float]]] = ()
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class LabelSample:
    """Container describing a single labelled example."""

    annotation_path: Path
    image_path: Optional[Path]
    geometries: List[Geometry]


def _color_for_label(label: str) -> Tuple[int, int, int]:
    """Pick a semi-stable color for the given label."""

    idx = abs(hash(label)) % len(COLORS)
    return COLORS[idx]


def find_annotation_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    """Return sorted annotation file paths with the given extensions."""

    extensions = {ext.lower() for ext in extensions}
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    files.sort()
    return files


def _geometry_from_json(obj: MutableMapping[str, object]) -> Optional[Geometry]:
    gtype = str(obj.get("geometryType", "")).lower()
    points = obj.get("points") or {}
    if gtype == "polygon":
        exterior = points.get("exterior") or []
        interior = points.get("interior") or []
        if not exterior:
            return None
        interior_rings: List[Sequence[Tuple[float, float]]] = []
        for ring in interior or []:
            ring_points = [tuple(map(float, xy)) for xy in ring]
            if ring_points:
                interior_rings.append(ring_points)
        return Geometry(
            type="polygon",
            points=[tuple(map(float, xy)) for xy in exterior],
            interior=tuple(interior_rings),
        )
    if gtype == "polyline":
        exterior = points.get("exterior") or []
        if len(exterior) < 2:
            return None
        return Geometry(type="polyline", points=[tuple(map(float, xy)) for xy in exterior])
    if gtype == "rectangle":
        exterior = points.get("exterior") or []
        if not exterior:
            return None
        xs = [float(pt[0]) for pt in exterior]
        ys = [float(pt[1]) for pt in exterior]
        return Geometry(
            type="rectangle",
            points=[tuple(map(float, xy)) for xy in exterior],
            bbox=(min(xs), min(ys), max(xs), max(ys)),
        )
    return None


def parse_json_objects(path: Path) -> List[Tuple[str, Geometry]]:
    """Extract (class, geometry) pairs from a JSON annotation file."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON annotation: {path}") from exc

    objects = data.get("objects", []) or []
    result: List[Tuple[str, Geometry]] = []
    for obj in objects:
        if not isinstance(obj, MutableMapping):
            continue
        class_title = obj.get("classTitle")
        if not class_title:
            continue
        geom = _geometry_from_json(obj)
        if geom is None:
            continue
        result.append((str(class_title), geom))
    return result


def parse_xml_objects(path: Path) -> List[Tuple[str, Geometry]]:
    """Extract (class, geometry) pairs from a Pascal VOC-like XML file."""

    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse XML annotation: {path}") from exc

    root = tree.getroot()
    result: List[Tuple[str, Geometry]] = []
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None or not name_tag.text:
            continue
        bbox_tag = obj.find("bndbox")
        if bbox_tag is None:
            continue
        try:
            xmin = float(bbox_tag.findtext("xmin", "0"))
            ymin = float(bbox_tag.findtext("ymin", "0"))
            xmax = float(bbox_tag.findtext("xmax", "0"))
            ymax = float(bbox_tag.findtext("ymax", "0"))
        except ValueError:
            continue
        result.append(
            (
                name_tag.text.strip(),
                Geometry(
                    type="rectangle",
                    points=((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)),
                    bbox=(xmin, ymin, xmax, ymax),
                ),
            )
        )
    return result


def _group_geometries_by_label(pairs: Sequence[Tuple[str, Geometry]]) -> Dict[str, List[Geometry]]:
    grouped: Dict[str, List[Geometry]] = defaultdict(list)
    for label, geom in pairs:
        grouped[label].append(geom)
    return grouped


def _resolve_image_path(
    annotation_path: Path,
    annotation_root: Path,
    image_root: Path,
    image_extensions: Sequence[str],
) -> Optional[Path]:
    """Attempt to locate an image that corresponds to the annotation file."""

    try:
        rel = annotation_path.relative_to(annotation_root)
    except ValueError:
        rel = annotation_path.name

    if isinstance(rel, Path):
        stem = rel.with_suffix("")
        candidates = [image_root / stem.with_suffix(ext) for ext in image_extensions]
    else:  # rel is a string
        stem = Path(rel).with_suffix("")
        candidates = [image_root / stem.with_suffix(ext) for ext in image_extensions]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # fall back to searching alongside the annotation file
    stem = annotation_path.with_suffix("")
    for ext in image_extensions:
        candidate = stem.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def collect_label_samples(
    annotation_files: Sequence[Path],
    annotation_root: Path,
    image_root: Path,
    image_extensions: Sequence[str],
) -> Dict[str, List[LabelSample]]:
    """Collect label samples with associated geometries and image paths."""

    label_to_samples: Dict[str, List[LabelSample]] = defaultdict(list)
    image_extensions = tuple(ext if ext.startswith(".") else f".{ext}" for ext in image_extensions)

    for ann_path in annotation_files:
        suffix = ann_path.suffix.lower()
        if suffix == ".json":
            pairs = parse_json_objects(ann_path)
        elif suffix == ".xml":
            pairs = parse_xml_objects(ann_path)
        else:
            continue
        if not pairs:
            continue
        grouped = _group_geometries_by_label(pairs)
        image_path = _resolve_image_path(ann_path, annotation_root, image_root, image_extensions)
        for label, geometries in grouped.items():
            label_to_samples[label].append(
                LabelSample(annotation_path=ann_path, image_path=image_path, geometries=geometries)
            )

    for label, samples in label_to_samples.items():
        samples.sort(key=lambda sample: (
            str(sample.annotation_path),
            str(sample.image_path) if sample.image_path else "",
        ))

    return label_to_samples


def print_summary(label_to_samples: Dict[str, List[LabelSample]], samples_per_class: int) -> None:
    if not label_to_samples:
        print("No labels found.")
        return

    print("Found labels:")
    for label in sorted(label_to_samples):
        samples = label_to_samples[label]
        print(f"- {label} (occurs in {len(samples)} annotation file(s))")
        if samples_per_class > 0:
            for sample in samples[:samples_per_class]:
                location = sample.annotation_path
                suffix = " (image missing)" if not sample.image_path else ""
                print(f"  • {location}{suffix}")


def write_mapping_template(label_to_samples: Dict[str, List[LabelSample]], destination: Path) -> None:
    mapping = {label: "" for label in sorted(label_to_samples)}
    destination.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")


def _to_int_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[int, int]]:
    return [(int(round(x)), int(round(y))) for x, y in points]


def _geometry_bbox(geometry: Geometry) -> Tuple[int, int, int, int]:
    if geometry.bbox is not None:
        xmin, ymin, xmax, ymax = geometry.bbox
        return int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))
    xs = [p[0] for p in geometry.points]
    ys = [p[1] for p in geometry.points]
    return (
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    )


def _measure_text(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    """Return the width and height of the rendered text for the given font."""

    # Pillow 10 removed ``getsize`` in favour of ``getlength`` / ``getbbox``.
    getbbox = getattr(font, "getbbox", None)
    if callable(getbbox):
        left, top, right, bottom = getbbox(text)
        return int(right - left), int(bottom - top)

    getsize = getattr(font, "getsize", None)
    if callable(getsize):  # pragma: no cover - older Pillow fallback
        return getsize(text)

    # Ultimate fallback: render into a mask and inspect its size.  This branch
    # is unlikely to be hit but keeps us compatible with exotic font objects.
    mask = font.getmask(text)
    return mask.size


def _draw_geometry(
    overlay: ImageDraw.ImageDraw,
    geometry: Geometry,
    color: Tuple[int, int, int],
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    outline = color
    fill = (*color, 80)
    if geometry.type == "polygon":
        overlay.polygon(_to_int_points(geometry.points), outline=outline, fill=fill)
        for hole in geometry.interior:
            overlay.polygon(_to_int_points(hole), outline=outline, fill=(0, 0, 0, 0))
    elif geometry.type == "rectangle":
        xmin, ymin, xmax, ymax = _geometry_bbox(geometry)
        overlay.rectangle([xmin, ymin, xmax, ymax], outline=outline, fill=fill, width=3)
    elif geometry.type == "polyline":
        overlay.line(_to_int_points(geometry.points), fill=outline, width=3)

    xmin, ymin, xmax, ymax = _geometry_bbox(geometry)
    text_x = max(0, xmin)
    text_w, text_h = _measure_text(font, label)
    text_y = max(0, ymin - text_h - 2)
    bg_coords = [text_x, text_y, text_x + text_w + 4, text_y + text_h + 2]
    overlay.rectangle(bg_coords, fill=(*color, 180))
    overlay.text((text_x + 2, text_y + 1), label, fill=(255, 255, 255, 255), font=font)


def save_visualisations(
    label_to_samples: Dict[str, List[LabelSample]],
    output_dir: Path,
    samples_per_class: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for label, samples in sorted(label_to_samples.items()):
        color = _color_for_label(label)
        safe_label = "_".join(label.lower().split()) or "label"
        for idx, sample in enumerate(samples[:samples_per_class]):
            if not sample.image_path or not sample.image_path.exists():
                continue
            try:
                image = Image.open(sample.image_path).convert("RGB")
            except OSError:
                continue
            overlay_img = Image.new("RGBA", image.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_img, "RGBA")
            for geom in sample.geometries:
                _draw_geometry(draw, geom, color, label, font)
            composed = Image.alpha_composite(image.convert("RGBA"), overlay_img)
            composed = composed.convert("RGB")
            image_stem = sample.image_path.stem
            out_name = f"{safe_label}_{idx:02d}_{image_stem}.jpg"
            composed.save(output_dir / out_name, quality=95)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize labels across annotation files (JSON or XML) and optionally "
            "write a mapping template."
        )
    )
    parser.add_argument(
        "annotation_root",
        type=Path,
        help="Directory containing annotation files (.json or .xml).",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="Number of sample annotation files to display per class (default: 10).",
    )
    parser.add_argument(
        "--mapping-template",
        type=Path,
        help="Optional path to write a JSON mapping template with discovered labels.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".json", ".xml"],
        help="Annotation file extensions to consider (default: .json .xml).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        help=(
            "Root directory containing images. If omitted the script will look "
            "next to the annotation files."
        ),
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
        help=(
            "Possible image extensions to try when resolving images (default: "
            ".jpg .jpeg .png .bmp .tif .tiff)."
        ),
    )
    parser.add_argument(
        "--viz-output",
        type=Path,
        help=(
            "If provided, export per-class visual examples to this directory. "
            "Images will have highlighted annotations for quick review."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    annotation_root: Path = args.annotation_root

    if not annotation_root.exists():
        print(f"Annotation directory does not exist: {annotation_root}", file=sys.stderr)
        return 1

    annotation_files = find_annotation_files(annotation_root, args.extensions)
    if not annotation_files:
        print("No annotation files found for the specified extensions.", file=sys.stderr)
        return 1

    image_root = args.image_root if args.image_root else annotation_root
    label_to_samples = collect_label_samples(
        annotation_files,
        annotation_root=annotation_root,
        image_root=image_root,
        image_extensions=args.image_extensions,
    )

    print_summary(label_to_samples, args.samples_per_class)

    if args.viz_output:
        save_visualisations(label_to_samples, args.viz_output, args.samples_per_class)
        print(f"Saved visual previews to: {args.viz_output.resolve()}")
    else:
        print(
            "No visual previews were written (pass --viz-output <dir> to export per-class examples)."
        )
        print(f"Visualisations saved to {args.viz_output}")

    if args.mapping_template:
        write_mapping_template(label_to_samples, args.mapping_template)
        print(f"Mapping template written to {args.mapping_template}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
