"""Summarize class labels in annotation files for dataset mapping preparation.

This utility walks through a directory of annotation files (JSON for
datasets such as DACL10K or XML for datasets like Prova) and reports the
set of class labels that appear, how many files each label occurs in, and
provides a list of sample annotation file names per class for manual
inspection.

In addition, an optional mapping template can be written to disk to help
with constructing class remapping dictionaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, MutableMapping, Set


def find_annotation_files(root: Path, extensions: Iterable[str]) -> List[Path]:
    """Return sorted annotation file paths with the given extensions."""

    extensions = {ext.lower() for ext in extensions}
    files: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    files.sort()
    return files


def parse_json_classes(path: Path) -> Set[str]:
    """Extract class titles from a JSON annotation file."""

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON annotation: {path}") from exc

    objects = data.get("objects", [])
    classes = {obj.get("classTitle") for obj in objects if obj.get("classTitle")}
    return classes


def parse_xml_classes(path: Path) -> Set[str]:
    """Extract class names from a Pascal VOC-like XML file."""

    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise ValueError(f"Failed to parse XML annotation: {path}") from exc

    root = tree.getroot()
    classes: Set[str] = set()
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is not None and name_tag.text:
            classes.add(name_tag.text.strip())
    return classes


def summarize_annotations(annotation_files: Iterable[Path]) -> Dict[str, Set[Path]]:
    """Build a mapping of class label -> set of annotation file paths."""

    label_to_files: MutableMapping[str, Set[Path]] = defaultdict(set)
    for ann_path in annotation_files:
        parser: Callable[[Path], Set[str]]
        if ann_path.suffix.lower() == ".json":
            parser = parse_json_classes
        elif ann_path.suffix.lower() == ".xml":
            parser = parse_xml_classes
        else:
            continue

        classes = parser(ann_path)
        for class_name in classes:
            label_to_files[class_name].add(ann_path)

    return dict(label_to_files)


def print_summary(label_to_files: Dict[str, Set[Path]], samples_per_class: int) -> None:
    if not label_to_files:
        print("No labels found.")
        return

    print("Found labels:")
    for label in sorted(label_to_files):
        files = sorted(label_to_files[label])
        print(f"- {label} (occurs in {len(files)} annotation file(s))")
        if samples_per_class > 0:
            sample_files = files[:samples_per_class]
            for sample in sample_files:
                print(f"  • {sample}")


def write_mapping_template(label_to_files: Dict[str, Set[Path]], destination: Path) -> None:
    mapping = {label: "" for label in sorted(label_to_files)}
    destination.write_text(json.dumps(mapping, indent=2, ensure_ascii=False) + "\n")


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

    label_to_files = summarize_annotations(annotation_files)
    print_summary(label_to_files, args.samples_per_class)

    if args.mapping_template:
        write_mapping_template(label_to_files, args.mapping_template)
        print(f"Mapping template written to {args.mapping_template}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
