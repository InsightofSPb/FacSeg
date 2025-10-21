"""Utilities for materialising tiled facade datasets for LPOSS."""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from datasets.datasets import MaskTilesIndex


@dataclass
class MaterialisedTilesDataset:
    """Result of copying tile roots into an MMCV-friendly hierarchy."""

    root: Path
    train_count: int
    val_count: int
    class_names: Sequence[str]
    per_split_counts: Mapping[str, Mapping[str, int]]


def _ensure_sequence(items: Optional[Iterable[str]]) -> List[str]:
    sequence: List[str] = []
    if not items:
        return sequence
    for item in items:
        if not item:
            continue
        path = Path(item).expanduser()
        sequence.append(str(path))
    return sequence


def _split_train_val(total: int, ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    if total <= 1:
        return list(range(total)), []
    ratio = max(0.0, min(1.0, float(ratio)))
    ids = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(ids)
    if ratio <= 0.0:
        return sorted(ids), []
    val_count = int(round(total * ratio))
    val_count = max(1, min(total - 1, val_count))
    val_ids = sorted(ids[:val_count])
    train_ids = sorted(ids[val_count:])
    return train_ids, val_ids


def _copy_file(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _derive_relative_image_path(record: Mapping[str, str], split: str) -> Path:
    dataset_name = (record.get("dataset") or "tiles").strip() or "tiles"
    relative = record.get("relative_path") or Path(record["path"]).name
    rel_path = Path(relative)
    if rel_path.is_absolute():
        parts = rel_path.parts[1:] if len(rel_path.parts) > 1 else (rel_path.name,)
        rel_path = Path(*parts)
    parts = [p for p in rel_path.parts if p not in ("", ".")]
    if split and parts and parts[0] == split:
        parts = parts[1:]
    if dataset_name and dataset_name != split:
        if not parts or parts[0] != dataset_name:
            parts = [dataset_name, *parts]
    if parts:
        return Path(*parts)
    return Path(Path(record["path"]).name)


def _write_manifest(destination: Path, class_names: Sequence[str], counts: Mapping[str, Mapping[str, int]]) -> None:
    manifest = {
        "classes": [
            {"index": idx, "name": name}
            for idx, name in enumerate(class_names, start=1)
        ],
        "keep_empty": False,
        "sources": {split: dict(counter) for split, counter in counts.items()},
    }
    manifest_path = destination / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)


def materialise_tiles_dataset(
    train_roots: Sequence[str],
    *,
    destination: Path,
    val_roots: Optional[Sequence[str]] = None,
    val_ratio: float = 0.1,
    seed: int = 42,
    class_aliases: str = "",
    expect_manifest: bool = True,
    cleanup: bool = True,
) -> MaterialisedTilesDataset:
    """Copy tiled datasets into ``destination`` with ``images/`` + ``masks/`` layout."""

    train_roots_seq = _ensure_sequence(train_roots)
    val_roots_seq = _ensure_sequence(val_roots)

    if not train_roots_seq:
        raise ValueError("At least one training tiles root must be supplied")

    destination = Path(destination).expanduser().resolve()
    if cleanup and destination.exists():
        shutil.rmtree(destination)

    for split in ("train", "val"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "masks" / split).mkdir(parents=True, exist_ok=True)

    train_index = MaskTilesIndex(train_roots_seq, class_aliases=class_aliases, expect_manifest=expect_manifest)
    train_items = list(train_index.items)

    train_records: List[MutableMapping[str, str]]
    val_records: List[MutableMapping[str, str]]

    if val_roots_seq:
        val_index = MaskTilesIndex(val_roots_seq, class_aliases=class_aliases, expect_manifest=expect_manifest)
        if list(val_index.classes) != list(train_index.classes):
            raise ValueError("Validation tiles must contain the same set of classes as training tiles")
        train_records = list(train_items)
        val_records = list(val_index.items)
    else:
        train_ids, val_ids = _split_train_val(len(train_items), val_ratio, seed)
        train_records = [train_items[idx] for idx in train_ids]
        val_records = [train_items[idx] for idx in val_ids]

    per_split_counter: Dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}

    def _process_records(records: Iterable[MutableMapping[str, str]], split: str) -> None:
        for record in records:
            rel_path = _derive_relative_image_path(record, split)
            dataset_name = (record.get("dataset") or "tiles").strip() or "tiles"
            per_split_counter[split][dataset_name] += 1
            img_src = Path(record["path"])
            mask_src = Path(record.get("mask_path", ""))
            if not mask_src:
                raise RuntimeError(f"Tile '{img_src}' is missing an associated mask path")
            mask_suffix = mask_src.suffix or ".png"
            img_dst = destination / "images" / split / rel_path
            mask_dst = destination / "masks" / split / rel_path.with_suffix(mask_suffix)
            _copy_file(img_src, img_dst)
            _copy_file(mask_src, mask_dst)

    _process_records(train_records, "train")
    _process_records(val_records, "val")

    class_names = list(train_index.classes)
    counts_mapping: Dict[str, Mapping[str, int]] = {
        split: dict(counter) for split, counter in per_split_counter.items() if counter
    }
    _write_manifest(destination, class_names, counts_mapping)

    return MaterialisedTilesDataset(
        root=destination,
        train_count=len(train_records),
        val_count=len(val_records),
        class_names=class_names,
        per_split_counts=counts_mapping,
    )

