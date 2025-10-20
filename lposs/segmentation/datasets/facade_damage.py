"""Custom dataset definition for facade damage segmentation."""
from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple
from mmseg.datasets import DATASETS, CustomDataset


@DATASETS.register_module(force=True)
class FacadeDamageDataset(CustomDataset):
    """Facade damage dataset tailored for the facade baseline pipeline."""

    CLASSES = (
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

    # Default palette roughly matches the Label Studio colours shared by the user.
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

    def __init__(self, **kwargs):
        img_suffix = kwargs.pop('img_suffix', ('.png', '.jpg', '.jpeg'))
        seg_map_suffix = kwargs.pop('seg_map_suffix', ('.png', '.jpg', '.jpeg'))

        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs,
        )

        classes = kwargs.get('classes')
        if classes is not None:
            self.CLASSES = tuple(classes)
        palette = kwargs.get('palette')
        if palette is not None:
            self.PALETTE = [list(color) for color in palette]
    @staticmethod
    def _normalize_suffixes(
        suffixes: Optional[Sequence[str]],
    ) -> Optional[Tuple[str, ...]]:
        if suffixes is None:
            return None
        if isinstance(suffixes, str):  # type: ignore[arg-type]
            return (suffixes,)
        return tuple(suffixes)

    @staticmethod
    def _strip_suffix(filename: str, suffix: str) -> str:
        if not filename.endswith(suffix):
            return filename
        return filename[: -len(suffix)]

    @staticmethod
    def _select_existing_candidate(
        directory: str,
        candidates: Iterable[str],
    ) -> Optional[str]:
        for candidate in candidates:
            if os.path.exists(os.path.join(directory, candidate)):
                return candidate
        return None

    def _iter_image_files(
        self,
        img_dir: str,
        suffixes: Optional[Tuple[str, ...]],
        recursive: bool,
    ) -> Iterable[str]:
        target_suffixes = None
        if suffixes is not None:
            target_suffixes = tuple(s.lower() for s in suffixes)

        for root, _, files in os.walk(img_dir):
            rel_root = os.path.relpath(root, img_dir)
            for name in sorted(files):
                if target_suffixes is not None:
                    lower_name = name.lower()
                    if not lower_name.endswith(target_suffixes):
                        continue
                rel_path = name if rel_root == '.' else os.path.join(rel_root, name)
                yield rel_path.replace('\\', '/')
            if not recursive:
                break

    def load_annotations(
        self,
        img_dir: str,
        img_suffix: Optional[Sequence[str] | str],
        ann_dir: Optional[str],
        seg_map_suffix: Optional[Sequence[str] | str],
        split: Optional[str],
    ):
        img_suffixes = self._normalize_suffixes(img_suffix)
        seg_suffixes = self._normalize_suffixes(seg_map_suffix)

        def resolve_filename(entry: str) -> str:
            stripped = entry.strip()
            if not stripped:
                raise ValueError('Empty filename encountered in split file.')
            if img_suffixes is None:
                return stripped
            if stripped.endswith(img_suffixes):
                return stripped
            candidates = [f"{stripped}{suffix}" for suffix in img_suffixes]
            existing = self._select_existing_candidate(img_dir, candidates)
            return existing or candidates[0]

        def build_seg_map(filename: str) -> Optional[str]:
            if seg_suffixes is None:
                return None
            for img_suffix_candidate in sorted(
                (s for s in img_suffixes or [] if filename.endswith(s)),
                key=len,
                reverse=True,
            ):
                base = self._strip_suffix(filename, img_suffix_candidate)
                candidates = [f"{base}{suffix}" for suffix in seg_suffixes]
                search_dir = ann_dir or img_dir
                existing = self._select_existing_candidate(search_dir, candidates)
                if existing:
                    return existing
                if candidates:
                    return candidates[0]
            # Fallback when image suffix isn't in img_suffixes (shouldn't happen)
            base = filename.rsplit('.', 1)[0]
            candidates = [f"{base}{suffix}" for suffix in seg_suffixes]
            search_dir = ann_dir or img_dir
            existing = self._select_existing_candidate(search_dir, candidates)
            return existing or candidates[0]

        img_infos = []
        if split is not None:
            with open(split) as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    filename = resolve_filename(line)
                    info = dict(filename=filename)
                    seg_map = build_seg_map(filename)
                    if seg_map is not None:
                        info['ann'] = dict(seg_map=seg_map)
                    img_infos.append(info)
        else:
            for rel_path in self._iter_image_files(
                img_dir, img_suffixes, getattr(self, 'recursive', False)
            ):
                if img_suffixes is not None and not rel_path.endswith(img_suffixes):
                    continue
                info = dict(filename=rel_path)
                seg_map = build_seg_map(rel_path)
                if seg_map is not None:
                    info['ann'] = dict(seg_map=seg_map)
                img_infos.append(info)

        return img_infos