import inspect
import os, json, random, uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import cv2
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.utils import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_CATEGORIES,
    LS_PALETTE,
    _norm_name,
    hex_to_bgr,
    load_aliases_json,
    polygons_to_mask,
    remap_id_to_canonical,
    save_overlay,
)


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_MASK_EXTS = (".png", ".tif", ".tiff")
_CANON_LOOKUP = {_norm_name(v): v for v in DEFAULT_CATEGORIES.values()}


def _merge_dicts(base: dict, override: dict) -> dict:
    if not override:
        return dict(base)
    result = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def _ensure_tuple(value, default=None, *, clamp_len: int = 0):
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        tup = tuple(value)
    else:
        tup = (value, value)
    if clamp_len and tup and len(tup) != clamp_len:
        if len(tup) > clamp_len:
            tup = tup[:clamp_len]
        else:
            tup = tuple(list(tup) + [tup[-1]] * (clamp_len - len(tup)))
    return tup


DEFAULT_PREPARE_AUG = {
    "train": {
        "hflip_p": 0.5,
        "vflip_p": 0.1,
        "rot90_p": 0.2,
        "perspective": {
            "scale": (0.05, 0.12),
            "keep_size": True,
            "pad_mode": cv2.BORDER_REFLECT_101,
            "p": 0.15,
        },
        "color_jitter": {
            "p": 0.35,
            "brightness": 0.3,
            "contrast": 0.3,
            "saturation": 0.2,
            "hue": 0.08,
        },
        "random_brightness_contrast": {
            "p": 0.3,
            "brightness_limit": 0.3,
            "contrast_limit": 0.3,
        },
        "random_gamma": {
            "gamma_limit": (90, 110),
            "p": 0.3,
        },
        "motion_blur": {
            "blur_limit": (3, 7),
            "p": 0.12,
        },
        "gaussian_blur": {
            "blur_limit": (3, 5),
            "p": 0.1,
        },
    },
    "val": {},
    "test": {
        "motion_blur": {
            "blur_limit": (3, 5),
            "p": 0.05,
        },
        "gaussian_blur": {
            "blur_limit": (3, 3),
            "p": 0.05,
        },
    },
}


DEFAULT_DATASET_AUG = {
    "train": {
        "hflip_p": 0.5,
        "vflip_p": 0.1,
        "perspective": {
            "scale": (0.05, 0.12),
            "keep_size": True,
            "pad_mode": cv2.BORDER_REFLECT_101,
            "p": 0.18,
        },
        "shift_scale_rotate": {
            "shift_limit": 0.05,
            "scale_limit": 0.2,
            "rotate_limit": 15,
            "p": 0.5,
            "border_mode": cv2.BORDER_REFLECT_101,
        },
        "motion_blur": {
            "blur_limit": (3, 7),
            "p": 0.1,
        },
        "color_aug_p": 0.4,
        "color_jitter": {
            "brightness": 0.3,
            "contrast": 0.3,
            "saturation": 0.25,
            "hue": 0.08,
        },
        "random_brightness_contrast": {
            "brightness_limit": 0.3,
            "contrast_limit": 0.3,
        },
        "hsv": {
            "hue_shift_limit": 10,
            "sat_shift_limit": 15,
            "val_shift_limit": 10,
        },
        "rgb_shift": {
            "r_shift_limit": 10,
            "g_shift_limit": 10,
            "b_shift_limit": 10,
        },
        "random_gamma": {
            "gamma_limit": (90, 110),
        },
    },
    "val": {},
}


def _resolve_coco_classes(coco: dict, aliases_tbl=None):
    """Return canonical class order and mapping from COCO ids -> canonical names."""
    aliases_tbl = aliases_tbl or {}

    categories = coco.get("categories", []) or []
    id_to_name = {c["id"]: c.get("name", f"cat_{c['id']}") for c in categories}

    norm2canon = {_norm_name(v): v for v in DEFAULT_CATEGORIES.values()}

    observed = []  # (cid, canonical_name)
    coco_id_to_canon = {}

    for cid, raw_name in id_to_name.items():
        norm = _norm_name(raw_name)
        if norm in aliases_tbl:
            canon_norm = aliases_tbl[norm]
            canon = norm2canon.get(canon_norm, canon_norm)
        elif norm in norm2canon:
            canon = norm2canon[norm]
        else:
            canon = raw_name.strip() or f"cat_{cid}"
        coco_id_to_canon[cid] = canon
        observed.append((cid, canon))

    canon_order = []
    present = {name for _, name in observed}
    for key in sorted(DEFAULT_CATEGORIES.keys()):
        nm = DEFAULT_CATEGORIES[key]
        if nm in present and nm not in canon_order:
            canon_order.append(nm)
    for _, name in observed:
        if name not in canon_order:
            canon_order.append(name)

    return canon_order, coco_id_to_canon


def load_coco_class_order(coco_json: str, class_aliases: str = ""):
    """Load ordered canonical class list from a COCO json file."""
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)
    aliases_tbl = load_aliases_json(class_aliases) if class_aliases else {}
    classes, _ = _resolve_coco_classes(coco, aliases_tbl)
    return classes

# ----------------------------- helpers -----------------------------

def _resolve_image_path(images_dir, im):
    """
    Расширенный резолвер пути:
    - images_dir/file_name
    - images_dir/basename(path) или абсолютный path
    - обрезка префикса до первого '-' у file_name (xxxx-IMG_123.jpg -> IMG_123.jpg)
    - в крайнем случае — os.walk по каталогу
    """
    cands = []

    fname = im.get("file_name", "") or ""
    if fname:
        cands.append(os.path.join(images_dir, fname))

    jpath = im.get("path", "") or ""
    if jpath:
        if os.path.isabs(jpath) and os.path.isfile(jpath):
            return jpath
        cands.append(os.path.join(images_dir, os.path.basename(jpath)))

    if fname and "-" in fname:
        tail = fname.split("-", 1)[1]
        cands.append(os.path.join(images_dir, tail))

    for p in cands:
        if os.path.isfile(p):
            return p

    base = os.path.basename(fname) if fname else ""
    tail = base.split("-", 1)[-1] if base else ""
    try:
        for root, _, files in os.walk(images_dir):
            for fn in files:
                low = fn.lower()
                if (base and low == base.lower()) or (tail and low == tail.lower()):
                    return os.path.join(root, fn)
    except Exception:
        pass
    return None

# ----------------------------- Segmentation index -----------------------------

class SegIndex:
    """
    Индекс сегментационного датасета из COCO-like json.
    Преобразует классы через алиасы -> канон (DEFAULT_CATEGORIES).
    Собирает целевую маску [0..C], где 0=фон, 1..C — канонические классы.
    """
    def __init__(self, images_dir: str, coco_json: str, class_aliases: str = ""):
        self.images_dir = images_dir
        self.coco_json = coco_json
        self.aliases_tbl = load_aliases_json(class_aliases) if class_aliases else {}

        with open(coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        canon_order, coco_id_to_canon = _resolve_coco_classes(coco, self.aliases_tbl)
        self.class_ids = [cid for cid, nm in coco_id_to_canon.items() if nm in set(canon_order)]
        self.classes = canon_order
        self.C = len(self.classes)
        self.num_classes = self.C  # совместимость со старым кодом

        # сгруппируем аннотации по изображению
        anns_by_img = defaultdict(list)
        for a in coco.get("annotations", []):
            if a.get("iscrowd", 0) == 1:
                continue
            if a.get("category_id") not in coco_id_to_canon:
                continue
            anns_by_img[a["image_id"]].append(a)

        # только изображения, по которым есть аннотации
        self.images = [im for im in coco.get("images", []) if im["id"] in anns_by_img]
        assert len(self.images) > 0, "No images with segmentation masks."

        self.items = []
        canon_to_idx = {nm: i + 1 for i, nm in enumerate(self.classes)}  # 1..C
        self.class_to_index = dict(canon_to_idx)

        for im in tqdm(self.images, desc="Index masks", unit="img"):
            H, W = int(im["height"]), int(im["width"])
            path = _resolve_image_path(images_dir, im)
            if path is None:
                continue

            mask = np.zeros((H, W), np.uint8)
            for a in anns_by_img[im["id"]]:
                canon = coco_id_to_canon[a["category_id"]]
                idx = canon_to_idx[canon]
                seg = a.get("segmentation", [])
                if not seg:
                    continue
                m = polygons_to_mask(seg, H, W, value=1)
                mask[m > 0] = idx

            if mask.sum() == 0:
                continue

            self.items.append({
                "image_id": im["id"],
                "path": path,
                "mask": mask,
            })

        assert len(self.items) > 0, "No valid items built (check paths and annotations)."
        self.meta_by_path: Dict[str, Dict[str, str]] = {}
        for rec in self.items:
            try:
                rel = os.path.relpath(rec["path"], images_dir)
            except Exception:
                rel = os.path.basename(rec["path"]) or rec["path"]
            self.meta_by_path[rec["path"]] = {
                "relative_path": rel.replace("\\", "/"),
                "dataset": Path(images_dir).name or ".",
            }

    def split_by_images(self, val_ratio=0.25, seed=42):
        n = len(self.items)
        ids = list(range(n))
        random.Random(seed).shuffle(ids)
        val_n = max(1, int(round(val_ratio * n)))
        va_idx = sorted(ids[:val_n])
        tr_idx = sorted(ids[val_n:])
        return tr_idx, va_idx


def _canonicalise_class_name(name: str, aliases_tbl: Dict[str, str]) -> str:
    norm = _norm_name(name)
    mapped = aliases_tbl.get(norm, norm)
    return _CANON_LOOKUP.get(mapped, name.strip() or f"class_{mapped}")


def _iter_tile_pairs(images_dir: Path, masks_dir: Path):
    mask_lookup: Dict[str, list] = defaultdict(list)
    for mask_path in masks_dir.rglob("*"):
        if mask_path.is_file() and mask_path.suffix.lower() in _MASK_EXTS:
            mask_lookup[mask_path.stem].append(mask_path)

    for image_path in images_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        candidates = mask_lookup.get(image_path.stem, [])
        if not candidates:
            continue
        if len(candidates) == 1:
            mask_path = candidates[0]
        else:
            try:
                rel = image_path.relative_to(images_dir)
            except ValueError:
                rel = Path(image_path.name)
            target_parent = masks_dir / rel.parent
            match = next((m for m in candidates if m.parent == target_parent), None)
            mask_path = match or candidates[0]
        yield image_path, mask_path


class MaskTilesIndex:
    """Index tiled datasets that provide ``images/`` + ``masks/`` directories."""

    def __init__(self, roots: Sequence[str], class_aliases: str = "", *, expect_manifest: bool = True):
        root_paths = [Path(r).expanduser() for r in (roots or []) if r]
        if not root_paths:
            raise ValueError("At least one tiles root must be supplied")

        aliases_tbl = load_aliases_json(class_aliases) if class_aliases else {}
        aliases_tbl = {k: v for k, v in aliases_tbl.items()}

        records = []
        seen_classes = set()
        root_infos = {}
        dropped_empty = Counter()

        for root in root_paths:
            if not root.exists():
                raise FileNotFoundError(f"Tiles root '{root}' does not exist")
            images_dir = root / "images"
            masks_dir = root / "masks"
            if not images_dir.exists() or not masks_dir.exists():
                raise FileNotFoundError(
                    f"Tiles root '{root}' must contain 'images/' and 'masks/' subdirectories"
                )

            manifest_path = root / "manifest.json"
            value_map: Dict[int, str] = {}
            keep_empty = False
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for entry in manifest.get("classes", []):
                    try:
                        idx = int(entry.get("index", 0))
                    except (TypeError, ValueError):
                        continue
                    if idx <= 0:
                        continue
                    raw_name = entry.get("name", f"class_{idx}")
                    canon_name = _canonicalise_class_name(raw_name, aliases_tbl)
                    value_map[idx] = canon_name
                keep_empty = bool(manifest.get("keep_empty", False))
            elif expect_manifest:
                raise FileNotFoundError(
                    f"Tiles root '{root}' is missing manifest.json. Re-run the tiler with --metadata."
                )

            if not value_map:
                raise ValueError(
                    f"Unable to determine class mapping for tiles in '{root}'. "
                    "Ensure the tiler wrote a manifest with class indices."
                )

            for name in value_map.values():
                seen_classes.add(name)

            for image_path, mask_path in _iter_tile_pairs(images_dir, masks_dir):
                mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    continue
                if mask.ndim == 3:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                mask = mask.astype(np.int32, copy=False)
                unique_values = set(int(v) for v in np.unique(mask) if int(v) != 0)
                missing = sorted(v for v in unique_values if v not in value_map)
                if missing:
                    raise ValueError(
                        f"Mask '{mask_path}' from '{root}' contains unmapped values: {missing}. "
                        f"Update the manifest or regenerate the tiles with a complete class mapping."
                    )

                records.append(
                    {
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "value_map": value_map,
                        "keep_empty": keep_empty,
                        "root": root,
                        "root_name": root.name or str(root),
                        "images_dir": images_dir,
                        "nonzero_pixels": int(np.count_nonzero(mask)),
                    }
                )

            root_infos[root.name or str(root)] = {
                "images_dir": images_dir,
                "masks_dir": masks_dir,
                "path": root,
            }

        if not records:
            raise RuntimeError("No image/mask pairs found across the provided tile roots")

        class_order = []
        for key in sorted(DEFAULT_CATEGORIES.keys()):
            name = DEFAULT_CATEGORIES[key]
            if name in seen_classes:
                class_order.append(name)
        for name in sorted(seen_classes):
            if name not in class_order:
                class_order.append(name)

        self.classes = class_order
        self.num_classes = len(self.classes)
        self.C = self.num_classes
        self.class_to_index = {name: idx for idx, name in enumerate(self.classes, start=1)}
        self.items = []
        self.meta_by_path: Dict[str, Dict[str, str]] = {}
        per_root_counts = Counter()

        for rec in records:
            mapping = {}
            for value, name in rec["value_map"].items():
                idx = self.class_to_index.get(name)
                if idx is not None:
                    mapping[int(value)] = int(idx)
            if not mapping:
                continue

            if rec.get("nonzero_pixels", 0) == 0 and not rec["keep_empty"]:
                dropped_empty[rec["root_name"]] += 1
                continue

            path_str = str(rec["image_path"])
            rel = rec["image_path"].relative_to(rec["images_dir"]).as_posix()
            self.items.append(
                {
                    "path": path_str,
                    "mask_path": str(rec["mask_path"]),
                    "value_map": mapping,
                    "dataset": rec["root_name"],
                    "relative_path": rel,
                }
            )
            self.meta_by_path[path_str] = {
                "relative_path": rel,
                "dataset": rec["root_name"],
            }
            per_root_counts[rec["root_name"]] += 1

        if not self.items:
            raise RuntimeError("No usable tiles remained after filtering out empty masks")

        self.root_infos = root_infos
        self.dropped_empty = dict(dropped_empty)
        self.per_root_counts = dict(per_root_counts)

    def split_by_images(self, val_ratio=0.25, seed=42):
        n = len(self.items)
        ids = list(range(n))
        random.Random(seed).shuffle(ids)
        val_n = max(1, int(round(val_ratio * n)))
        va_idx = sorted(ids[:val_n])
        tr_idx = sorted(ids[val_n:])
        return tr_idx, va_idx

def build_prepare_tf(split: str, norm_mode: str, include_normalize: bool = True, aug_config: Optional[dict] = None):
    split = (split or "train").lower()
    cfg = _merge_dicts(DEFAULT_PREPARE_AUG.get(split, {}), (aug_config or {}).get(split, {}))
    if split == "train":
        aug = [
            A.HorizontalFlip(p=cfg.get("hflip_p", 0.5)),
            A.VerticalFlip(p=cfg.get("vflip_p", 0.1)),
            A.RandomRotate90(p=cfg.get("rot90_p", 0.2)),
        ]

        persp = cfg.get("perspective", {})
        if persp.get("p", 0) > 0:
            aug.append(A.Perspective(
                scale=_ensure_tuple(persp.get("scale"), (0.05, 0.12), clamp_len=2),
                keep_size=persp.get("keep_size", True),
                pad_mode=persp.get("pad_mode", cv2.BORDER_REFLECT_101),
                p=persp.get("p", 0.15),
            ))

        color_jit = cfg.get("color_jitter", {})
        if color_jit.get("p", 0) > 0:
            aug.append(A.ColorJitter(
                brightness=color_jit.get("brightness", 0.3),
                contrast=color_jit.get("contrast", 0.3),
                saturation=color_jit.get("saturation", 0.2),
                hue=color_jit.get("hue", 0.08),
                p=color_jit.get("p", 0.35),
            ))

        rand_bc = cfg.get("random_brightness_contrast", {})
        if rand_bc.get("p", 0) > 0:
            aug.append(A.RandomBrightnessContrast(
                brightness_limit=rand_bc.get("brightness_limit", 0.3),
                contrast_limit=rand_bc.get("contrast_limit", 0.3),
                p=rand_bc.get("p", 0.3),
            ))

        rand_gamma = cfg.get("random_gamma", {})
        if rand_gamma.get("p", 0) > 0:
            aug.append(A.RandomGamma(
                gamma_limit=_ensure_tuple(rand_gamma.get("gamma_limit"), (90, 110), clamp_len=2),
                gain=rand_gamma.get("gain", 1.0),
                p=rand_gamma.get("p", 0.3),
            ))

        motion_blur = cfg.get("motion_blur", {})
        if motion_blur.get("p", 0) > 0:
            aug.append(A.MotionBlur(
                blur_limit=_ensure_tuple(motion_blur.get("blur_limit"), (3, 7), clamp_len=2),
                p=motion_blur.get("p", 0.12),
            ))

        gauss_blur = cfg.get("gaussian_blur", {})
        if gauss_blur.get("p", 0) > 0:
            aug.append(A.GaussianBlur(
                blur_limit=_ensure_tuple(gauss_blur.get("blur_limit"), (3, 5), clamp_len=2),
                p=gauss_blur.get("p", 0.1),
            ))
    else:  # val, test и fallback
        aug = []
        motion_blur = cfg.get("motion_blur", {})
        if motion_blur.get("p", 0) > 0:
            aug.append(A.MotionBlur(
                blur_limit=_ensure_tuple(motion_blur.get("blur_limit"), (3, 5), clamp_len=2),
                p=motion_blur.get("p", 0.05),
            ))
        gauss_blur = cfg.get("gaussian_blur", {})
        if gauss_blur.get("p", 0) > 0:
            aug.append(A.GaussianBlur(
                blur_limit=_ensure_tuple(gauss_blur.get("blur_limit"), (3, 3), clamp_len=2),
                p=gauss_blur.get("p", 0.05),
            ))
    if include_normalize:
        if norm_mode == "clip":
            aug.append(A.Normalize(mean=CLIP_MEAN, std=CLIP_STD, max_pixel_value=255.0))
        else:
            aug.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0))
    return A.Compose(aug)
# ----------------------------- Dataset wrapper -----------------------------

def _make_random_resized_crop(size: int):
    """Build RandomResizedCrop compatible with Albumentations v1/v2 APIs."""
    params = inspect.signature(A.RandomResizedCrop.__init__).parameters
    if "size" in params:
        return A.RandomResizedCrop(
            size=(size, size),
            scale=(0.4, 1.0),
            ratio=(0.75, 1.33),
            interpolation=cv2.INTER_LINEAR,
        )
    return A.RandomResizedCrop(
        height=size,
        width=size,
        scale=(0.4, 1.0),
        ratio=(0.75, 1.33),
        interpolation=cv2.INTER_LINEAR,
    )

def _build_tf(train: bool, size: int, norm_mode: str, include_normalize: bool = True, aug_config: Optional[dict] = None):
    if train:
        cfg = _merge_dicts(DEFAULT_DATASET_AUG.get("train", {}), (aug_config or {}).get("train", {}))
        resize_or_crop = A.OneOf([
            _make_random_resized_crop(size),
            A.Compose([
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(
                    min_height=size,
                    min_width=size,
                    border_mode=cv2.BORDER_CONSTANT,
                ),
            ]),
        ], p=1.0)

        color_aug = A.OneOf([
            A.ColorJitter(
                brightness=cfg.get("color_jitter", {}).get("brightness", 0.3),
                contrast=cfg.get("color_jitter", {}).get("contrast", 0.3),
                saturation=cfg.get("color_jitter", {}).get("saturation", 0.25),
                hue=cfg.get("color_jitter", {}).get("hue", 0.08),
                p=cfg.get("color_jitter", {}).get("p", 1.0),
            ),
            A.RandomBrightnessContrast(
                brightness_limit=cfg.get("random_brightness_contrast", {}).get("brightness_limit", 0.3),
                contrast_limit=cfg.get("random_brightness_contrast", {}).get("contrast_limit", 0.3),
                p=cfg.get("random_brightness_contrast", {}).get("p", 1.0),
            ),
            A.HueSaturationValue(
                hue_shift_limit=cfg.get("hsv", {}).get("hue_shift_limit", 10),
                sat_shift_limit=cfg.get("hsv", {}).get("sat_shift_limit", 15),
                val_shift_limit=cfg.get("hsv", {}).get("val_shift_limit", 10),
                p=cfg.get("hsv", {}).get("p", 1.0),
            ),
            A.RGBShift(
                r_shift_limit=cfg.get("rgb_shift", {}).get("r_shift_limit", 10),
                g_shift_limit=cfg.get("rgb_shift", {}).get("g_shift_limit", 10),
                b_shift_limit=cfg.get("rgb_shift", {}).get("b_shift_limit", 10),
                p=cfg.get("rgb_shift", {}).get("p", 1.0),
            ),
            A.RandomGamma(
                gamma_limit=_ensure_tuple(cfg.get("random_gamma", {}).get("gamma_limit"), (90, 110), clamp_len=2),
                gain=cfg.get("random_gamma", {}).get("gain", 1.0),
                p=cfg.get("random_gamma", {}).get("p", 1.0),
            ),
        ], p=cfg.get("color_aug_p", 0.4))

        aug = [
            resize_or_crop,
            A.HorizontalFlip(p=cfg.get("hflip_p", 0.5)),
            A.VerticalFlip(p=cfg.get("vflip_p", 0.1)),
        ]

        persp = cfg.get("perspective", {})
        if persp.get("p", 0) > 0:
            aug.append(A.Perspective(
                scale=_ensure_tuple(persp.get("scale"), (0.05, 0.12), clamp_len=2),
                keep_size=persp.get("keep_size", True),
                pad_mode=persp.get("pad_mode", cv2.BORDER_REFLECT_101),
                p=persp.get("p", 0.18),
            ))

        ssr = cfg.get("shift_scale_rotate", {})
        aug.append(A.ShiftScaleRotate(
            shift_limit=ssr.get("shift_limit", 0.05),
            scale_limit=ssr.get("scale_limit", 0.2),
            rotate_limit=ssr.get("rotate_limit", 15),
            p=ssr.get("p", 0.5),
            border_mode=ssr.get("border_mode", cv2.BORDER_REFLECT_101),
        ))

        motion_blur = cfg.get("motion_blur", {})
        if motion_blur.get("p", 0) > 0:
            aug.append(A.MotionBlur(
                blur_limit=_ensure_tuple(motion_blur.get("blur_limit"), (3, 7), clamp_len=2),
                p=motion_blur.get("p", 0.1),
            ))

        aug.append(color_aug)
    else:
        aug = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),  # без value
        ]
    tf = A.ReplayCompose(aug)
    normalize = None
    if include_normalize:
        if norm_mode == "clip":
            normalize = A.Normalize(mean=CLIP_MEAN, std=CLIP_STD, max_pixel_value=255.0)
        else:
            normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0)
    return tf, normalize

def _mask_to_color(mask: np.ndarray, class_names):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    for idx, name in enumerate(class_names, start=1):
        color[mask == idx] = hex_to_bgr(LS_PALETTE.get(name, "#FF00FF"))
    return color


def _sanitize_name(name: str) -> str:
    chars = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
class SegDataset(Dataset):
    def __init__(self, seg_index: SegIndex, idxs, train: bool, size: int, norm_mode: str = "clip",
                 aug_dump_dir: str = "", aug_dump_limit: int = 0, aug_config: Optional[dict] = None):
        self.si = seg_index
        self.idxs = list(idxs)
        self.tf, self.normalize_tf = _build_tf(
            train,
            size,
            norm_mode,
            include_normalize=True,
            aug_config=aug_config,
        )
        self.to_tensor = ToTensorV2(transpose_mask=False)
        self.dump_dir = aug_dump_dir or ""
        self.dump_limit = int(aug_dump_limit) if aug_dump_limit else 0
        self.dump_enabled = bool(self.dump_dir and self.dump_limit > 0)
        self._dump_counter = 0
        if self.dump_enabled:
            os.makedirs(self.dump_dir, exist_ok=True)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        rec = self.si.items[self.idxs[i]]
        img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = self._load_mask(rec)
        tfed = self.tf(image=img, mask=mask)
        aug_img = tfed["image"]
        aug_mask = tfed["mask"].astype(np.uint8)
        replay = tfed.get("replay")

        norm_img = aug_img
        if self.normalize_tf is not None:
            norm_img = self.normalize_tf(image=aug_img)["image"]

        norm_img = np.ascontiguousarray(norm_img)

        x = self.to_tensor(image=norm_img)["image"]
        m = torch.from_numpy(aug_mask.astype(np.int64, copy=False))

        if self.dump_enabled and self._dump_counter < self.dump_limit:
            self._dump_sample(
                rec["path"],
                rec.get("mask_path"),
                img,
                aug_img,
                aug_mask,
                replay,
            )

        return x, m, rec["path"]

    def _load_mask(self, rec):
        mask = rec.get("mask")
        if isinstance(mask, np.ndarray):
            return mask

        mask_path = rec.get("mask_path")
        if mask_path:
            mask_arr = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask_arr is None:
                raise FileNotFoundError(f"Failed to read mask from '{mask_path}'")
            if mask_arr.ndim == 3:
                mask_arr = cv2.cvtColor(mask_arr, cv2.COLOR_BGR2GRAY)
            mask_arr = mask_arr.astype(np.int32, copy=False)
            mapping = rec.get("value_map") or {}
            if mapping:
                mapped = np.zeros_like(mask_arr, dtype=np.uint8)
                for src_val, dst_idx in mapping.items():
                    mapped[mask_arr == int(src_val)] = int(dst_idx)
                mask_arr = mapped
            else:
                mask_arr = mask_arr.astype(np.uint8, copy=False)
            return np.ascontiguousarray(mask_arr)

        raise ValueError("Record does not contain mask data")

    def _dump_sample(self, source_path, source_mask_path, original_img, aug_img, aug_mask, replay):
        idx = self._dump_counter
        self._dump_counter += 1
        base = os.path.splitext(os.path.basename(source_path))[0]
        safe_base = _sanitize_name(base)
        uid = uuid.uuid4().hex[:8]
        prefix = f"{idx:04d}_{safe_base}_{uid}"

        image_path = os.path.join(self.dump_dir, prefix + "_image.png")
        mask_path = os.path.join(self.dump_dir, prefix + "_mask.png")
        overlay_path = os.path.join(self.dump_dir, prefix + "_overlay.png")
        meta_path = os.path.join(self.dump_dir, prefix + "_meta.json")

        aug_vis = aug_img
        if aug_vis.dtype != np.uint8:
            aug_vis = np.clip(aug_vis, 0, 255).astype(np.uint8)
        cv2.imwrite(image_path, cv2.cvtColor(aug_vis, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_path, aug_mask)

        color_map = _mask_to_color(aug_mask, self.si.classes)
        save_overlay(aug_vis, color_map, overlay_path, alpha=0.6, blur=0,
                     add_legend=True, class_names=self.si.classes, palette=LS_PALETTE)

        orig_h, orig_w = original_img.shape[:2]
        aug_h, aug_w = aug_mask.shape[:2]
        meta = {
            "source_path": source_path,
            "source_mask_path": source_mask_path,
            "image_path": image_path,
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "original_size": [int(orig_h), int(orig_w)],
            "augmented_size": [int(aug_h), int(aug_w)],
            "replay": _to_serializable(replay) if replay is not None else None,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
