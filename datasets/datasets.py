import os, json, random
from collections import defaultdict

import numpy as np
import cv2
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.utils import (
    CLIP_MEAN, CLIP_STD, IMAGENET_MEAN, IMAGENET_STD,
    DEFAULT_CATEGORIES, load_aliases_json, _norm_name, remap_id_to_canonical,
    polygons_to_mask
)


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

    def split_by_images(self, val_ratio=0.25, seed=42):
        n = len(self.items)
        ids = list(range(n))
        random.Random(seed).shuffle(ids)
        val_n = max(1, int(round(val_ratio * n)))
        va_idx = sorted(ids[:val_n])
        tr_idx = sorted(ids[val_n:])
        return tr_idx, va_idx

def build_prepare_tf(split: str, norm_mode: str, include_normalize: bool = True):
    split = (split or "train").lower()
    if split == "train":
        aug = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ]
    elif split == "val":
        aug = []
    else:  # test and fallback
        aug = [
            A.GaussianBlur(blur_limit=(3, 3), p=0.05),
        ]
    if include_normalize:
        if norm_mode == "clip":
            aug.append(A.Normalize(mean=CLIP_MEAN, std=CLIP_STD, max_pixel_value=255.0))
        else:
            aug.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0))
    return A.Compose(aug)
# ----------------------------- Dataset wrapper -----------------------------

def _build_tf(train: bool, size: int, norm_mode: str, include_normalize: bool = True):
    if train:
        aug = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),  # без value
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=15, p=0.5,
                               border_mode=cv2.BORDER_REFLECT_101),
        ]
    else:
        aug = [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),  # без value
        ]
    if include_normalize:
        if norm_mode == "clip":
            aug.append(A.Normalize(mean=CLIP_MEAN, std=CLIP_STD, max_pixel_value=255.0))
        else:
            aug.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0))
    return A.Compose(aug)

class SegDataset(Dataset):
    def __init__(self, seg_index: SegIndex, idxs, train: bool, size: int, norm_mode: str = "clip"):
        self.si = seg_index
        self.idxs = list(idxs)
        self.tf = _build_tf(train, size, norm_mode, include_normalize=True)
        self.to_tensor = ToTensorV2(transpose_mask=False)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        rec = self.si.items[self.idxs[i]]
        img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = rec["mask"]
        tfed = self.tf(image=img, mask=mask)
        x = tfed["image"]
        m = tfed["mask"].astype(np.int64)  # 0..C
        x = self.to_tensor(image=x)["image"]
        return x, torch.from_numpy(m), rec["path"]

