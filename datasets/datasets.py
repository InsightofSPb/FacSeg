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
    polygons_to_mask, safe_path
)

# ---------- Tile (multi-label classification) ----------
class TileIndex:
    def __init__(self, images_dir, coco_json, tile_size=768, stride=512,
                 cover_thr=0.01, keep_empty=True, class_aliases=""):
        self.images_dir = images_dir
        self.tile_size = int(tile_size)
        self.stride = int(stride)

        with open(coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        # categories -> canonical names
        if "categories" in coco and coco["categories"]:
            raw_map = {c["id"]: c.get("name", f"cat_{c['id']}") for c in coco["categories"]}
        else:
            raw_map = DEFAULT_CATEGORIES.copy()

        aliases = load_aliases_json(class_aliases)
        norm_map = {cid: _norm_name(n) for cid, n in raw_map.items()}
        canon_map, present_names = remap_id_to_canonical(norm_map, aliases, DEFAULT_CATEGORIES)

        # Stable order by DEFAULT_CATEGORIES but only those present
        canon_order = [DEFAULT_CATEGORIES[k] for k in sorted(DEFAULT_CATEGORIES.keys()) if DEFAULT_CATEGORIES[k] in present_names]
        self.class_ids = [cid for cid, nm in canon_map.items() if nm in set(canon_order)]
        self.classes = canon_order
        self.C = len(self.classes)

        anns_by_img = defaultdict(list)
        for a in coco.get("annotations", []):
            if a.get("iscrowd", 0) == 1:
                continue
            anns_by_img[a["image_id"]].append(a)

        self.images = [im for im in coco.get("images", []) if im["id"] in anns_by_img]
        assert len(self.images) > 0, "No annotated images found in COCO."

        self.records = []
        for im in tqdm(self.images, desc="Index tiles"):
            path = safe_path(images_dir, im["file_name"])
            if path is None:
                print(f"[!] Missing image file: {im['file_name']}")
                continue
            H, W = im["height"], im["width"]

            class_masks = {cid: np.zeros((H, W), np.uint8) for cid in self.class_ids}
            for a in anns_by_img[im["id"]]:
                cid = a["category_id"]
                seg = a.get("segmentation", [])
                if isinstance(seg, list) and seg and isinstance(seg[0], list):
                    class_masks[cid] |= polygons_to_mask(seg, H, W, 1)

                for y in range(0, max(1, H - self.tile_size + 1), self.stride):
                    for x in range(0, max(1, W - self.tile_size + 1), self.stride):
                        tile_h = min(self.tile_size, H - y)
                        tile_w = min(self.tile_size, W - x)
                        tile_area = float(tile_h * tile_w)
                        labels = np.zeros(self.C, np.float32)
                        y2, x2 = y + tile_h, x + tile_w
                        for j, cid in enumerate(self.class_ids):
                            inter = class_masks[cid][y:y2, x:x2].sum()
                            if inter / tile_area >= cover_thr:
                                labels[j] = 1.0
                        if keep_empty or labels.sum() > 0:
                            self.records.append((im["id"], path, x, y, min(self.tile_size, tile_w), labels))

        self.image_ids = sorted({r[0] for r in self.records})

    def split_by_images(self, val_ratio=0.25, seed=42):
        rng = random.Random(seed)
        ids = self.image_ids[:]
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_ratio)))
        val_ids = set(ids[:n_val])
        train_idx, val_idx = [], []
        for i, rec in enumerate(self.records):
            img_id = rec[0]
            (val_idx if img_id in val_ids else train_idx).append(i)
        return train_idx, val_idx


class TilesDataset(Dataset):
    def __init__(self, index: TileIndex, indices, augment=True, img_size=224):
        self.index = index
        self.idxs = indices
        self.size = img_size
        if augment:
            self.tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-10, 10), fit_output=False, border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                A.Resize(self.size, self.size, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
                ToTensorV2()
            ])
        else:
            self.tf = A.Compose([
                A.Resize(self.size, self.size, interpolation=cv2.INTER_AREA),
                A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        rec = self.index.records[self.idxs[i]]
        _, path, x, y, ts, labels = rec
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        tile = img[y:y + ts, x:x + ts]
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile = self.tf(image=tile)["image"]
        return tile, torch.from_numpy(labels), (path, x, y, ts)


# ---------- Dense Segmentation (COCO -> dense masks) ----------
class SegIndex:
    """
    Собирает плотные маски из COCO полигонов.
    Ремап категорий в 1..C, фон=0. Алиасы применяются корректно.
    """
    def __init__(self, images_dir, coco_json, class_aliases=""):
        self.images_dir = images_dir
        with open(coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        if "categories" in coco and coco["categories"]:
            raw_id_to_name = {c["id"]: c.get("name", f"cat_{c['id']}") for c in coco["categories"]}
        else:
            raw_id_to_name = DEFAULT_CATEGORIES.copy()

        aliases = load_aliases_json(class_aliases)
        norm_id_to_name = {cid: _norm_name(n) for cid, n in raw_id_to_name.items()}
        canon_map, present_names = remap_id_to_canonical(norm_id_to_name, aliases, DEFAULT_CATEGORIES)

        # Стабильный порядок классов (только те, что реально присутствуют)
        canon_order = [DEFAULT_CATEGORIES[k] for k in sorted(DEFAULT_CATEGORIES.keys()) if DEFAULT_CATEGORIES[k] in present_names]
        name_to_newidx = {nm: i+1 for i, nm in enumerate(canon_order)}  # 1..C

        self.idx_to_name = {i+1: nm for i, nm in enumerate(canon_order)}
        self.num_classes = len(self.idx_to_name)

        anns_by_img = defaultdict(list)
        for a in coco.get("annotations", []):
            if a.get("iscrowd", 0) == 1:
                continue
            anns_by_img[a["image_id"]].append(a)

        self.items = []
        for im in coco.get("images", []):
            img_id = im["id"]
            if img_id not in anns_by_img:
                continue
            path = safe_path(images_dir, im["file_name"])
            if path is None:
                print(f"[!] Missing image file: {im['file_name']}")
                continue
            H, W = im["height"], im["width"]

            # 1) аккумулируем бинарки по классам
            class_bin = {i+1: np.zeros((H, W), np.uint8) for i in range(len(canon_order))}
            for a in anns_by_img[img_id]:
                cid = a["category_id"]
                raw_name = norm_id_to_name.get(cid, None)
                if raw_name is None:
                    continue
                canon_name = canon_map.get(cid, raw_name)
                idx = name_to_newidx.get(canon_name, 0)  # 1..C
                seg = a.get("segmentation", [])
                if idx > 0 and isinstance(seg, list) and seg and isinstance(seg[0], list):
                    poly_mask = polygons_to_mask(seg, H, W, 1)
                    class_bin[idx] |= poly_mask

            # 2) склеиваем по приоритету (игнорируя отсутствующие в датасете)
            PRIORITY = ["CRACK","MISSING_ELEMENT","SPALLING","DELAMINATION",
                        "CORROSION","REPAIRS","TEXT_OR_IMAGES",
                        "WATER_STAIN","EFFLORESCENCE","ORNAMENT_INTACT"]
            mask = np.zeros((H, W), np.uint8)
            unassigned = mask == 0
            for name in PRIORITY:
                idx = name_to_newidx.get(name, 0)
                if idx == 0:
                    continue
                m = class_bin[idx] > 0
                take = m & unassigned
                mask[take] = idx
                unassigned &= ~take

            self.items.append(dict(id=img_id, path=path, height=H, width=W, mask=mask))

        assert len(self.items) > 0, "No images with segmentation masks."

    def split_by_images(self, val_ratio=0.25, seed=42):
        rng = random.Random(seed)
        ids = list(range(len(self.items)))
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_ratio)))
        return ids[n_val:], ids[:n_val]


class SegDataset(Dataset):
    """
    Train: random scale -> pad -> random crop -> aug.
    Val/Test: resize longest -> pad to square.
    norm_mode: 'imagenet' (torchvision) или 'clip' (OVSeg).
    """
    def __init__(self, seg_index: SegIndex, indices, train=True, size=512, norm_mode="imagenet"):
        self.si = seg_index
        self.idxs = indices
        self.size = size
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if norm_mode == "imagenet" else (CLIP_MEAN, CLIP_STD)
        if train:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=int(size * 1.25)),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT),
                A.RandomCrop(size, size),
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-10, 10), fit_output=False, border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02, p=0.3),
                A.Normalize(mean=mean, std=std)
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=mean, std=std)
            ])
        self.to_tensor = ToTensorV2()

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
