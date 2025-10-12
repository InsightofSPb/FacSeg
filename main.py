import os, json, math, argparse, hashlib, random, glob
from collections import defaultdict
from datetime import datetime

import numpy as np
import cv2
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import open_clip
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve

from torchvision import models as tvm


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

DEFAULT_CATEGORIES = {
    1: "CRACK",
    2: "SPALLING",
    3: "DELAMINATION",
    4: "MISSING_ELEMENT",
    5: "WATER_STAIN",
    6: "EFFLORESCENCE",
    7: "CORROSION",
    8: "ORNAMENT_INTACT",
}

LS_PALETTE = {
    "CRACK": "#E53935",
    "SPALLING": "#1E88E5",
    "DELAMINATION": "#43A047",
    "MISSING_ELEMENT": "#FB8C00",
    "WATER_STAIN": "#8E24AA",
    "EFFLORESCENCE": "#FDD835",
    "CORROSION": "#00ACC1",
    "ORNAMENT_INTACT": "#9E9E9E",
}


def hex_to_bgr(hex_str: str):
    """Convert web hex color to OpenCV BGR tuple."""
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def seed_everything(seed=42):
    """Deterministic seeds for Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stamp_out_dir(base: str):
    """Create timestamped output directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def safe_path(images_dir: str, coco_file_name: str):
    """Resolve COCO file_name coming from Label Studio export to a real path under images_dir."""
    p1 = os.path.join(images_dir, coco_file_name)
    if os.path.isfile(p1):
        return p1
    tail = coco_file_name.split("-", 1)[-1] if "-" in coco_file_name else coco_file_name
    p2 = os.path.join(images_dir, tail)
    if os.path.isfile(p2):
        return p2
    tail_low = tail.lower()
    for fn in os.listdir(images_dir):
        if fn.lower().endswith(tail_low):
            return os.path.join(images_dir, fn)
    return None


def polygons_to_mask(polygons, h, w, fill_value=1):
    """Rasterize COCO polygon list into a binary mask of size HxW with given fill value."""
    mask = np.zeros((h, w), np.uint8)
    for poly in polygons:
        if not poly:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        ipts = pts.astype(np.int32)
        cv2.fillPoly(mask, [ipts], int(fill_value))
    return mask


def count_params(module: nn.Module):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_int(n: int) -> str:
    """Human-friendly integer with thousands separators."""
    return f"{n:,}"


class TileIndex:
    """
    Precompute tiles and multi-labels for CLIP-tile training from COCO polygons.
    Each tile becomes a multi-label example: class is positive if coverage >= cover_thr.
    """

    def __init__(self, images_dir, coco_json, tile_size=512, stride=256, cover_thr=0.005, keep_empty=False):
        self.images_dir = images_dir
        self.tile_size = tile_size
        self.stride = stride
        with open(coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        if "categories" in coco and coco["categories"]:
            self.cat_map = {c["id"]: c.get("name", f"cat_{c['id']}") for c in coco["categories"]}
        else:
            self.cat_map = DEFAULT_CATEGORIES.copy()
        self.class_ids = sorted(self.cat_map.keys())
        self.classes = [self.cat_map[k] for k in self.class_ids]
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

            for y in range(0, max(1, H - tile_size + 1), stride):
                for x in range(0, max(1, W - tile_size + 1), stride):
                    tile_area = float(tile_size * tile_size)
                    labels = np.zeros(self.C, np.float32)
                    for j, cid in enumerate(self.class_ids):
                        inter = class_masks[cid][y:y + tile_size, x:x + tile_size].sum()
                        if inter / tile_area >= cover_thr:
                            labels[j] = 1.0
                    if keep_empty or labels.sum() > 0:
                        self.records.append((im["id"], path, x, y, tile_size, labels))

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
    """Dataset of CLIP tiles with light augmentations."""

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


class CLIPHead(nn.Module):
    """
    Frozen CLIP visual encoder with configurable head: linear or MLP.
    """

    def __init__(self, model_name="ViT-B-32-quickgelu", pretrained="openai",
                 n_classes=8, device="cpu",
                 head_type="mlp", head_hidden=2048, head_dropout=0.1):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.visual = self.model.visual
        for p in self.visual.parameters():
            p.requires_grad = False
        self.proj_dim = self.visual.output_dim
        if head_type == "linear":
            self.head = nn.Linear(self.proj_dim, n_classes, bias=True)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.proj_dim, head_hidden, bias=True),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        with torch.no_grad():
            feats = self.visual(x)
            feats = F.normalize(feats.float(), dim=-1)
        logits = self.head(feats)
        return logits, feats


def init_head_from_text(head_module, class_names, model_name="ViT-B-32-quickgelu", pretrained="openai", device="cpu"):
    """Initialize CLIP head weights from text prompts by averaging multiple templates."""
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    templates = [
        "a building facade with {}",
        "{} on a wall",
        "structural damage: {} on masonry",
        "a photo of {} on a facade",
    ]
    W = []
    with torch.no_grad():
        for name in class_names:
            phrases = [t.format(name.replace("_", " ").lower()) for t in templates]
            txt = tokenizer(phrases).to(device)
            emb = model.encode_text(txt).float()
            emb = F.normalize(emb, dim=-1).mean(0, keepdim=True)
            W.append(emb)
    W = torch.cat(W, dim=0)
    if isinstance(head_module, nn.Linear):
        head_module.weight.data.copy_(W)
        head_module.bias.data.zero_()
    else:
        for m in head_module.modules():
            if isinstance(m, nn.Linear) and m.out_features == W.shape[0]:
                m.weight.data.copy_(W)
                m.bias.data.zero_()
                break


@torch.no_grad()
def evaluate_clip(model, loader, device, n_classes, progress_desc=None):
    """Evaluate CLIP-tile model with ROC-AUC, AP and best-F1 per class and macro averages."""
    model.eval()
    all_y, all_p = [], []
    it = tqdm(loader, desc=progress_desc, leave=False) if progress_desc else loader
    for x, y, _ in it:
        x = x.to(device)
        logits, _ = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_p.append(probs)
        all_y.append(y.numpy())
    if not all_y:
        return {}, dict(roc_auc=np.nan, ap=np.nan, best_f1=np.nan), np.zeros((0, n_classes)), np.zeros((0, n_classes))
    Y = np.concatenate(all_y, 0)
    P = np.concatenate(all_p, 0)

    metrics = {}
    roc_list, ap_list, f1_list = [], [], []
    for c in range(n_classes):
        y, p = Y[:, c], P[:, c]
        if y.size == 0 or y.max() == 0 or y.min() == 1:
            roc, ap, f1 = np.nan, np.nan, np.nan
        else:
            try:
                roc = roc_auc_score(y, p)
            except Exception:
                roc = np.nan
            ap = average_precision_score(y, p)
            pr, rc, _ = precision_recall_curve(y, p)
            f1 = (2 * pr * rc / (pr + rc + 1e-9)).max()
        metrics[c] = dict(roc_auc=float(roc), ap=float(ap), best_f1=float(f1))
        if not math.isnan(roc):
            roc_list.append(roc)
        if not math.isnan(ap):
            ap_list.append(ap)
        if not math.isnan(f1):
            f1_list.append(f1)
    macro = dict(
        roc_auc=float(np.nanmean(roc_list) if roc_list else np.nan),
        ap=float(np.nanmean(ap_list) if ap_list else np.nan),
        best_f1=float(np.nanmean(f1_list) if f1_list else np.nan),
    )
    return metrics, macro, Y, P


def compute_pos_weight(loader, n_classes):
    """Compute positive weights per class for BCE loss from a loader of multi-label tiles."""
    pos = np.zeros(n_classes, np.float64)
    neg = np.zeros(n_classes, np.float64)
    for _, y, _ in loader:
        y = y.numpy()
        pos += y.sum(axis=0)
        neg += (1 - y).sum(axis=0)
    pos = np.maximum(pos, 1.0)
    neg = np.maximum(neg, 1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)


def infer_batch_clip(model, tiles_batch, device):
    """Forward a batch of tiles through CLIP head and return probabilities."""
    x = torch.cat(tiles_batch, 0).to(device)
    logits, _ = model(x)
    probs = torch.sigmoid(logits).cpu().detach().numpy()
    return probs


def cosine_kernel(ts: int):
    """Create a 2D Hann window of size ts x ts for smooth tile blending."""
    win1 = np.hanning(ts)
    w = np.outer(win1, win1).astype(np.float32)
    w = w / w.max()
    return w


def save_composite_overlay(src_bgr, prob_C_H_W, class_names, out_path, palette=LS_PALETTE, vis_thr=0.5, alpha=0.7, blur=0):
    """
    Build a single composite overlay from C-class probability maps.
    For each class, transparency is proportional to probability above vis_thr.
    """
    src = src_bgr.astype(np.float32) / 255.0
    out = src.copy()
    C, H, W = prob_C_H_W.shape
    for c, name in enumerate(class_names):
        prob = np.clip(prob_C_H_W[c], 0.0, 1.0).astype(np.float32)
        if blur and blur >= 3 and blur % 2 == 1:
            prob = cv2.GaussianBlur(prob, (blur, blur), 0)
        a = np.zeros_like(prob, dtype=np.float32)
        m = prob >= vis_thr
        a[m] = (prob[m] - vis_thr) / (1.0 - vis_thr + 1e-6)
        a = np.clip(a * alpha, 0.0, 1.0)
        color_bgr = np.array(hex_to_bgr(palette.get(name, "#FF00FF")), dtype=np.float32) / 255.0
        color = np.ones((H, W, 3), np.float32) * color_bgr[None, None, :]
        out = out * (1 - a[..., None]) + color * a[..., None]
    out = (out * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, out)


def clip_infer_on_dir(model, args, class_names):
    """
    CLIP inference over test_dir with sliding-window accumulation and a single composite output per image.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    out_base = stamp_out_dir(os.path.join(args.out_dir, "clip_infer"))
    out_comp = os.path.join(out_base, "overlays_composite")
    os.makedirs(out_comp, exist_ok=True)

    tf = A.Compose([
        A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2()
    ])

    ts = args.tile_size
    stride = args.stride
    kernel = cosine_kernel(ts)

    img_list = sorted([p for p in glob.glob(os.path.join(args.test_dir, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))])
    for path in tqdm(img_list, desc="CLIP infer"):
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        if src is None:
            continue
        H, W = src.shape[:2]
        acc = np.zeros((len(class_names), H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)

        batch_tiles, coords = [], []
        for y in range(0, max(1, H - ts + 1), stride):
            for x in range(0, max(1, W - ts + 1), stride):
                img = cv2.cvtColor(src[y:y + ts, x:x + ts], cv2.COLOR_BGR2RGB)
                tile = tf(image=img)["image"].unsqueeze(0)
                batch_tiles.append(tile)
                coords.append((x, y))
                if len(batch_tiles) == args.batch_size:
                    probs = infer_batch_clip(model, batch_tiles, device)
                    for p, (xx, yy) in zip(probs, coords):
                        for c in range(len(class_names)):
                            acc[c, yy:yy + ts, xx:xx + ts] += p[c] * kernel
                        cnt[yy:yy + ts, xx:xx + ts] += kernel
                    batch_tiles, coords = [], []
        if batch_tiles:
            probs = infer_batch_clip(model, batch_tiles, device)
            for p, (xx, yy) in zip(probs, coords):
                for c in range(len(class_names)):
                    acc[c, yy:yy + ts, xx:xx + ts] += p[c] * kernel
                cnt[yy:yy + ts, xx:xx + ts] += kernel

        cnt[cnt == 0] = 1.0
        acc /= cnt[None, :, :]

        save_composite_overlay(src, acc, class_names,
                               os.path.join(out_comp, os.path.basename(path)),
                               palette=LS_PALETTE, vis_thr=args.vis_thr, alpha=args.vis_max_alpha, blur=args.vis_blur)


def duplicate_positive_indices(index: TileIndex, indices, dup_factor: int):
    """Upsample tiles that contain at least one positive class by duplication."""
    if dup_factor <= 1:
        return indices
    out = []
    for i in indices:
        labels = index.records[i][5]
        if labels.sum() > 0:
            out.extend([i] * dup_factor)
        else:
            out.append(i)
    return out


def colorize_mask(mask_hw: np.ndarray, id_to_name: dict):
    """Convert HxW class-id mask into BGR color image using Label Studio palette."""
    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), np.uint8)
    for cid, name in id_to_name.items():
        if cid == 0:
            continue
        color = hex_to_bgr(LS_PALETTE.get(name, "#FF00FF"))
        out[mask_hw == cid] = color
    return out


class SegIndex:
    """
    COCO-to-segmentation index that groups annotations per image and builds dense label masks.
    Class ids are remapped to contiguous range 1..C, background=0.
    """

    def __init__(self, images_dir, coco_json):
        self.images_dir = images_dir
        with open(coco_json, "r", encoding="utf-8") as f:
            coco = json.load(f)

        if "categories" in coco and coco["categories"]:
            id_to_name = {c["id"]: c.get("name", f"cat_{c['id']}") for c in coco["categories"]}
        else:
            id_to_name = DEFAULT_CATEGORIES.copy()

        class_ids = sorted(id_to_name.keys())
        self.id_to_name = {i: id_to_name[i] for i in class_ids}
        self.name_to_id = {v: k for k, v in self.id_to_name.items()}

        self.cid_to_idx = {cid: i + 1 for i, cid in enumerate(class_ids)}
        self.idx_to_name = {i + 1: id_to_name[cid] for i, cid in enumerate(class_ids)}
        self.num_classes = len(class_ids)

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

            mask = np.zeros((H, W), np.uint8)
            for a in anns_by_img[img_id]:
                cid = a["category_id"]
                idx = self.cid_to_idx.get(cid, 0)
                seg = a.get("segmentation", [])
                if isinstance(seg, list) and seg and isinstance(seg[0], list):
                    poly_mask = polygons_to_mask(seg, H, W, 1)
                    mask[poly_mask > 0] = idx

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
    Segmentation dataset using Albumentations.
    Train: random scale, pad, random crop, flips, mild color jitter and rotation.
    Val/Test: resize long side, pad to square without cropping.
    """

    def __init__(self, seg_index: SegIndex, indices, train=True, size=512):
        self.si = seg_index
        self.idxs = indices
        self.size = size
        if train:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=int(size * 1.25)),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.RandomCrop(size, size),
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-10, 10), fit_output=False, border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02, p=0.3),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        rec = self.si.items[self.idxs[i]]
        img = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = rec["mask"]
        aug = self.tf(image=img, mask=mask)
        img = aug["image"]
        mask = aug["mask"].astype(np.int64)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return img, torch.from_numpy(mask), rec["path"]


def get_seg_model(name: str, n_classes: int, weights_tag: str):
    """
    Build a torchvision segmentation model and replace classifier for n_classes.
    Supported names: deeplabv3_resnet50/101, fcn_resnet50/101.
    """
    name = name.lower()

    if name == "deeplabv3_resnet50":
        weights = tvm.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if weights_tag == "default" else None
        model = tvm.segmentation.deeplabv3_resnet50(weights=weights, aux_loss=False)
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, n_classes + 1, 1)
        return model

    if name == "deeplabv3_resnet101":
        weights = tvm.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT if weights_tag == "default" else None
        model = tvm.segmentation.deeplabv3_resnet101(weights=weights, aux_loss=False)
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, n_classes + 1, 1)
        return model

    if name == "fcn_resnet50":
        weights = tvm.segmentation.FCN_ResNet50_Weights.DEFAULT if weights_tag == "default" else None
        model = tvm.segmentation.fcn_resnet50(weights=weights, aux_loss=False)
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, n_classes + 1, 1)
        return model

    if name == "fcn_resnet101":
        weights = tvm.segmentation.FCN_ResNet101_Weights.DEFAULT if weights_tag == "default" else None
        model = tvm.segmentation.fcn_resnet101(weights=weights, aux_loss=False)
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, n_classes + 1, 1)
        return model

    raise ValueError("Unsupported seg_model.")


def seg_metrics_from_confmat(conf: np.ndarray):
    """
    Compute pixel accuracy and IoU per class from confusion matrix.
    Returns dict with per-class IoU, mean IoU (with and without background) and pixel accuracy.
    """
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn + 1e-9
    iou = tp / denom
    miou_incl_bg = float(np.nanmean(iou))
    if len(iou) > 1:
        miou_excl_bg = float(np.nanmean(iou[1:]))
    else:
        miou_excl_bg = float("nan")
    pix_acc = float(tp.sum() / (conf.sum() + 1e-9))
    return dict(per_class_iou=iou.tolist(), miou_incl_bg=miou_incl_bg, miou_excl_bg=miou_excl_bg, pixel_acc=pix_acc)


def seg_evaluate(model, loader, num_classes_plus_bg: int, device):
    """Evaluate segmentation model and compute confusion matrix-based metrics."""
    model.eval()
    conf = np.zeros((num_classes_plus_bg, num_classes_plus_bg), np.int64)
    with torch.no_grad():
        for x, y, _ in tqdm(loader, desc="Seg Validate", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)["out"]
            pred = logits.argmax(1)
            for yi, pi in zip(y, pred):
                y1 = yi.view(-1).cpu().numpy()
                p1 = pi.view(-1).cpu().numpy()
                m = (y1 >= 0) & (y1 < num_classes_plus_bg)
                y1 = y1[m]
                p1 = p1[m]
                cm = np.bincount(y1 * num_classes_plus_bg + p1, minlength=num_classes_plus_bg ** 2)
                conf += cm.reshape(num_classes_plus_bg, num_classes_plus_bg)
    return seg_metrics_from_confmat(conf)


def sliding_window_seg_infer(model, bgr, size=768, stride=512, device="cpu"):
    """
    High-resolution segmentation inference with overlapping tiles and cosine blending.
    Returns HxW logits tensor (C+1 channels). Model expects ImageNet-normalized RGB tensors.
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    pad_h = (size - H % size) % size
    pad_w = (size - W % size) % size
    rgb_pad = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    Hp, Wp = rgb_pad.shape[:2]

    kernel = cosine_kernel(size)
    acc = None
    wsum = np.zeros((Hp, Wp), np.float32)

    def prep_tile(img):
        img = img.astype(np.float32) / 255.0
        img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        ten = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        return ten

    model.eval()
    with torch.no_grad():
        for y in range(0, Hp - size + 1, stride):
            for x in range(0, Wp - size + 1, stride):
                tile = rgb_pad[y:y + size, x:x + size]
                ten = prep_tile(tile)
                out = model(ten)["out"]
                out_np = out.squeeze(0).cpu().numpy()
                if acc is None:
                    acc = np.zeros((out_np.shape[0], Hp, Wp), np.float32)
                for c in range(out_np.shape[0]):
                    acc[c, y:y + size, x:x + size] += out_np[c] * kernel
                wsum[y:y + size, x:x + size] += kernel

    wsum[wsum == 0] = 1.0
    acc /= wsum[None, :, :]

    acc = acc[:, :H, :W]
    return acc


def seg_infer_save(model, args, seg_index: SegIndex):
    """
    Run segmentation inference on images from test_dir and save color masks and overlays.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_base = stamp_out_dir(os.path.join(args.out_dir, "seg_infer"))
    out_masks = os.path.join(out_base, "masks")
    out_over = os.path.join(out_base, "overlays")
    os.makedirs(out_masks, exist_ok=True)
    os.makedirs(out_over, exist_ok=True)

    img_list = sorted([p for p in glob.glob(os.path.join(args.test_dir, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))])
    id_to_name = {i: seg_index.idx_to_name[i] for i in seg_index.idx_to_name}

    for path in tqdm(img_list, desc="Seg infer"):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        logits = sliding_window_seg_infer(model, bgr, size=args.seg_tile_size, stride=args.seg_stride, device=device)
        pred = logits.argmax(0).astype(np.uint8)

        color = colorize_mask(pred, id_to_name)
        overlay = cv2.addWeighted(bgr, 0.4, color, 0.6, 0)

        cv2.imwrite(os.path.join(out_masks, os.path.basename(path)), color)
        cv2.imwrite(os.path.join(out_over,  os.path.basename(path)), overlay)


def train(args):
    """
    Entry point that dispatches between CLIP-based tile training/inference and full segmentation training/inference.
    """
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "infer":
        class_names = list(DEFAULT_CATEGORIES.values())
        if args.coco_json and os.path.isfile(args.coco_json):
            try:
                with open(args.coco_json, "r", encoding="utf-8") as f:
                    coco = json.load(f)
                if "categories" in coco and coco["categories"]:
                    id_to_name = {c["id"]: c.get("name", f"cat_{c['id']}") for c in coco["categories"]}
                    class_names = [id_to_name[k] for k in sorted(id_to_name.keys())]
            except Exception:
                pass

        model = CLIPHead(model_name=args.clip_model, pretrained=args.clip_ckpt,
                         n_classes=len(class_names), device=device,
                         head_type=args.head_type, head_hidden=args.head_hidden, head_dropout=args.head_dropout).to(device)
        init_head_from_text(model.head, class_names, model_name=args.clip_model, pretrained=args.clip_ckpt, device=device)

        ckpt_path = args.ckpt
        if not ckpt_path:
            subdirs = sorted([d for d in glob.glob(os.path.join(args.out_dir, "*")) if os.path.isdir(d)])
            for d in reversed(subdirs):
                cand = os.path.join(d, "model.pt")
                if os.path.isfile(cand):
                    ckpt_path = cand
                    break
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError("Checkpoint not found. Provide --ckpt or put model.pt under out_dir/<timestamp>/")

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)

        print(f"[i] CLIP infer with ckpt: {ckpt_path}")
        clip_infer_on_dir(model, args, class_names)
        print("[OK] CLIP inference finished.")
        return

    if args.mode == "seg_infer":
        if not args.ckpt or not os.path.isfile(args.ckpt):
            raise FileNotFoundError("Provide segmentation checkpoint with --ckpt")
        seg_index = SegIndex(args.images_dir, args.coco_json)
        model = get_seg_model(args.seg_model, seg_index.num_classes, weights_tag="none").to(device)
        sd = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(sd["state_dict"], strict=True)
        seg_infer_save(model, args, seg_index)
        print("[OK] Segmentation inference finished.")
        return

    if args.mode == "seg_train":
        out_dir = stamp_out_dir(args.out_dir)
        print(f"[i] results will be saved to: {out_dir}")

        seg_index = SegIndex(args.images_dir, args.coco_json)
        tr_idx, va_idx = seg_index.split_by_images(val_ratio=args.val_ratio, seed=42)
        ds_tr = SegDataset(seg_index, tr_idx, train=True,  size=args.seg_img_size)
        ds_va = SegDataset(seg_index, va_idx, train=False, size=args.seg_img_size)

        dl_tr = DataLoader(ds_tr, batch_size=args.seg_batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=args.seg_batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = get_seg_model(args.seg_model, seg_index.num_classes, weights_tag=args.seg_weights).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.seg_lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        best_miou = -1.0
        ckpt_path = os.path.join(out_dir, "seg_model.pt")

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for x, y, _ in tqdm(dl_tr, desc=f"Seg Epoch {epoch}/{args.epochs}"):
                x = x.to(device)
                y = y.to(device)
                out = model(x)["out"]
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            metrics = seg_evaluate(model, dl_va, seg_index.num_classes + 1, device)
            print(f"[{epoch}] seg loss={np.mean(losses):.4f} | miou_excl_bg={metrics['miou_excl_bg']:.3f} | pixel_acc={metrics['pixel_acc']:.3f}")

            if metrics["miou_excl_bg"] > best_miou:
                best_miou = metrics["miou_excl_bg"]
                torch.save({"state_dict": model.state_dict(), "args": vars(args), "metrics": metrics}, ckpt_path)

        print(f"[OK] Segmentation training finished. Checkpoint: {ckpt_path}")
        return

    if args.mode == "train":
        args.out_dir = stamp_out_dir(args.out_dir)
        print(f"[i] results will be saved to: {args.out_dir}")

        index = TileIndex(args.images_dir, args.coco_json,
                          tile_size=args.tile_size, stride=args.stride,
                          cover_thr=args.cover_thr, keep_empty=False)

        tr_idx, va_idx = index.split_by_images(val_ratio=args.val_ratio, seed=42)
        tr_idx = duplicate_positive_indices(index, tr_idx, dup_factor=args.pos_dup)
        print(f"[i] train tiles: {len(tr_idx)} (dup={args.pos_dup}) | val tiles: {len(va_idx)} | classes: {index.classes}")

        ds_tr = TilesDataset(index, tr_idx, augment=True,  img_size=args.img_size)
        ds_va = TilesDataset(index, va_idx, augment=False, img_size=args.img_size)

        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = CLIPHead(model_name=args.clip_model, pretrained=args.clip_ckpt,
                         n_classes=len(index.classes), device=device,
                         head_type=args.head_type, head_hidden=args.head_hidden, head_dropout=args.head_dropout).to(device)
        init_head_from_text(model.head, index.classes, model_name=args.clip_model, pretrained=args.clip_ckpt, device=device)

        vis_total, vis_train = count_params(model.visual)
        head_total, head_train = count_params(model.head)
        full_total, full_train = vis_total + head_total, vis_train + head_train
        print(f"[i] Visual encoder params: total={format_int(vis_total)}, trainable={format_int(vis_train)}")
        print(f"[i] Head params          : total={format_int(head_total)}, trainable={format_int(head_train)}")
        print(f"[i] Visual+Head total    : total={format_int(full_total)}, trainable={format_int(full_train)}")

        pos_weight = compute_pos_weight(dl_tr, len(index.classes)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)

        best_macro = -1.0
        ckpt_path = os.path.join(args.out_dir, "model.pt")

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            for x, y, _ in tqdm(dl_tr, desc=f"Epoch {epoch}/{args.epochs}"):
                x = x.to(device)
                y = y.to(device)
                logits, _ = model(x)
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            _, M_tr, _, _ = evaluate_clip(model, dl_tr, device, len(index.classes), progress_desc=None)
            _, M_va, _, _ = evaluate_clip(model, dl_va, device, len(index.classes), progress_desc="Validate")
            print(f"[{epoch}] loss={np.mean(losses):.4f} | TRAIN mAP={M_tr['ap']:.3f} F1={M_tr['best_f1']:.3f} | VAL mAP={M_va['ap']:.3f} F1={M_va['best_f1']:.3f}")

            if M_va['ap'] > best_macro:
                best_macro = M_va['ap']
                torch.save({"state_dict": model.state_dict(), "classes": index.classes, "args": vars(args)}, ckpt_path)

        print(f"[OK] CLIP training finished. Checkpoint: {ckpt_path}")
        return

    raise ValueError("Unknown mode")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["train", "infer", "seg_train", "seg_infer"], default="infer")

    ap.add_argument("--images_dir", default="/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/train")
    ap.add_argument("--coco_json",  default="/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/annotations/result_coco.json")
    ap.add_argument("--out_dir",    default="/home/sasha/Facade_segmentation/results")
    ap.add_argument("--test_dir",   default="/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/test")

    ap.add_argument("--ckpt", default='/home/sasha/Facade_segmentation/results/20251012_205308/model.pt')

    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--stride",    type=int, default=256)
    ap.add_argument("--cover_thr", type=float, default=0.005)

    ap.add_argument("--img_size",   type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs",     type=int, default=2)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--val_ratio",  type=float, default=0.25)

    ap.add_argument("--clip_model", default="ViT-B-32-quickgelu")
    ap.add_argument("--clip_ckpt",  default="openai")
    ap.add_argument("--head_type", choices=["linear", "mlp"], default="mlp")
    ap.add_argument("--head_hidden", type=int, default=512)
    ap.add_argument("--head_dropout", type=float, default=0.1)

    ap.add_argument("--vis_thr",       type=float, default=0.5)
    ap.add_argument("--vis_max_alpha", type=float, default=0.5)
    ap.add_argument("--vis_blur",      type=int,   default=10)

    ap.add_argument("--pos_dup", type=int, default=3)

    ap.add_argument("--seg_model",
                    choices=["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"],
                    default="deeplabv3_resnet101")
    ap.add_argument("--seg_weights", choices=["default", "none"], default="default")
    ap.add_argument("--seg_img_size",  type=int, default=512)
    ap.add_argument("--seg_batch_size",type=int, default=6)
    ap.add_argument("--seg_lr",        type=float, default=2e-4)
    ap.add_argument("--seg_tile_size", type=int, default=768)
    ap.add_argument("--seg_stride",    type=int, default=512)

    args = ap.parse_args()
    train(args)
