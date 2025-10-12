"""
Tiles-based multilabel training and inference on facade COCO masks using a frozen CLIP image encoder + lightweight head.
Includes: tiling, mild augmentations, class balancing by duplicating positive tiles, validation metrics,
per-class and composite overlays, multi-threshold visualization, timestamped output folders, and pure inference mode.
"""

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


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

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
    """Convert HEX color to OpenCV BGR tuple."""
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (b, g, r)


def seed_everything(seed=42):
    """Make runs deterministic-ish."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def safe_path(images_dir: str, coco_file_name: str):
    """Resolve actual image path when COCO stores prefixed names like 'hash-IMG_123.jpg'."""
    p1 = os.path.join(images_dir, coco_file_name)
    if os.path.isfile(p1): return p1
    tail = coco_file_name.split("-", 1)[-1] if "-" in coco_file_name else coco_file_name
    p2 = os.path.join(images_dir, tail)
    if os.path.isfile(p2): return p2
    tail_low = tail.lower()
    for fn in os.listdir(images_dir):
        if fn.lower().endswith(tail_low):
            return os.path.join(images_dir, fn)
    return None


def polygons_to_mask(polygons, h, w):
    """Rasterize COCO polygon list into a binary mask."""
    mask = np.zeros((h, w), np.uint8)
    for poly in polygons:
        if not poly: 
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, w-1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h-1)
        ipts = pts.astype(np.int32)
        cv2.fillPoly(mask, [ipts], 1)
    return mask


def count_params(module: nn.Module):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def stamp_out_dir(base: str):
    """Create a timestamped subfolder and return its path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def parse_thrs(s: str):
    """Parse comma/semicolon separated thresholds into a float list."""
    return [float(x) for x in str(s).replace(";", ",").split(",") if x.strip()]


class TileIndex:
    """Pre-compute tiles and multilabels per image by intersecting with COCO masks."""
    def __init__(self, images_dir, coco_json, tile_size=512, stride=256, cover_thr=0.005, keep_empty=False):
        self.images_dir = images_dir
        self.tile_size = tile_size; self.stride = stride

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
        assert len(self.images) > 0, "No labeled images in COCO."

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
                    class_masks[cid] |= polygons_to_mask(seg, H, W)

            for y in range(0, max(1, H - tile_size + 1), stride):
                for x in range(0, max(1, W - tile_size + 1), stride):
                    tile_area = float(tile_size * tile_size)
                    labels = np.zeros(self.C, np.float32)
                    for j, cid in enumerate(self.class_ids):
                        inter = class_masks[cid][y:y+tile_size, x:x+tile_size].sum()
                        if inter / tile_area >= cover_thr:
                            labels[j] = 1.0
                    if keep_empty or labels.sum() > 0:
                        self.records.append((im["id"], path, x, y, tile_size, labels))

        self.image_ids = sorted({r[0] for r in self.records})

    def split_by_images(self, val_ratio=0.25, seed=42):
        """Split tiles by image ids to avoid leakage."""
        rng = random.Random(seed)
        ids = self.image_ids[:]
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_ratio)))
        val_ids = set(ids[:n_val])
        train_idx, val_idx = [], []
        for i, rec in enumerate(self.records):
            (img_id, _, _, _, _, _) = rec
            (val_idx if img_id in val_ids else train_idx).append(i)
        return train_idx, val_idx


class TilesDataset(Dataset):
    """Return (augmented) CLIP-normalized tiles and multilabels."""
    def __init__(self, index: TileIndex, indices, augment=True, img_size=224):
        self.index = index
        self.idxs = indices
        self.size = img_size

        if augment:
            self.tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-10, 10), fit_output=False, cval=0, mode=cv2.BORDER_REFLECT, p=0.5),
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
        tile = img[y:y+ts, x:x+ts]
        tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tile = self.tf(image=tile)["image"]
        return tile, torch.from_numpy(labels), (path, x, y, ts)


class CLIPHead(nn.Module):
    """Frozen CLIP visual encoder with a small linear multilabel head."""
    def __init__(self, model_name="ViT-B-32", pretrained="openai", n_classes=8, device="cpu"):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.visual = self.model.visual
        for p in self.visual.parameters():
            p.requires_grad = False
        self.proj_dim = self.visual.output_dim
        self.head = nn.Linear(self.proj_dim, n_classes, bias=True)

    def forward(self, x):
        with torch.no_grad():
            feats = self.visual(x)
            feats = F.normalize(feats.float(), dim=-1)
        logits = self.head(feats)
        return logits, feats


def init_head_from_text(head_module, class_names, model_name="ViT-B-32", pretrained="openai", device="cpu"):
    """Initialize linear head from CLIP text prototypes averaged over prompt templates."""
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
    head_module.weight.data.copy_(W)
    head_module.bias.data.zero_()


@torch.no_grad()
def evaluate(model, loader, device, n_classes, progress_desc=None):
    """Compute per-class ROC-AUC, AP and best-F1; return metrics, macro, labels and probs."""
    model.eval()
    all_y, all_p = [], []
    it = tqdm(loader, desc=progress_desc, leave=False) if progress_desc else loader
    for x, y, _ in it:
        x = x.to(device)
        logits, _ = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_p.append(probs)
        all_y.append(y.numpy())
    Y = np.concatenate(all_y, 0) if all_y else np.zeros((0, n_classes), np.float32)
    P = np.concatenate(all_p, 0) if all_p else np.zeros((0, n_classes), np.float32)

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
        if not math.isnan(roc): roc_list.append(roc)
        if not math.isnan(ap): ap_list.append(ap)
        if not math.isnan(f1): f1_list.append(f1)
    macro = dict(
        roc_auc=float(np.nanmean(roc_list) if roc_list else np.nan),
        ap=float(np.nanmean(ap_list) if ap_list else np.nan),
        best_f1=float(np.nanmean(f1_list) if f1_list else np.nan),
    )
    return metrics, macro, Y, P


def compute_pos_weight(loader, n_classes):
    """Estimate class imbalance for BCEWithLogitsLoss pos_weight."""
    pos = np.zeros(n_classes, np.float64)
    neg = np.zeros(n_classes, np.float64)
    for _, y, _ in loader:
        y = y.numpy()
        pos += y.sum(axis=0)
        neg += (1 - y).sum(axis=0)
    pos = np.maximum(pos, 1.0)
    neg = np.maximum(neg, 1.0)
    return torch.tensor(neg / pos, dtype=torch.float32)


def save_prob_overlay(src_bgr, prob01, out_path, vis_thr=0.35, max_alpha=0.85, blur_ksize=0, color_map=cv2.COLORMAP_JET):
    """Single-class overlay with thresholded alpha and optional blur."""
    prob = np.clip(prob01.astype(np.float32), 0.0, 1.0)
    if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
        prob = cv2.GaussianBlur(prob, (blur_ksize, blur_ksize), 0)

    heat = (prob * 255).astype(np.uint8)
    color = cv2.applyColorMap(heat, color_map).astype(np.float32) / 255.0

    alpha = np.zeros_like(prob, dtype=np.float32)
    m = prob >= vis_thr
    alpha[m] = (prob[m] - vis_thr) / (1.0 - vis_thr)
    alpha = np.clip(alpha * max_alpha, 0.0, 1.0)
    alpha3 = np.repeat(alpha[..., None], 3, axis=2)

    src = src_bgr.astype(np.float32) / 255.0
    out = src * (1 - alpha3) + color * alpha3
    out = (out * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, out)


def save_composite_overlay(src_bgr, acc_C_H_W, class_names, out_path, vis_thr=0.35, max_alpha=0.85, blur_ksize=0, palette=LS_PALETTE):
    """Composite overlay for all classes at once using their LS colors."""
    src = src_bgr.astype(np.float32) / 255.0
    out = src.copy()
    C, H, W = acc_C_H_W.shape
    for c, name in enumerate(class_names):
        prob = np.clip(acc_C_H_W[c], 0.0, 1.0).astype(np.float32)
        if blur_ksize and blur_ksize >= 3 and blur_ksize % 2 == 1:
            prob = cv2.GaussianBlur(prob, (blur_ksize, blur_ksize), 0)
        alpha = np.zeros_like(prob, dtype=np.float32)
        m = prob >= vis_thr
        alpha[m] = (prob[m] - vis_thr) / (1.0 - vis_thr)
        alpha = np.clip(alpha * max_alpha, 0.0, 1.0)
        color_bgr = np.array(hex_to_bgr(palette.get(name, "#FF00FF")), dtype=np.float32) / 255.0
        color = np.ones((H, W, 3), np.float32) * color_bgr[None, None, :]
        alpha3 = np.repeat(alpha[..., None], 3, axis=2)
        out = out * (1 - alpha3) + color * alpha3
    out = (out * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(out_path, out)


def infer_batch(model, tiles_batch, device):
    """Forward a batch of tiles and return sigmoid probabilities."""
    x = torch.cat(tiles_batch, 0).to(device)
    logits, _ = model(x)
    probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def add_to_acc(acc, cnt, probs, coords, ts):
    """Accumulate tile probabilities into full-image maps with count norm."""
    C = acc.shape[0]
    for p, (x, y) in zip(probs, coords):
        for c in range(C):
            acc[c, y:y+ts, x:x+ts] += p[c]
        cnt[y:y+ts, x:x+ts] += 1.0


@torch.no_grad()
def build_heatmaps_for_split(model, index: TileIndex, split_indices, args, out_heat, out_overlay, out_overlay_all, device):
    """Build per-class and composite overlays for a dataset split."""
    model.eval()
    by_img = defaultdict(list)
    for i in split_indices:
        img_id, path, x, y, ts, _ = index.records[i]
        by_img[(img_id, path, ts)].append((x, y, i))

    for (img_id, path, ts), items in tqdm(by_img.items(), desc="Heatmaps"):
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        H, W = src.shape[:2]
        acc = np.zeros((len(index.classes), H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)

        batch_tiles, coords = [], []
        tf = A.Compose([
            A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_AREA),
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ToTensorV2()
        ])

        for (x, y, _) in items:
            img = cv2.cvtColor(src[y:y+ts, x:x+ts], cv2.COLOR_BGR2RGB)
            tile = tf(image=img)["image"].unsqueeze(0)
            batch_tiles.append(tile); coords.append((x, y))
            if len(batch_tiles) == args.batch_size:
                probs = infer_batch(model, batch_tiles, device)
                add_to_acc(acc, cnt, probs, coords, ts)
                batch_tiles, coords = [], []
        if batch_tiles:
            probs = infer_batch(model, batch_tiles, device)
            add_to_acc(acc, cnt, probs, coords, ts)

        cnt[cnt == 0] = 1.0
        acc /= cnt[None, :, :]

        if args.save_heatmaps or args.save_per_class:
            for c, name in enumerate(index.classes):
                h = np.clip(acc[c], 0.0, 1.0)
                if args.save_heatmaps:
                    cv2.imwrite(
                        os.path.join(out_heat, f"{os.path.basename(path)}__{name}.png"),
                        cv2.applyColorMap((h*255).astype(np.uint8), cv2.COLORMAP_JET)
                    )
                if args.save_per_class:
                    save_prob_overlay(
                        src, h,
                        os.path.join(out_overlay, f"{os.path.basename(path)}__{name}.png"),
                        vis_thr=min(args.vis_thrs_list), max_alpha=args.vis_max_alpha, blur_ksize=args.vis_blur
                    )

        for thr in args.vis_thrs_list:
            out_all_thr = os.path.join(out_overlay_all, f"T{int(thr*100)}")
            os.makedirs(out_all_thr, exist_ok=True)
            save_composite_overlay(
                src, acc, index.classes,
                os.path.join(out_all_thr, os.path.basename(path)),
                vis_thr=thr, max_alpha=args.vis_max_alpha, blur_ksize=args.vis_blur
            )


@torch.no_grad()
def infer_on_image_path(model, img_path, args, class_names):
    """Sliding-window inference over a single image; returns source BGR and per-class probability maps."""
    src = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if src is None:
        return None
    H, W = src.shape[:2]
    acc = np.zeros((len(class_names), H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    ts, stride = args.tile_size, args.stride

    tf = A.Compose([
        A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2()
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch, coords = [], []
    for y in range(0, max(1, H - ts + 1), stride):
        for x in range(0, max(1, W - ts + 1), stride):
            img = cv2.cvtColor(src[y:y+ts, x:x+ts], cv2.COLOR_BGR2RGB)
            tile = tf(image=img)["image"].unsqueeze(0)
            batch.append(tile); coords.append((x, y))
            if len(batch) == args.batch_size:
                probs = infer_batch(model, batch, device)
                add_to_acc(acc, cnt, probs, coords, ts)
                batch, coords = [], []
    if batch:
        probs = infer_batch(model, batch, device)
        add_to_acc(acc, cnt, probs, coords, ts)

    cnt[cnt == 0] = 1.0
    acc /= cnt[None, :, :]
    return src, acc


@torch.no_grad()
def run_infer_on_dir(model, args, class_names):
    """Run inference on all images in images_dir and save overlays for multiple thresholds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    out_base = stamp_out_dir(args.infer_out_dir or os.path.join(args.out_dir, "infer"))
    out_heat = os.path.join(out_base, "heatmaps_all");           os.makedirs(out_heat, exist_ok=True)
    out_overlay = os.path.join(out_base, "overlays_per_class");  os.makedirs(out_overlay, exist_ok=True)
    out_overlay_all_base = os.path.join(out_base, "overlays_composite"); os.makedirs(out_overlay_all_base, exist_ok=True)

    img_list = sorted([p for p in glob.glob(os.path.join(args.images_dir, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))])
    for p in tqdm(img_list, desc="Infer"):
        res = infer_on_image_path(model, p, args, class_names)
        if res is None:
            continue
        src, acc = res

        if args.save_heatmaps or args.save_per_class:
            for c, name in enumerate(class_names):
                h = np.clip(acc[c], 0.0, 1.0)
                if args.save_heatmaps:
                    cv2.imwrite(
                        os.path.join(out_heat, f"{os.path.basename(p)}__{name}.png"),
                        cv2.applyColorMap((h*255).astype(np.uint8), cv2.COLORMAP_JET)
                    )
                if args.save_per_class:
                    save_prob_overlay(
                        src, h,
                        os.path.join(out_overlay, f"{os.path.basename(p)}__{name}.png"),
                        vis_thr=min(args.vis_thrs_list), max_alpha=args.vis_max_alpha, blur_ksize=args.vis_blur
                    )

        for thr in args.vis_thrs_list:
            out_overlay_all = os.path.join(out_overlay_all_base, f"T{int(thr*100)}")
            os.makedirs(out_overlay_all, exist_ok=True)
            save_composite_overlay(
                src, acc, class_names,
                os.path.join(out_overlay_all, os.path.basename(p)),
                vis_thr=thr, max_alpha=args.vis_max_alpha, blur_ksize=args.vis_blur
            )


def duplicate_positive_indices(index: TileIndex, indices, dup_factor: int):
    """Simple oversampling: duplicate tiles that contain any positive class."""
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


def train(args):
    """Main entry: training or pure inference depending on args.mode."""
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

        model = CLIPHead(model_name=args.clip_model, pretrained=args.clip_ckpt, n_classes=len(class_names), device=device).to(device)
        init_head_from_text(model.head, class_names, model_name=args.clip_model, pretrained=args.clip_ckpt, device=device)

        ckpt_path = args.ckpt
        if not ckpt_path:
            subdirs = sorted([d for d in glob.glob(os.path.join(args.out_dir, "*")) if os.path.isdir(d)])
            for d in reversed(subdirs):
                cand = os.path.join(d, "model.pt")
                if os.path.isfile(cand):
                    ckpt_path = cand; break
        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError("Checkpoint not found. Pass --ckpt /path/to/model.pt or place model.pt under out_dir/<timestamp>/")

        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=False)

        print(f"[i] INFER with ckpt: {ckpt_path}")
        run_infer_on_dir(model, args, class_names)
        print("[OK] Inference done.")
        return

    args.out_dir = stamp_out_dir(args.out_dir)
    print(f"[i] results will be saved to: {args.out_dir}")

    index = TileIndex(args.images_dir, args.coco_json, tile_size=args.tile_size, stride=args.stride, cover_thr=args.cover_thr, keep_empty=False)

    tr_idx, va_idx = index.split_by_images(val_ratio=args.val_ratio, seed=42)
    tr_idx = duplicate_positive_indices(index, tr_idx, dup_factor=args.pos_dup)
    print(f"[i] train tiles: {len(tr_idx)} (dup={args.pos_dup}) | val tiles: {len(va_idx)} | classes: {index.classes}")

    ds_tr = TilesDataset(index, tr_idx, augment=True,  img_size=args.img_size)
    ds_va = TilesDataset(index, va_idx, augment=False, img_size=args.img_size)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CLIPHead(model_name=args.clip_model, pretrained=args.clip_ckpt, n_classes=len(index.classes), device=device).to(device)
    init_head_from_text(model.head, index.classes, model_name=args.clip_model, pretrained=args.clip_ckpt, device=device)

    vis_total, vis_train = count_params(model.visual)
    head_total, head_train = count_params(model.head)
    full_total, full_train = vis_total + head_total, vis_train + head_train
    print(f"[i] Visual encoder params: total={vis_total:,}, trainable={vis_train:,}")
    print(f"[i] Head params          : total={head_total:,}, trainable={head_train:,}")
    print(f"[i] Visual+Head total    : total={full_total:,}, trainable={full_train:,}")

    pos_weight = compute_pos_weight(dl_tr, len(index.classes)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)

    best_macro = -1.0
    ckpt_path = os.path.join(args.out_dir, "model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _ in tqdm(dl_tr, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device); y = y.to(device)
            logits, _ = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())

        _, M_tr, _, _ = evaluate(model, dl_tr, device, len(index.classes), progress_desc=None)
        _, M_va, _, _ = evaluate(model, dl_va, device, len(index.classes), progress_desc="Validate")
        print(f"[{epoch}] loss={np.mean(losses):.4f} | TRAIN mAP={M_tr['ap']:.3f} F1={M_tr['best_f1']:.3f} | VAL mAP={M_va['ap']:.3f} F1={M_va['best_f1']:.3f}")

        if M_va['ap'] > best_macro:
            best_macro = M_va['ap']
            torch.save({"state_dict": model.state_dict(), "classes": index.classes, "args": vars(args)}, ckpt_path)

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    m_va, M_va, _, _ = evaluate(model, dl_va, device, len(index.classes), progress_desc="Validate (final)")
    with open(os.path.join(args.out_dir, "metrics_val.json"), "w", encoding="utf-8") as f:
        json.dump({"per_class": {index.classes[c]: m_va[c] for c in m_va}, "macro": M_va}, f, ensure_ascii=False, indent=2)

    heat_dir = os.path.join(args.out_dir, "heatmaps_val");        os.makedirs(heat_dir, exist_ok=True)
    overlay_dir = os.path.join(args.out_dir, "overlays_val");     os.makedirs(overlay_dir, exist_ok=True)
    overlay_all = os.path.join(args.out_dir, "overlays_val_all"); os.makedirs(overlay_all, exist_ok=True)
    build_heatmaps_for_split(model, index, va_idx, args, out_heat=heat_dir, out_overlay=overlay_dir, out_overlay_all=overlay_all, device=device)

    run_infer_on_dir(model, args, index.classes)

    print(f"[OK] Done. Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","infer"], default="train")

    ap.add_argument("--images_dir", default="/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/images")
    ap.add_argument("--coco_json", default="/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/annotations/result_coco.json")
    ap.add_argument("--out_dir",   default="/home/sasha/Facade_segmentation/results")
    ap.add_argument("--infer_out_dir", default=None)

    ap.add_argument("--ckpt", default=None)

    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--stride",    type=int, default=256)
    ap.add_argument("--cover_thr", type=float, default=0.005)

    ap.add_argument("--img_size",  type=int, default=224)
    ap.add_argument("--batch_size",type=int, default=64)
    ap.add_argument("--epochs",    type=int, default=6)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.25)

    ap.add_argument("--clip_model", default="ViT-B-32")
    ap.add_argument("--clip_ckpt",  default="openai")

    ap.add_argument("--vis_thrs",       default="0.5,0.75,0.9")
    ap.add_argument("--vis_max_alpha",  type=float, default=0.85)
    ap.add_argument("--vis_blur",       type=int,   default=7)
    ap.add_argument("--save_heatmaps",  action="store_true")
    ap.add_argument("--save_per_class", action="store_true")

    ap.add_argument("--pos_dup", type=int, default=3)

    args = ap.parse_args()
    args.vis_thrs_list = parse_thrs(args.vis_thrs)
    train(args)
