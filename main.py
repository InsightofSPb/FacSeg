import os, json, math, argparse, hashlib, random, glob, shutil
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

torch.set_float32_matmul_precision('high')

# ---------- stats & palettes ----------
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


# ---------- utils ----------
def hex_to_bgr(hex_str: str):
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


def make_palette_from_names(class_names, fallback=LS_PALETTE):
    pal = []
    for n in class_names:
        if n in fallback:
            pal.append(hex_to_bgr(fallback[n]))
        else:
            h = int(hashlib.md5(n.encode('utf-8')).hexdigest()[:6], 16)
            pal.append(((h >> 0) & 255, (h >> 8) & 255, (h >> 16) & 255))
    return pal


def draw_legend(class_names, palette, max_h=None):
    pad = 8
    row_h = 22
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.55
    thickness = 1
    text_w = max([cv2.getTextSize(f"{i}: {n}", font, fs, thickness)[0][0] for i, n in enumerate(class_names, 1)] + [120])
    w = 44 + 10 + text_w + pad * 2
    h = pad*2 + row_h * len(class_names)
    img = np.full((h, w, 3), 255, np.uint8)
    y = pad
    for i, name in enumerate(class_names, 1):
        color = palette[i-1]
        cv2.rectangle(img, (pad, y), (pad+44, y+18), color, -1)
        cv2.rectangle(img, (pad, y), (pad+44, y+18), (0,0,0), 1)
        cv2.putText(img, f"{i}: {name}", (pad+54, y+15), font, fs, (0,0,0), thickness, cv2.LINE_AA)
        y += row_h
    if (max_h is not None) and h != max_h:
        scale = max_h / float(h)
        new_w = max(1, int(round(w * scale)))
        img = cv2.resize(img, (new_w, max_h), interpolation=cv2.INTER_AREA)
    return img


def attach_legend_to_right(image_bgr, class_names, palette):
    leg = draw_legend(class_names, palette, max_h=image_bgr.shape[0])
    spacer = np.full((image_bgr.shape[0], 8, 3), 255, np.uint8)
    return np.hstack([image_bgr, spacer, leg])


def make_comparison_panel(bgr, pred_idx, gt_idx, class_names, palette, alpha=0.45):
    H, W = bgr.shape[:2]

    def idx_to_color(idx):
        cm = np.zeros((H, W, 3), np.uint8)
        for i in range(1, len(class_names)+1):
            cm[idx == i] = palette[i-1]
        return cm

    pred_c = idx_to_color(pred_idx)
    gt_c   = idx_to_color(gt_idx)

    pred_over = cv2.addWeighted(bgr, 1-alpha, pred_c, alpha, 0)
    gt_over   = cv2.addWeighted(bgr, 1-alpha, gt_c, alpha, 0)

    # error map: green TP (fg), red FP, blue FN
    tp = (pred_idx == gt_idx) & (gt_idx > 0)
    fp = (pred_idx > 0) & (gt_idx == 0)
    fn = (pred_idx == 0) & (gt_idx > 0)
    err = bgr.copy()
    err[tp] = (0.6*err[tp] + np.array([60,180,60])).clip(0,255).astype(np.uint8)
    err[fp] = (0.5*err[fp] + np.array([40,40,200])).clip(0,255).astype(np.uint8)   # red-ish
    err[fn] = (0.5*err[fn] + np.array([200,40,40])).clip(0,255).astype(np.uint8)   # blue-ish

    def add_title(img, text):
        bar = img.copy()
        cv2.rectangle(bar, (0,0), (bar.shape[1], 28), (255,255,255), -1)
        cv2.putText(bar, text, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
        bar[28:] = img[:-28]
        return bar

    col1 = add_title(bgr, "Original")
    col2 = add_title(pred_over, "Pred overlay")
    col3 = add_title(gt_over, "GT overlay")
    col4 = add_title(err, "TP/FP/FN")

    panel = np.hstack([col1, col2, col3, col4])
    panel = attach_legend_to_right(panel, class_names, palette)
    return panel


def save_index_and_color_maps(acc_or_indices, class_names, out_mask_path, out_color_path,
                              vis_thr=0.35, palette=None, add_legend=False):
    """
    Сохраняет:
      - индексную маску PNG (0=фон, i=класс i)
      - цветную карту PNG (+ легенда опц.)
    """
    if acc_or_indices.ndim == 3:
        acc = acc_or_indices
        C, H, W = acc.shape
        max_prob = acc.max(axis=0)
        cls_idx = acc.argmax(axis=0) + 1
        cls_idx[max_prob < vis_thr] = 0
    else:
        cls_idx = acc_or_indices
        H, W = cls_idx.shape
        C = len(class_names)

    mask = (cls_idx.astype(np.uint16) if cls_idx.max() > 255 else cls_idx.astype(np.uint8))
    cv2.imwrite(out_mask_path, mask)

    if palette is None:
        palette = make_palette_from_names(class_names)
    bgr_map = np.zeros((H, W, 3), np.uint8)
    for i in range(1, len(class_names) + 1):
        bgr = palette[i - 1]
        bgr_map[cls_idx == i] = bgr
    if add_legend:
        bgr_map = attach_legend_to_right(bgr_map, class_names, palette)
    cv2.imwrite(out_color_path, bgr_map)
    return mask, bgr_map


def save_overlay(bgr_src, bgr_map, out_path, alpha=0.45, blur=0, add_legend=False, class_names=None, palette=None):
    Hs, Ws = bgr_src.shape[:2]
    Hm, Wm = bgr_map.shape[:2]
    if (Hs, Ws) != (Hm, Wm):
        bgr_map_small = cv2.resize(bgr_map, (Ws, Hs), interpolation=cv2.INTER_NEAREST)
    else:
        bgr_map_small = bgr_map
    vis = bgr_src.copy()
    if blur and blur >= 1:
        bgr_map_small = cv2.GaussianBlur(bgr_map_small, (0, 0), blur)
    cv2.addWeighted(bgr_map_small, alpha, vis, 1 - alpha, 0, vis)
    if add_legend and class_names is not None and palette is not None:
        vis = attach_legend_to_right(vis, class_names, palette)
    cv2.imwrite(out_path, vis)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stamp_out_dir(base: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, ts)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def safe_path(images_dir: str, coco_file_name: str):
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
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def format_int(n: int) -> str:
    return f"{n:,}"


def maybe_compile(model, do_compile: bool, mode: str = "default", backend: str = "inductor"):
    if not do_compile:
        return model
    if not hasattr(torch, "compile"):
        print("[!] torch.compile is not available in this PyTorch. Skipping compile.")
        return model
    try:
        model = torch.compile(model, mode=mode, backend=backend)
        print(f"[i] Model compiled (mode={mode}, backend={backend}).")
    except Exception as e:
        print(f"[!] torch.compile failed: {e}. Using eager.")
    return model

# ---------- NEW: robust (un)wrap + flexible state_dict I/O ----------

def unwrap_model(m: nn.Module) -> nn.Module:
    """Return the underlying real module (unwrapping torch.compile/DDP wrappers)."""
    # unwrap torch.compile wrapper
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    # unwrap DDP/DataParallel
    if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        m = m.module
    return m


def get_state_dict_unwrapped(m: nn.Module) -> dict:
    return unwrap_model(m).state_dict()


def strip_prefixes(sd: dict) -> dict:
    """Strip common wrapper prefixes like _orig_mod. and module."""
    new_sd = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("_orig_mod."):
            k2 = k2[len("_orig_mod."):]
        if k2.startswith("module."):
            k2 = k2[len("module."):]
        new_sd[k2] = v
    return new_sd


def load_state_dict_flexible(model: nn.Module, payload: dict, strict: bool = True):
    """Load from a checkpoint payload handling possible wrappers/prefixes."""
    if isinstance(payload, dict) and "state_dict" in payload:
        sd = payload["state_dict"]
    else:
        sd = payload
    sd = strip_prefixes(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if strict and (missing or unexpected):
        print(f"[!] Non-strict load due to key mismatch. Missing={len(missing)}, Unexpected={len(unexpected)}")
    else:
        print("[i] State dict loaded.")
    return missing, unexpected


# ---------- checkpoint manager ----------
class CheckpointManager:
    """
    Сохраняет топ-K чекпоинтов по ключу (miou_excl_bg, pixel_acc, f1_macro_excl_bg, -loss).
    Если >K, удаляет наихудшие. Имя файла содержит epoch и метрики.
    """
    def __init__(self, out_dir, topk=3, prefix="seg", greater_is_better=True):
        self.dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(self.dir, exist_ok=True)
        self.topk = topk
        self.prefix = prefix
        self.items = []  # list of (key_tuple, path)

    @staticmethod
    def _key(metrics: dict, loss: float):
        miou = float(metrics.get("miou_excl_bg", 0.0))
        acc  = float(metrics.get("pixel_acc", 0.0))
        f1   = float(metrics.get("f1_macro_excl_bg", 0.0))
        return (miou, acc, f1, -float(loss))

    def save(self, model_state_dict, epoch: int, metrics: dict, loss: float):
        miou = metrics.get("miou_excl_bg", 0.0)
        acc  = metrics.get("pixel_acc", 0.0)
        f1   = metrics.get("f1_macro_excl_bg", 0.0)
        fname = f"{self.prefix}_ep{epoch:03d}_miou_{miou:.3f}_acc_{acc:.3f}_f1_{f1:.3f}_loss_{loss:.4f}.pt"
        path = os.path.join(self.dir, fname)
        torch.save({"state_dict": model_state_dict, "metrics": metrics, "epoch": epoch, "loss": loss}, path)

        key = self._key(metrics, loss)
        self.items.append((key, path))
        # sort best -> worst
        self.items.sort(key=lambda t: t[0], reverse=True)
        # prune
        while len(self.items) > self.topk:
            _, worst_path = self.items.pop(-1)
            try:
                os.remove(worst_path)
                print(f"[i] Pruned checkpoint: {os.path.basename(worst_path)}")
            except OSError:
                pass
        print(f"[i] Saved checkpoint: {os.path.basename(path)} (top{self.topk} manager)")


# ------------------------------
# CLIP-tile classification part
# ------------------------------
class TileIndex:
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

            for y in range(0, max(1, H - self.tile_size + 1), self.stride):
                for x in range(0, max(1, W - self.tile_size + 1), self.stride):
                    tile_area = float(self.tile_size * self.tile_size)
                    labels = np.zeros(self.C, np.float32)
                    for j, cid in enumerate(self.class_ids):
                        inter = class_masks[cid][y:y + self.tile_size, x:x + self.tile_size].sum()
                        if inter / tile_area >= cover_thr:
                            labels[j] = 1.0
                    if keep_empty or labels.sum() > 0:
                        self.records.append((im["id"], path, x, y, self.tile_size, labels))

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


class CLIPHead(nn.Module):
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
    else:
        for m in head_module.modules():
            if isinstance(m, nn.Linear) and m.out_features == W.shape[0]:
                m.weight.data.copy_(W)
                break
    for m in head_module.modules():
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)


@torch.no_grad()
def evaluate_clip(model, loader, device, n_classes, progress_desc=None):
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
            roc, ap, f1m = np.nan, np.nan, np.nan
        else:
            try:
                roc = roc_auc_score(y, p)
            except Exception:
                roc = np.nan
            ap = average_precision_score(y, p)
            pr, rc, _ = precision_recall_curve(y, p)
            f1m = (2 * pr * rc / (pr + rc + 1e-9)).max()
        metrics[c] = dict(roc_auc=float(roc), ap=float(ap), best_f1=float(f1m))
        if not math.isnan(roc):
            roc_list.append(roc)
        if not math.isnan(ap):
            ap_list.append(ap)
        if not math.isnan(f1m):
            f1_list.append(f1m)
    macro = dict(
        roc_auc=float(np.nanmean(roc_list) if roc_list else np.nan),
        ap=float(np.nanmean(ap_list) if ap_list else np.nan),
        best_f1=float(np.nanmean(f1_list) if f1_list else np.nan),
    )
    return metrics, macro, Y, P


def compute_pos_weight(loader, n_classes):
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
    x = torch.cat(tiles_batch, 0).to(device)
    logits, _ = model(x)
    probs = torch.sigmoid(logits).cpu().detach().numpy()
    return probs


def cosine_kernel(ts: int):
    win1 = np.hanning(ts)
    w = np.outer(win1, win1).astype(np.float32)
    w = w / max(w.max(), 1e-6)
    return w


def save_composite_overlay(src_bgr, prob_C_H_W, class_names, out_path, palette=LS_PALETTE, vis_thr=0.5, alpha=0.7, blur=0):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    out_base  = stamp_out_dir(os.path.join(args.out_dir, "clip_infer"))
    out_comp  = os.path.join(out_base, "overlays_composite")
    out_idx   = os.path.join(out_base, "masks_idx")
    out_color = os.path.join(out_base, "colors")
    out_over  = os.path.join(out_base, "overlays")
    for d in [out_comp, out_idx, out_color, out_over]:
        os.makedirs(d, exist_ok=True)

    tf = A.Compose([
        A.Resize(args.img_size, args.img_size, interpolation=cv2.INTER_AREA),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2()
    ])

    ts = args.tile_size
    stride = args.stride
    kernel_full = cosine_kernel(ts)
    palette = make_palette_from_names(class_names)

    img_list = sorted([p for p in glob.glob(os.path.join(args.test_dir, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))])
    for path in tqdm(img_list, desc="CLIP infer"):
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        if src is None:
            continue
        H, W = src.shape[:2]
        acc = np.zeros((len(class_names), H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)

        batch_tiles, coords = [], []
        y_stops = [0] if H < ts else list(range(0, H - ts + 1, stride))
        x_stops = [0] if W < ts else list(range(0, W - ts + 1, stride))

        for y in y_stops:
            for x in x_stops:
                tile_bgr = src[y:min(y + ts, H), x:min(x + ts, W)]
                img = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
                tile = tf(image=img)["image"].unsqueeze(0)
                batch_tiles.append(tile)
                hh, ww = tile_bgr.shape[:2]
                coords.append((x, y, ww, hh))
                if len(batch_tiles) == args.batch_size:
                    probs = infer_batch_clip(model, batch_tiles, device)
                    for p, (xx, yy, ww, hh) in zip(probs, coords):
                        k = kernel_full[:hh, :ww]
                        for c in range(len(class_names)):
                            acc[c, yy:yy + hh, xx:xx + ww] += p[c] * k
                        cnt[yy:yy + hh, xx:xx + ww] += k
                    batch_tiles, coords = [], []

        if batch_tiles:
            probs = infer_batch_clip(model, batch_tiles, device)
            for p, (xx, yy, ww, hh) in zip(probs, coords):
                k = kernel_full[:hh, :ww]
                for c in range(len(class_names)):
                    acc[c, yy:yy + hh, xx:xx + ww] += p[c] * k
                cnt[yy:yy + hh, xx:xx + ww] += k

        cnt[cnt == 0] = 1.0
        acc /= cnt[None, :, :]

        save_composite_overlay(src, acc, class_names,
                               os.path.join(out_comp, os.path.basename(path)),
                               palette=LS_PALETTE, vis_thr=args.vis_thr, alpha=args.vis_max_alpha, blur=args.vis_blur)

        base = os.path.splitext(os.path.basename(path))[0]
        idx_path   = os.path.join(out_idx,   base + ".png")
        color_path = os.path.join(out_color, base + ".png")
        over_path  = os.path.join(out_over,  base + ".jpg")
        _, color_map = save_index_and_color_maps(acc, class_names, idx_path, color_path,
                                                 vis_thr=args.vis_thr, palette=palette, add_legend=True)
        save_overlay(src, color_map, over_path, alpha=args.vis_max_alpha, blur=args.vis_blur,
                     add_legend=True, class_names=class_names, palette=palette)


def duplicate_positive_indices(index: TileIndex, indices, dup_factor: int):
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
    h, w = mask_hw.shape
    out = np.zeros((h, w, 3), np.uint8)
    for cid, name in id_to_name.items():
        if cid == 0:
            continue
        color = hex_to_bgr(LS_PALETTE.get(name, "#FF00FF"))
        out[mask_hw == cid] = color
    return out


# ------------------------------
# Dense Segmentation (COCO -> masks)
# ------------------------------
class SegIndex:
    """
    Собирает плотные маски из COCO полигонов.
    Идентификаторы категорий ремапятся в 1..C, фон=0.
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
    Train: random scale, pad, random crop, flips, mild color jitter & rotation.
    Val/Test: resize long side, pad to square без кропа.
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
        self.norm_mode = norm_mode

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


# ------------------------------
# Torchvision segmentation models
# ------------------------------
def get_seg_model(name: str, n_classes: int, weights_tag: str):
    """
    Для torchvision>=0.15 при предобученных весах aux_loss должен быть True.
    Мы подменяем main/aux головы на (C+1) каналов.
    """
    name = name.lower()

    def _replace_heads(model):
        in_ch = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_ch, n_classes + 1, kernel_size=1)
        if getattr(model, "aux_classifier", None) is not None:
            in_ch_aux = model.aux_classifier[-1].in_channels
            model.aux_classifier[-1] = nn.Conv2d(in_ch_aux, n_classes + 1, kernel_size=1)
        return model

    use_weights = (weights_tag == "default")
    aux_loss_flag = True if use_weights else False

    if name == "deeplabv3_resnet50":
        weights = tvm.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if use_weights else None
        model = tvm.segmentation.deeplabv3_resnet50(weights=weights, aux_loss=aux_loss_flag)
        return _replace_heads(model)

    if name == "deeplabv3_resnet101":
        weights = tvm.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT if use_weights else None
        model = tvm.segmentation.deeplabv3_resnet101(weights=weights, aux_loss=aux_loss_flag)
        return _replace_heads(model)

    if name == "fcn_resnet50":
        weights = tvm.segmentation.FCN_ResNet50_Weights.DEFAULT if use_weights else None
        model = tvm.segmentation.fcn_resnet50(weights=weights, aux_loss=aux_loss_flag)
        return _replace_heads(model)

    if name == "fcn_resnet101":
        weights = tvm.segmentation.FCN_ResNet101_Weights.DEFAULT if use_weights else None
        model = tvm.segmentation.fcn_resnet101(weights=weights, aux_loss=aux_loss_flag)
        return _replace_heads(model)

    raise ValueError("Unsupported seg_model.")


def _seg_metrics_from_confmat(conf: np.ndarray):
    """
    IoU по классам + средние только по присутствующим в GT (реалистичнее на маленькой валидации),
    а также macro precision/recall/F1 по foreground-классам.
    """
    tp = np.diag(conf).astype(np.float64)
    fp = conf.sum(axis=0) - tp
    fn = conf.sum(axis=1) - tp
    denom = tp + fp + fn + 1e-9
    iou = tp / denom

    present = (tp + fn) > 0               # класс есть в GT
    miou_incl_bg = float(np.nanmean(iou[present])) if present.any() else 0.0

    present_fg = present.copy()
    if len(present_fg) > 0:
        present_fg[0] = False             # исключить фон
    miou_excl_bg = float(np.nanmean(iou[present_fg])) if present_fg.any() else 0.0

    # Macro P/R/F1 по fg
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    if present_fg.any():
        prec_macro_fg = float(np.nanmean(prec[present_fg]))
        rec_macro_fg  = float(np.nanmean(rec[present_fg]))
        f1_macro_fg   = float(np.nanmean(f1[present_fg]))
    else:
        prec_macro_fg = rec_macro_fg = f1_macro_fg = 0.0

    pix_acc = float(tp.sum() / (conf.sum() + 1e-9))
    return dict(
        per_class_iou=iou.tolist(),
        miou_incl_bg=miou_incl_bg,
        miou_excl_bg=miou_excl_bg,
        pixel_acc=pix_acc,
        precision_macro_excl_bg=prec_macro_fg,
        recall_macro_excl_bg=rec_macro_fg,
        f1_macro_excl_bg=f1_macro_fg
    )


def seg_evaluate(model, loader, num_classes_plus_bg: int, device,
                 dump_dir: str = None, dump_max: int = 0, class_names=None, denorm="imagenet"):
    """
    Оценка + (опц.) сохранение dump_max примеров:
      - masks_gt_idx/ (индексы GT)
      - masks_pred_idx/
      - colors_pred/ (+ легенда)
      - overlays/ (+ легенда)
      - panels/ (сравнение на одной картинке, +легенда)
    denorm: 'imagenet' | 'clip' — как денормализовать превью.
    """
    model.eval()
    conf = np.zeros((num_classes_plus_bg, num_classes_plus_bg), np.int64)
    saved = 0
    palette = make_palette_from_names(class_names or [str(i) for i in range(1, num_classes_plus_bg)])
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if denorm == "imagenet" else (CLIP_MEAN, CLIP_STD)

    if dump_dir and dump_max > 0:
        for d in ["masks_gt_idx", "masks_pred_idx", "colors_pred", "overlays", "panels"]:
            os.makedirs(os.path.join(dump_dir, d), exist_ok=True)

    with torch.no_grad():
        for x, y, metas in tqdm(loader, desc="Seg Validate", leave=False):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)["out"]
            pred = logits.argmax(1)

            for yi, pi in zip(y, pred):
                y1 = yi.view(-1).cpu().numpy()
                p1 = pi.view(-1).cpu().numpy()
                m = (y1 >= 0) & (y1 < num_classes_plus_bg)
                y1 = y1[m]; p1 = p1[m]
                cm = np.bincount(y1 * num_classes_plus_bg + p1, minlength=num_classes_plus_bg ** 2)
                conf += cm.reshape(num_classes_plus_bg, num_classes_plus_bg)

            if dump_dir and saved < dump_max:
                for i in range(x.shape[0]):
                    if saved >= dump_max:
                        break
                    xi = x[i].detach().cpu().numpy().transpose(1, 2, 0)
                    xi = np.clip(xi * np.array(std)[None, None, :] + np.array(mean)[None, None, :], 0, 1)
                    xi = (xi * 255.0).astype(np.uint8)
                    bgr = cv2.cvtColor(xi, cv2.COLOR_RGB2BGR)

                    pi = pred[i].detach().cpu().numpy().astype(np.uint8)
                    yi = y[i].detach().cpu().numpy().astype(np.int16)
                    yi[yi < 0] = 0
                    yi = yi.astype(np.uint8)

                    base = os.path.splitext(os.path.basename(metas if isinstance(metas, str) else metas[i]))[0]
                    cv2.imwrite(os.path.join(dump_dir, "masks_gt_idx",   base + ".png"), yi)
                    cv2.imwrite(os.path.join(dump_dir, "masks_pred_idx", base + ".png"), pi)

                    idx_path   = os.path.join(dump_dir, "masks_pred_idx", base + ".png")
                    color_path = os.path.join(dump_dir, "colors_pred",   base + ".png")
                    over_path  = os.path.join(dump_dir, "overlays",      base + ".jpg")
                    _, color_map = save_index_and_color_maps(
                        pi,
                        class_names or [str(i) for i in range(1, num_classes_plus_bg)],
                        out_mask_path=idx_path,
                        out_color_path=color_path,
                        vis_thr=0.0, palette=palette, add_legend=True
                    )
                    save_overlay(bgr, color_map, over_path, alpha=0.45, blur=0,
                                 add_legend=True, class_names=class_names, palette=palette)

                    panel = make_comparison_panel(bgr, pi, yi, class_names or [str(i) for i in range(1, num_classes_plus_bg)], palette)
                    cv2.imwrite(os.path.join(dump_dir, "panels", base + ".jpg"), panel)
                    saved += 1

    return _seg_metrics_from_confmat(conf)


def sliding_window_seg_infer(model, bgr, size=768, stride=512, device="cpu", norm_mode="imagenet"):
    """
    Слайдинг-инференс (разные нормировки для torchvision/OVSeg).
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
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if norm_mode == "imagenet" else (CLIP_MEAN, CLIP_STD)
        img = (img - np.array(mean)) / np.array(std)
        ten = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        return ten

    tiles = [(y, x) for y in range(0, Hp - size + 1, stride) for x in range(0, Wp - size + 1, stride)]
    pbar = tqdm(total=len(tiles), desc="Seg tiles", leave=False)

    model.eval()
    with torch.no_grad():
        for (y, x) in tiles:
            tile = rgb_pad[y:y + size, x:x + size]
            out = model(prep_tile(tile))["out"]
            out_np = out.squeeze(0).cpu().numpy()
            if acc is None:
                acc = np.zeros((out_np.shape[0], Hp, Wp), np.float32)
            for c in range(out_np.shape[0]):
                acc[c, y:y + size, x:x + size] += out_np[c] * kernel
            wsum[y:y + size, x:x + size] += kernel
            pbar.update(1)
    pbar.close()

    wsum[wsum == 0] = 1.0
    acc /= wsum[None, :, :]

    acc = acc[:, :H, :W]
    return acc


def seg_infer_save(model, args, seg_index: SegIndex):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_base = stamp_out_dir(os.path.join(args.out_dir, "seg_infer"))
    out_idx   = os.path.join(out_base, "masks_idx")
    out_color = os.path.join(out_base, "colors")
    out_over  = os.path.join(out_base, "overlays")
    os.makedirs(out_idx, exist_ok=True)
    os.makedirs(out_color, exist_ok=True)
    os.makedirs(out_over, exist_ok=True)

    img_list = sorted([p for p in glob.glob(os.path.join(args.test_dir, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))])
    class_names = [seg_index.idx_to_name[i] for i in sorted(seg_index.idx_to_name.keys())]
    palette = make_palette_from_names(class_names)

    for path in tqdm(img_list, desc="Seg infer (images)"):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        logits = sliding_window_seg_infer(model, bgr, size=args.seg_tile_size, stride=args.seg_stride, device=device, norm_mode="imagenet")
        pred = logits.argmax(0).astype(np.uint8)

        base = os.path.splitext(os.path.basename(path))[0]
        idx_path   = os.path.join(out_idx,   base + ".png")
        color_path = os.path.join(out_color, base + ".png")
        over_path  = os.path.join(out_over,  base + ".jpg")

        cv2.imwrite(idx_path, pred)
        _, color_map = save_index_and_color_maps(pred, class_names, idx_path, color_path,
                                                 vis_thr=0.0, palette=palette, add_legend=True)
        save_overlay(bgr, color_map, over_path, alpha=0.45, blur=0,
                     add_legend=True, class_names=class_names, palette=palette)


def seg_duplicate_indices(indices, dup_factor: int):
    if dup_factor <= 1:
        return indices
    return [i for i in indices for _ in range(dup_factor)]


def seg_duplicate_positive(seg_index: SegIndex, indices, dup_factor: int):
    """
    Дублируем только те картинки, где в маске есть хоть один непустой класс (не фон).
    """
    if dup_factor <= 1:
        return indices
    out = []
    for i in indices:
        rec = seg_index.items[i]
        pos = (rec["mask"].max() > 0)
        if pos:
            out.extend([i] * dup_factor)
        else:
            out.append(i)
    return out


def compute_ce_class_weights(seg_index: SegIndex, indices):
    """
    Веса классов для CrossEntropy по частотам пикселей на трейне.
    Возвращает torch.tensor размера [C+1] (включая фон).
    """
    C = seg_index.num_classes + 1
    hist = np.zeros(C, np.float64)
    for i in indices:
        m = seg_index.items[i]["mask"]
        h = np.bincount(m.ravel(), minlength=C).astype(np.float64)
        hist += h
    hist = np.maximum(hist, 1.0)
    freq = hist / hist.sum()
    w = 1.0 / freq
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ------------------------------
# OVSeg: CLIP-ViT backbone + lightweight decoder
# ------------------------------
def _get_positional_embedding_param(visual):
    pe = getattr(visual, "positional_embedding", None)
    if pe is None:
        pe = getattr(visual, "pos_embed", None)
    if pe is None:
        raise RuntimeError("CLIP visual has no positional embedding attribute")
    if pe.dim() == 2:
        pe = pe.unsqueeze(0)   # [1, n_ctx, C]
    return pe


def _interpolate_positional_embedding(visual, grid_h, grid_w, device, dtype):
    pe = _get_positional_embedding_param(visual).to(device=device, dtype=dtype)  # [1, n_ctx, C]
    class_pe = pe[:, :1, :]           # [1, 1, C]
    patch_pe = pe[:, 1:, :]           # [1, N, C]
    old_n = patch_pe.shape[1]
    old_g = int(round(old_n ** 0.5))

    if old_g * old_g == old_n:
        patch_pe = patch_pe.reshape(1, old_g, old_g, -1).permute(0, 3, 1, 2)     # [1, C, old_g, old_g]
        patch_pe = F.interpolate(patch_pe, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)  # [1, N', C]
    else:
        new_n = grid_h * grid_w
        patch_pe = patch_pe.permute(0, 2, 1)                                      # [1, C, N]
        patch_pe = F.interpolate(patch_pe, size=new_n, mode="linear", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 1)                                      # [1, N', C]

    return torch.cat([class_pe, patch_pe], dim=1)                                 # [1, 1+N', C]


def _clip_spatial_tokens(visual, x: torch.Tensor):
    B, _, H, W = x.shape
    x = visual.conv1(x)                              # [B, C, H', W']
    gh, gw = x.shape[-2], x.shape[-1]

    x = x.reshape(B, x.shape[1], gh * gw).permute(0, 2, 1)  # [B, N, C]
    class_emb = visual.class_embedding.to(x.dtype)[None, None, :].expand(B, 1, -1)  # [B,1,C]
    pos = _interpolate_positional_embedding(visual, gh, gw, x.device, x.dtype)      # [1,1+N,C]
    x = torch.cat([class_emb, x], dim=1) + pos                                      # [B,1+N,C]

    ln_pre = getattr(visual, "ln_pre", nn.Identity())
    x = ln_pre(x)

    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)

    ln_post = getattr(visual, "ln_post", nn.Identity())
    x = ln_post(x)

    x = x[:, 1:, :]                                   # [B, N, C]
    x = x.permute(0, 2, 1).reshape(B, x.shape[-1], gh, gw)  # [B, C, H', W']
    return x


class OvSegDecoder(nn.Module):
    def __init__(self, in_dim, out_classes, decoder_channels=256, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, decoder_channels, 1)
        self.block = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(decoder_channels, out_classes, 1)

    def forward(self, feat_map, out_hw):
        x = self.proj(feat_map)
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x = self.block(x)
        logits = self.head(x)
        return logits


class OvSegModel(nn.Module):
    """
    CLIP-VиT бэкбон (open_clip) + лёгкий декодер -> dict(out=логиты) как у torchvision.
    """
    def __init__(self, model_name, pretrained, n_classes, freeze_backbone=False,
                 decoder_channels=256, decoder_dropout=0.0, device="cpu"):
        super().__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.visual = self.clip_model.visual
        if freeze_backbone:
            for p in self.visual.parameters():
                p.requires_grad = False
        in_dim = getattr(self.visual, "width", None)
        if in_dim is None:
            in_dim = int(self.visual.ln_post.normalized_shape[0])
        self.decoder = OvSegDecoder(in_dim, n_classes + 1, decoder_channels, dropout=decoder_dropout)

    def forward(self, x):
        feat = _clip_spatial_tokens(self.visual, x)                   # [B, C, H', W']
        logits = self.decoder(feat, out_hw=(x.shape[-2], x.shape[-1]))  # [B, C+1, H, W]
        return {"out": logits}


# ------------------------------
# Training / Inference drivers
# ------------------------------
def plan_dump_epochs(epochs: int, save_every: int, max_dumps: int):
    max_dumps = max(0, min(max_dumps, 10))
    if max_dumps == 0:
        return []
    if save_every and save_every > 0:
        arr = [e for e in range(1, epochs + 1) if e % save_every == 0]
        return arr[:max_dumps]
    # равномерно распределяем не более max_dumps эпох
    if epochs <= max_dumps:
        return list(range(1, epochs + 1))
    pts = np.linspace(1, epochs, max_dumps)
    arr = sorted({int(round(x)) for x in pts})
    return arr[:max_dumps]


def sliding_window_ovseg_infer(model, bgr, size=768, stride=512, device="cpu"):
    return sliding_window_seg_infer(model, bgr, size=size, stride=stride, device=device, norm_mode="clip")


def ovseg_infer_save(model, args, seg_index: SegIndex):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_base = stamp_out_dir(os.path.join(args.out_dir, "ovseg_infer"))
    out_idx   = os.path.join(out_base, "masks_idx")
    out_color = os.path.join(out_base, "colors")
    out_over  = os.path.join(out_base, "overlays")
    os.makedirs(out_idx, exist_ok=True)
    os.makedirs(out_color, exist_ok=True)
    os.makedirs(out_over, exist_ok=True)

    img_list = sorted([p for p in glob.glob(os.path.join(args.test_dir, "*")) if p.lower().endswith((".jpg", ".jpeg", ".png"))])
    class_names = [seg_index.idx_to_name[i] for i in sorted(seg_index.idx_to_name.keys())]
    palette = make_palette_from_names(class_names)

    for path in tqdm(img_list, desc="OVSeg infer (images)"):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        logits = sliding_window_ovseg_infer(model, bgr, size=args.seg_tile_size, stride=args.seg_stride, device=device)
        pred = logits.argmax(0).astype(np.uint8)

        base = os.path.splitext(os.path.basename(path))[0]
        idx_path   = os.path.join(out_idx,   base + ".png")
        color_path = os.path.join(out_color, base + ".png")
        over_path  = os.path.join(out_over,  base + ".jpg")

        cv2.imwrite(idx_path, pred)
        _, color_map = save_index_and_color_maps(pred, class_names, idx_path, color_path,
                                                 vis_thr=0.0, palette=palette, add_legend=True)
        save_overlay(bgr, color_map, over_path, alpha=0.45, blur=0,
                     add_legend=True, class_names=class_names, palette=palette)


def train(args):
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- CLIP tiles ----------
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

        vis_total, vis_train = count_params(model.visual)
        head_total, head_train = count_params(model.head)
        full_total, full_train = vis_total + head_total, vis_train + head_train
        print(f"[i] Visual encoder params: total={format_int(vis_total)}, trainable={format_int(vis_train)}")
        print(f"[i] Head params          : total={format_int(head_total)}, trainable={format_int(head_train)}")
        print(f"[i] Visual+Head total    : total={format_int(full_total)}, trainable={format_int(full_train)}")

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
        # Flexible load (handles prefixes)
        load_state_dict_flexible(model, ckpt, strict=False)

        print(f"[i] CLIP infer with ckpt: {ckpt_path}")
        clip_infer_on_dir(model, args, class_names)
        print("[OK] CLIP inference finished.")
        return

    # ---------- torchvision seg inference ----------
    if args.mode == "seg_infer":
        if not args.ckpt or not os.path.isfile(args.ckpt):
            raise FileNotFoundError("Provide segmentation checkpoint with --ckpt")
        seg_index = SegIndex(args.images_dir, args.coco_json)
        model = get_seg_model(args.seg_model, seg_index.num_classes, weights_tag="none").to(device)
        try:
            sd = torch.load(args.ckpt, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(args.ckpt, map_location=device)
        load_state_dict_flexible(model, sd, strict=True)

        model = maybe_compile(model, args.compile, mode=args.compile_mode, backend=args.compile_backend)

        seg_total, seg_trainable = count_params(model)
        print(f"[i] Segmentation model params: total={format_int(seg_total)}, trainable={format_int(seg_trainable)}")

        seg_infer_save(model, args, seg_index)
        print("[OK] Segmentation inference finished.")
        return

    # ---------- torchvision seg train ----------
    if args.mode == "seg_train":
        out_dir = stamp_out_dir(args.out_dir)
        print(f"[i] results will be saved to: {out_dir}")

        seg_index = SegIndex(args.images_dir, args.coco_json)
        tr_idx_orig, va_idx = seg_index.split_by_images(val_ratio=args.val_ratio, seed=42)

        tr_idx = seg_duplicate_indices(tr_idx_orig, args.seg_dup)
        print(f"[i] seg train images: {len(tr_idx_orig)} (dup x{args.seg_dup} -> {len(tr_idx)}) | val images: {len(va_idx)}")

        ds_tr = SegDataset(seg_index, tr_idx, train=True,  size=args.seg_img_size, norm_mode="imagenet")
        ds_va = SegDataset(seg_index, va_idx, train=False, size=args.seg_img_size, norm_mode="imagenet")

        num_workers = 2
        dl_tr = DataLoader(ds_tr, batch_size=args.seg_batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
        dl_va = DataLoader(ds_va, batch_size=args.seg_batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

        model = get_seg_model(args.seg_model, seg_index.num_classes, weights_tag=args.seg_weights).to(device)
        model = maybe_compile(model, args.compile, mode=args.compile_mode, backend=args.compile_backend)

        seg_total, seg_trainable = count_params(model)
        print(f"[i] Segmentation model params: total={format_int(seg_total)}, trainable={format_int(seg_trainable)}")

        class_weights = compute_ce_class_weights(seg_index, tr_idx).to(device)
        print(f"[i] CE class weights: {class_weights.cpu().numpy().round(3)}")
        criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.seg_lr, weight_decay=1e-4)

        ckpt_mgr = CheckpointManager(out_dir, topk=3, prefix="seg")

        class_names = [seg_index.idx_to_name[i] for i in sorted(seg_index.idx_to_name.keys())]
        planned_dumps = plan_dump_epochs(args.epochs, args.val_save_every, args.val_vis_max_dumps)
        print(f"[i] Planned val dumps at epochs: {planned_dumps} (each up to {min(args.val_vis_n,10)} imgs)")

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            pbar = tqdm(dl_tr, desc=f"Seg Epoch {epoch}/{args.epochs}", leave=False)
            for x, y, _ in pbar:
                x = x.to(device)
                y = y.to(device)
                out = model(x)["out"]
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                pbar.set_postfix(loss=f"{np.mean(losses):.4f}")
            pbar.close()
            train_loss = float(np.mean(losses)) if losses else 0.0

            dump_dir = None
            dump_max = 0
            if epoch in planned_dumps:
                dump_dir = os.path.join(out_dir, f"val_dump_ep_{epoch:03d}")
                dump_max = min(args.val_vis_n, 10)

            metrics = seg_evaluate(model, dl_va, seg_index.num_classes + 1, device,
                                   dump_dir=dump_dir, dump_max=dump_max,
                                   class_names=class_names, denorm="imagenet")
            print(f"[{epoch}] seg loss={train_loss:.4f} | mIoU(excl_bg)={metrics['miou_excl_bg']:.3f} | "
                  f"F1_fg={metrics['f1_macro_excl_bg']:.3f} | pixel_acc={metrics['pixel_acc']:.3f}")

            # save UNWRAPPED state_dict to avoid _orig_mod.* keys
            ckpt_mgr.save(get_state_dict_unwrapped(model), epoch=epoch, metrics=metrics, loss=train_loss)

        print(f"[OK] Segmentation training finished. Checkpoints in {os.path.join(out_dir,'checkpoints')}")
        return

    # ---------- OVSeg train ----------
    if args.mode == "ovseg_train":
        out_dir = stamp_out_dir(args.out_dir)
        print(f"[i] results will be saved to: {out_dir}")

        seg_index = SegIndex(args.images_dir, args.coco_json)
        tr_idx_orig, va_idx = seg_index.split_by_images(val_ratio=args.val_ratio, seed=42)

        tr_idx = seg_duplicate_positive(seg_index, tr_idx_orig, args.seg_dup)
        print(f"[i] ovseg train images: {len(tr_idx_orig)} (pos-dup x{args.seg_dup} -> {len(tr_idx)}) | val images: {len(va_idx)}")

        ds_tr = SegDataset(seg_index, tr_idx, train=True,  size=args.seg_img_size, norm_mode="clip")
        ds_va = SegDataset(seg_index, va_idx, train=False, size=args.seg_img_size, norm_mode="clip")

        num_workers = 2
        dl_tr = DataLoader(ds_tr, batch_size=args.seg_batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
        dl_va = DataLoader(ds_va, batch_size=args.seg_batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

        ov = OvSegModel(model_name=args.ovseg_model, pretrained=args.ovseg_ckpt,
                        n_classes=seg_index.num_classes,
                        freeze_backbone=args.ovseg_freeze_backbone,
                        decoder_channels=args.ovseg_decoder_ch,
                        decoder_dropout=args.ovseg_decoder_dropout,
                        device=device).to(device)
        ov = maybe_compile(ov, args.compile, mode=args.compile_mode, backend=args.compile_backend)

        vis_total, vis_train = count_params(ov.visual)
        dec_total, dec_train = count_params(ov.decoder)
        full_total, full_train = vis_total + dec_total, vis_train + dec_train
        print(f"[i] OVSeg backbone (CLIP) params: total={format_int(vis_total)}, trainable={format_int(vis_train)}")
        print(f"[i] OVSeg decoder           params: total={format_int(dec_total)}, trainable={format_int(dec_train)}")
        print(f"[i] OVSeg total             params: total={format_int(full_total)}, trainable={format_int(full_train)}")

        params = []
        if any(p.requires_grad for p in ov.visual.parameters()):
            params.append({"params": [p for p in ov.visual.parameters() if p.requires_grad],
                           "lr": args.ovseg_lr_backbone})
        params.append({"params": [p for p in ov.decoder.parameters() if p.requires_grad],
                       "lr": args.ovseg_lr_decoder})
        optimizer = torch.optim.AdamW(params, weight_decay=1e-4)

        class_weights = compute_ce_class_weights(seg_index, tr_idx).to(device)
        print(f"[i] CE class weights: {class_weights.cpu().numpy().round(3)}")
        criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)

        ckpt_mgr = CheckpointManager(out_dir, topk=3, prefix="ovseg")

        class_names = [seg_index.idx_to_name[i] for i in sorted(seg_index.idx_to_name.keys())]
        planned_dumps = plan_dump_epochs(args.epochs, args.val_save_every, args.val_vis_max_dumps)
        print(f"[i] Planned val dumps at epochs: {planned_dumps} (each up to {min(args.val_vis_n,10)} imgs)")

        for epoch in range(1, args.epochs + 1):
            ov.train()
            losses = []
            pbar = tqdm(dl_tr, desc=f"OVSeg Epoch {epoch}/{args.epochs}", leave=True)
            for x, y, _ in pbar:
                x = x.to(device)
                y = y.to(device)
                out = ov(x)["out"]
                loss = criterion(out, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
                pbar.set_postfix(loss=f"{np.mean(losses):.4f}")
            pbar.close()
            train_loss = float(np.mean(losses)) if losses else 0.0

            dump_dir = None
            dump_max = 0
            if epoch in planned_dumps:
                dump_dir = os.path.join(out_dir, f"val_dump_ep_{epoch:03d}")
                dump_max = min(args.val_vis_n, 10)

            metrics = seg_evaluate(ov, dl_va, seg_index.num_classes + 1, device,
                                   dump_dir=dump_dir, dump_max=dump_max,
                                   class_names=class_names, denorm="clip")
            print(f"[{epoch}] ovseg loss={train_loss:.4f} | mIoU(excl_bg)={metrics['miou_excl_bg']:.3f} | "
                  f"F1_fg={metrics['f1_macro_excl_bg']:.3f} | pixel_acc={metrics['pixel_acc']:.3f}")

            # save UNWRAPPED state_dict to avoid _orig_mod.* keys
            ckpt_mgr.save(get_state_dict_unwrapped(ov), epoch=epoch, metrics=metrics, loss=train_loss)

        print(f"[OK] OVSeg training finished. Checkpoints in {os.path.join(out_dir,'checkpoints')}")
        return

    # ---------- OVSeg infer ----------
    if args.mode == "ovseg_infer":
        if not args.ckpt or not os.path.isfile(args.ckpt):
            raise FileNotFoundError("Provide ovseg checkpoint with --ckpt")
        seg_index = SegIndex(args.images_dir, args.coco_json)
        ov = OvSegModel(model_name=args.ovseg_model, pretrained=args.ovseg_ckpt,
                        n_classes=seg_index.num_classes,
                        freeze_backbone=False,
                        decoder_channels=args.ovseg_decoder_ch,
                        decoder_dropout=args.ovseg_decoder_dropout,
                        device=device).to(device)
        try:
            sd = torch.load(args.ckpt, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(args.ckpt, map_location=device)
        load_state_dict_flexible(ov, sd, strict=True)
        ov = maybe_compile(ov, args.compile, mode=args.compile_mode, backend=args.compile_backend)

        vis_total, vis_train = count_params(ov.visual)
        dec_total, dec_train = count_params(ov.decoder)
        full_total, full_train = vis_total + dec_total, vis_train + dec_train
        print(f"[i] OVSeg backbone (CLIP) params: total={format_int(vis_total)}, trainable={format_int(vis_train)}")
        print(f"[i] OVSeg decoder           params: total={format_int(dec_total)}, trainable={format_int(dec_train)}")
        print(f"[i] OVSeg total             params: total={format_int(full_total)}, trainable={format_int(full_train)}")

        ovseg_infer_save(ov, args, seg_index)
        print("[OK] OVSeg inference finished.")
        return

    raise ValueError("Unknown mode")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["train", "infer", "seg_train", "seg_infer", "ovseg_train", "ovseg_infer"], default="seg_train")

    ap.add_argument("--images_dir", default="/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/train")
    ap.add_argument("--coco_json",  default="/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/annotations/result_coco.json")
    ap.add_argument("--out_dir",    default="/home/sasha/Facade_segmentation/results")
    ap.add_argument("--test_dir",   default="/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/test")

    ap.add_argument("--ckpt", default='')  # путь к чекпоинтам для infer-режимов

    # CLIP tiles
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--stride",    type=int, default=256)
    ap.add_argument("--cover_thr", type=float, default=0.005)

    ap.add_argument("--img_size",   type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs",     type=int, default=10)
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

    # torchvision segmentation
    ap.add_argument("--seg_model",
                    choices=["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101"],
                    default="deeplabv3_resnet101")
    ap.add_argument("--seg_weights", choices=["default", "none"], default="default")
    ap.add_argument("--seg_img_size",  type=int, default=512)
    ap.add_argument("--seg_batch_size",type=int, default=6)
    ap.add_argument("--seg_lr",        type=float, default=2e-4)
    ap.add_argument("--seg_tile_size", type=int, default=768)
    ap.add_argument("--seg_stride",    type=int, default=512)

    # Валидационные дампы (расписание)
    ap.add_argument("--val_vis_n", type=int, default=10, help="сколько картинок сохранять на дамп (<=10)")
    ap.add_argument("--val_save_every", type=int, default=0, help="каждые N эпох сохранять; 0 — равномерно распределить не более 10 дампов")
    ap.add_argument("--val_vis_max_dumps", type=int, default=10, help="сколько максимум дампов эпох *всего* (<=10)")

    # Дублирование для seg/ovseg train
    ap.add_argument("--seg_dup", type=int, default=1, help="Во сколько раз продублировать train-примеры (в ovseg_train дублируются только positive)")

    # OVSeg (CLIP-ViT + decoder)
    ap.add_argument("--ovseg_model", default="ViT-B-16")
    ap.add_argument("--ovseg_ckpt",  default="openai")
    ap.add_argument("--ovseg_decoder_ch", type=int, default=256)
    ap.add_argument("--ovseg_decoder_dropout", type=float, default=0.0)
    ap.add_argument("--ovseg_freeze_backbone", action="store_true", help="Заморозить CLIP-бэкбон во время обучения")
    ap.add_argument("--ovseg_lr_backbone", type=float, default=1e-5)
    ap.add_argument("--ovseg_lr_decoder",  type=float, default=3e-4)

    # torch.compile
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile_mode", default="default", choices=["default", "reduce-overhead", "max-autotune"])
    ap.add_argument("--compile_backend", default="inductor")

    args = ap.parse_args()
    train(args)
