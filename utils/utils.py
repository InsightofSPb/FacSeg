import os, json, math, hashlib
from datetime import datetime
from collections import defaultdict

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

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
    9: "REPAIRS",
    10: "TEXT_OR_IMAGES",
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
    "REPAIRS": "#4E9E9E",
    "TEXT_OR_IMAGES": "#8E7E47",
}

# ---------- name utils / aliases ----------
def _norm_name(s: str) -> str:
    return s.strip().upper().replace(" ", "_")

def load_aliases_json(path: str):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    table = {}
    for canon, alist in raw.items():
        canon_n = _norm_name(canon)
        table[canon_n] = canon_n
        for a in alist:
            table[_norm_name(a)] = canon_n
    return table

def remap_id_to_canonical(id_to_name: dict, aliases: dict, fallback=DEFAULT_CATEGORIES):
    # id_to_name: {coco_id: raw_name}
    mapped = {cid: aliases.get(_norm_name(n), _norm_name(n)) for cid, n in id_to_name.items()}
    canon_order = [fallback[k] for k in sorted(fallback.keys())]
    present = [n for n in canon_order if n in set(mapped.values())]
    return mapped, present

# ---------- misc ----------
def hex_to_bgr(hex_str: str):
    h = hex_str.lstrip("#")
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return (b, g, r)

def safe_path(images_dir, file_name):
    p = file_name
    if not os.path.isabs(p):
        p = os.path.join(images_dir, file_name)
    return p if os.path.isfile(p) else None

def polygons_to_mask(segmentation, H, W, value=1):
    mask = np.zeros((H, W), np.uint8)
    if isinstance(segmentation, list):
        for poly in segmentation:
            pts = np.array(poly, np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], value)
    return mask

def format_int(n: int) -> str:
    s = str(n); out = []
    while s:
        out.append(s[-3:]); s = s[:-3]
    return " ".join(reversed(out))

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def maybe_compile(model, do_compile: bool, mode="default", backend=None):
    if not do_compile:
        return model
    if hasattr(torch, "compile"):
        return torch.compile(model, mode=mode, backend=backend)
    return model

def stamp_out_dir(base):
    os.makedirs(base, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, tag)
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(out, "vis"), exist_ok=True)
    return out

# ---------- CLIP helpers ----------
@torch.no_grad()
def infer_batch_clip(model, tiles_batch, device):
    if isinstance(tiles_batch, list):
        x = torch.stack(tiles_batch, dim=0).to(device, non_blocking=True)
    else:
        x = tiles_batch.to(device, non_blocking=True)
    logits, _ = model(x)
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    return probs

def compute_pos_weight(dl_tr, n_classes):
    pos = torch.zeros(n_classes, dtype=torch.float32)
    cnt = 0
    for _, y, _ in dl_tr:
        pos += y.sum(dim=0); cnt += y.shape[0]
    pos = torch.clamp(pos, min=1.0)
    neg = torch.clamp(cnt - pos, min=1.0)
    return neg / pos

def evaluate_clip(model, dl, device, n_classes, progress_desc="CLIP Val"):
    model.eval()
    ys, ps = [], []
    for x, y, _ in dl:
        x = x.to(device); y = y.to(device)
        with torch.no_grad():
            logits, _ = model(x)
            prob = torch.sigmoid(logits)
        ys.append(y.detach().cpu().numpy())
        ps.append(prob.detach().cpu().numpy())
    Y = np.concatenate(ys, 0); P = np.concatenate(ps, 0)
    ap, auc = [], []
    for c in range(n_classes):
        try: ap.append(float(average_precision_score(Y[:, c], P[:, c])))
        except Exception: ap.append(float("nan"))
        try: auc.append(float(roc_auc_score(Y[:, c], P[:, c])))
        except Exception: auc.append(float("nan"))
    macro = {"ap": np.nanmean(ap), "auc": np.nanmean(auc)}
    return P, macro, ap, auc

# ---------- kernels & viz ----------
def make_cosine_kernel(ts: int):
    wx = 0.5 * (1 - np.cos(2 * np.pi * (np.arange(ts) + 0.5) / ts))
    wy = wx.copy()
    k = np.outer(wy, wx).astype(np.float32)
    k /= k.max()
    return k

def save_index_and_color_maps(acc_C_hw, class_names, idx_path, color_path, vis_thr=0.5, palette=LS_PALETTE, add_legend=True):
    C, H, W = acc_C_hw.shape
    idx = acc_C_hw.argmax(0).astype(np.uint8) + 1  # 1..C
    color = np.zeros((H, W, 3), np.uint8)
    for c, name in enumerate(class_names, start=1):
        color[idx == c] = hex_to_bgr(palette.get(name, "#FF00FF"))
    cv2.imwrite(idx_path, idx)
    cv2.imwrite(color_path, color)
    return idx, color

def _draw_legend(canvas, class_names, palette, alpha_bg=0.6):
    h, w = canvas.shape[:2]
    pad, sw, sh = 8, 22, 18
    x, y = pad, pad
    overlay = canvas.copy()
    for name in class_names:
        color = hex_to_bgr(palette.get(name, "#FF00FF"))
        cv2.rectangle(overlay, (x, y), (x+sw, y+sh), color, -1)
        cv2.putText(overlay, name, (x+sw+6, y+sh-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += sh + 6
    cv2.addWeighted(overlay, alpha_bg, canvas, 1-alpha_bg, 0, dst=canvas)

def save_overlay(src_rgb, color_map, out_path, alpha=0.6, blur=0, add_legend=False, class_names=None, palette=LS_PALETTE):
    base = color_map.copy()
    if blur and blur > 0:
        k = max(1, int(blur)) | 1
        base = cv2.GaussianBlur(base, (k, k), 0)
    over = (alpha * base + (1 - alpha) * src_rgb).astype(np.uint8)
    if add_legend and class_names:
        _draw_legend(over, class_names, palette)
    cv2.imwrite(out_path, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))

def save_composite_overlay(src_rgb, acc_C_hw, class_names, out_path,
                           palette=LS_PALETTE, vis_thr=0.5, alpha=0.6, blur=0,
                           add_legend=False):
    C, H, W = acc_C_hw.shape
    idx = acc_C_hw.argmax(0).astype(np.uint8)
    conf = acc_C_hw.max(0)
    color = np.zeros((H, W, 3), np.uint8)
    for c, name in enumerate(class_names):
        m = (idx == c) & (conf >= vis_thr)
        color[m] = hex_to_bgr(palette.get(name, "#FF00FF"))
    save_overlay(src_rgb, color, out_path, alpha=alpha, blur=blur,
                 add_legend=add_legend, class_names=class_names, palette=palette)
