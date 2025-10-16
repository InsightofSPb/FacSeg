import os, json
from datetime import datetime

import numpy as np
import cv2

import torch


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

def _decode_rle_counts(counts, size):
    """Decode an uncompressed RLE counts list into a mask."""
    h, w = map(int, size)
    total = h * w
    flat = np.zeros(total, dtype=np.uint8)
    idx = 0
    val = 0
    for c in counts:
        c = abs(int(c))
        end = min(idx + c, total)
        if val == 1 and end > idx:
            flat[idx:end] = 1
        idx = end
        if idx >= total:
            break
        val = 1 - val
    return flat.reshape((h, w), order="F")


def _decode_rle(segmentation, H, W):
    size = segmentation.get("size", [H, W])
    counts = segmentation.get("counts", [])
    if isinstance(counts, list):
        return _decode_rle_counts(counts, size)
    if isinstance(counts, str):
        # try to use pycocotools if available (supports compressed RLE)
        try:
            from pycocotools import mask as mask_utils  # type: ignore
        except ImportError:
            # fallback: decode compressed string per COCO spec
            counts_bytes = counts.encode("utf-8")
            nums = []
            value = 0
            shift = 0
            for ch in counts_bytes:
                ch -= 48
                value |= (ch & 0x1F) << shift
                shift += 5
                if ch & 0x20:
                    continue
                if ch & 0x10:
                    value = -value
                nums.append(value)
                value = 0
                shift = 0
            return _decode_rle_counts(nums, size)
        else:
            decoded = mask_utils.decode({"counts": counts, "size": size})
            return decoded.astype(np.uint8)
    return np.zeros((size[0], size[1]), np.uint8)

def polygons_to_mask(segmentation, H, W, value=1):
    mask = np.zeros((H, W), np.uint8)
    if segmentation is None:
        return mask
    if isinstance(segmentation, list):
        if not segmentation:
            return mask
        if isinstance(segmentation[0], (list, tuple)):
            for poly in segmentation:
                pts = np.array(poly, np.float32).reshape(-1, 2)
                pts = np.round(pts).astype(np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], value)
        elif isinstance(segmentation[0], dict):
            for seg in segmentation:
                mask = np.maximum(mask, _decode_rle(seg, H, W) * value)
        else:
            pts = np.array(segmentation, np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], value)
    elif isinstance(segmentation, dict):
        mask = np.maximum(mask, _decode_rle(segmentation, H, W) * value)
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


def save_index_and_color_maps(acc_C_hw, class_names, idx_path, color_path,
                              vis_thr=0.5, palette=LS_PALETTE, add_legend=True):
    if isinstance(acc_C_hw, torch.Tensor):
        acc = acc_C_hw.detach().cpu().numpy()
    else:
        acc = np.asarray(acc_C_hw)

    if acc.ndim != 3:
        raise ValueError("acc_C_hw must have shape (C, H, W)")

    C, H, W = acc.shape
    idx = acc.argmax(0).astype(np.uint8)
    scores = acc.max(0)

    has_background = (C == len(class_names) + 1)

    if vis_thr is not None:
        if has_background:
            idx[scores < vis_thr] = 0
        scores_mask = (scores >= vis_thr)
    else:
        scores_mask = None

    color = np.zeros((H, W, 3), np.uint8)
    start_channel = 1 if has_background else 0
    max_classes = min(len(class_names), C - start_channel)

    for offset, name in enumerate(class_names[:max_classes], start=start_channel):
        mask = (idx == offset)
        if scores_mask is not None:
            mask = np.logical_and(mask, scores_mask)
        if not np.any(mask):
            continue
        color[mask] = hex_to_bgr(palette.get(name, "#FF00FF"))

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
    if base.ndim == 3 and base.shape[2] == 3:
        base = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    over = (alpha * base.astype(np.float32) + (1 - alpha) * src_rgb.astype(np.float32)).astype(np.uint8)
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


def seg_hist_np(pred, target, num_classes: int, ignore_index: int = 255):
    """
    Быстрый confusion-matrix для сегментации.
    pred, target: одинаковой формы (H,W) или (N,) с int-метками [0..K-1]
    Возвращает матрицу KxK, где [gt, pred].
    Совместимо с NumPy<2.0 (без dtype в bincount).
    """
    import numpy as np

    p = np.asarray(pred).reshape(-1).astype(np.int64)
    t = np.asarray(target).reshape(-1).astype(np.int64)

    # валидные пиксели: в диапазоне классов и не равны ignore_index
    mask = (t >= 0) & (t < num_classes)
    if ignore_index is not None:
        mask &= (t != ignore_index)

    # сворачиваем пары (gt, pred) в одно число и считаем частоты
    comb = num_classes * t[mask] + p[mask]
    hist = np.bincount(comb, minlength=num_classes * num_classes)  # без dtype
    hist = hist.reshape(num_classes, num_classes).astype(np.int64, copy=False)
    return hist

def seg_metrics_from_hist(hist: np.ndarray, ignore_background: bool = True):
    """Compute segmentation metrics from a confusion matrix.

    Args:
        hist: Confusion matrix of shape (K, K) where rows are GT classes and
            columns are predictions.
        ignore_background: If ``True`` the background class (index 0) is
            excluded from IoU/F1 averaging, but its interactions still count as
            false positives/negatives for the foreground classes.
    """
    eps = 1e-7
    hist = np.asarray(hist, dtype=np.float64)
    K = hist.shape[0]

    total = float(hist.sum())
    acc = float(hist.trace()) / float(total + eps)

    if ignore_background:
        if K <= 1:
            iou_per_class = np.array([], dtype=np.float64)
            f1_per_class = np.array([], dtype=np.float64)
        else:
            tp = np.diag(hist)[1:]
            # Предсказания класса c (включая фон как ложные срабатывания)
            fp = hist[:, 1:].sum(0) - tp
            # Пиксели класса c в разметке, предсказанные чем-то ещё (включая фон)
            fn = hist[1:, :].sum(1) - tp

            denom_iou = tp + fp + fn + eps
            iou_per_class = np.divide(tp, denom_iou, out=np.zeros_like(tp), where=denom_iou > 0)

            denom_f1 = 2.0 * tp + fp + fn + eps
            f1_per_class = np.divide(2.0 * tp, denom_f1, out=np.zeros_like(tp), where=denom_f1 > 0)
    else:
        tp = np.diag(hist)
        fp = hist.sum(0) - tp
        fn = hist.sum(1) - tp
        denom_iou = tp + fp + fn + eps
        iou_per_class = np.divide(tp, denom_iou, out=np.zeros_like(tp), where=denom_iou > 0)

        denom_f1 = 2.0 * tp + fp + fn + eps
        f1_per_class = np.divide(2.0 * tp, denom_f1, out=np.zeros_like(tp), where=denom_f1 > 0)

    miou = float(np.nanmean(iou_per_class)) if iou_per_class.size else 0.0
    f1_macro = float(np.nanmean(f1_per_class)) if f1_per_class.size else 0.0

    return {
        "pixel_acc": float(acc),
        "iou_per_class": iou_per_class,
        "miou": float(miou),
        "f1_per_class": f1_per_class,
        "f1_macro": float(f1_macro),
    }