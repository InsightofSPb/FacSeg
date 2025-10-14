# main.py
import os, json, math, argparse, random, sys
import numpy as np
import cv2
from tqdm.auto import tqdm

# ensure project root on sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import *
from datasets.datasets import TileIndex, TilesDataset, SegIndex, SegDataset, _build_tf, load_coco_class_order
from model_zoo.models import CLIPHead, init_head_from_text, OvSegModel, get_seg_model

torch.set_float32_matmul_precision('high')


DEFAULT_IMAGES_DIR = "/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/images"
DEFAULT_COCO_JSON = "/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/annotations/result_coco.json"
DEFAULT_TEST_DIR = "/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/test"
DEFAULT_OUT_DIR = "/home/sasha/Facade_segmentation/results"
DEFAULT_PREPARED_DIR = "/home/sasha/Facade_segmentation/prepared"


def parse_args():
    ap = argparse.ArgumentParser("Facade defects (CLIP tiles + OVSeg + torchvision)")
    # common
    ap.add_argument("--mode", choices=["infer","clip_train","seg_train","seg_infer","ovseg_train","ovseg_infer","seg_prepare"], default="infer")
    ap.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--coco_json",  type=str, default=DEFAULT_COCO_JSON)
    ap.add_argument("--test_dir",   type=str, default=DEFAULT_TEST_DIR)
    ap.add_argument("--test_coco_json", type=str, default=DEFAULT_COCO_JSON)
    ap.add_argument("--out_dir",    type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile_mode", default="default")
    ap.add_argument("--compile_backend", default=None)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--class_aliases", type=str, default="")
    ap.add_argument("--vis_legend", action="store_true", help="Draw legend on saved visuals")
    ap.add_argument("--dump_aliases_template", type=str, default="", help="Path to write a sample aliases.json and exit")

    # clip head
    ap.add_argument("--clip_model", default="ViT-B-32-quickgelu")
    ap.add_argument("--clip_ckpt",  default="openai")
    ap.add_argument("--head_type",  default="mlp", choices=["mlp","linear"])
    ap.add_argument("--head_hidden", type=int, default=2048)
    ap.add_argument("--head_dropout", type=float, default=0.1)

    ap.add_argument("--tile_size", type=int, default=768)
    ap.add_argument("--stride",    type=int, default=512)
    ap.add_argument("--cover_thr", type=float, default=0.01)
    ap.add_argument("--keep_empty", action="store_true")
    ap.add_argument("--img_size",  type=int, default=336)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs",     type=int, default=12)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--pos_dup",    type=int, default=2)

    ap.add_argument("--vis_thr",    type=float, default=0.5)
    ap.add_argument("--vis_blur",   type=int, default=0)
    ap.add_argument("--vis_max_alpha", type=float, default=0.6)

    # torchvision seg
    ap.add_argument("--seg_model", default="deeplabv3_resnet50")
    ap.add_argument("--seg_img_size", type=int, default=512)
    ap.add_argument("--seg_batch_size", type=int, default=6)
    ap.add_argument("--seg_dup", type=int, default=1)
    ap.add_argument("--seg_weights", default="imagenet")

    # ovseg
    ap.add_argument("--ovseg_model", default="ViT-B-16")
    ap.add_argument("--ovseg_ckpt",  default="openai")
    ap.add_argument("--ovseg_freeze_backbone", action="store_true")
    ap.add_argument("--ovseg_lr_decoder", type=float, default=3e-4)
    ap.add_argument("--ovseg_lr_backbone", type=float, default=1e-5)
    ap.add_argument("--ovseg_decoder_ch", type=int, default=256)
    ap.add_argument("--ovseg_decoder_dropout", type=float, default=0.0)
    ap.add_argument("--train_miou_ratio", type=float, default=0.10,
                    help="Доля train для оценки mIoU/F1 после эпохи (0..1, 0=пропустить, 1=всё)")
    # checkpoints
    ap.add_argument("--ckpt", type=str, default="", help="Path to load checkpoint for infer / resume")
    ap.add_argument("--prep_out_dir", type=str, default=DEFAULT_PREPARED_DIR,
                    help="Куда выгружать подготовленные датасеты (seg_prepare)")
    ap.add_argument("--prep_norm_mode", choices=["clip", "imagenet"], default="imagenet",
                    help="Какой набор аугментаций использовать при подготовке (влияет на нормировку при трене)")
    ap.add_argument("--prep_seed", type=int, default=42, help="Сид для аугментаций в seg_prepare")

    args = ap.parse_args()

    for attr in ["images_dir", "coco_json", "test_dir", "test_coco_json", "out_dir", "prep_out_dir", "class_aliases", "ckpt"]:
        val = getattr(args, attr, "")
        if isinstance(val, str):
            setattr(args, attr, os.path.expanduser(val.strip()))

    if args.dump_aliases_template:
        template = {
            "CRACK": ["CRACKS", "FISSURE", "裂缝", "FRACTURE", "CRACK_LINE"],
            "SPALLING": ["SCALING", "CONCRETE_SPALL", "EXFOLIATION", "SURFACE_SPALL"],
            "DELAMINATION": ["DELAM", "PEELING_PLASTER", "LAYER_SEPARATION"],
            "MISSING_ELEMENT": ["LOSS", "MISSING_PIECE", "MISSING_PART", "CHIPPING", "BROKEN_PIECE"],
            "WATER_STAIN": ["DAMP_STAIN", "MOISTURE_STAIN", "WET_MARK", "WATERMARK"],
            "EFFLORESCENCE": ["SALT", "WHITE_SALT", "SALT_STAIN", "SALTPETER", "BLOOM"],
            "CORROSION": ["RUST", "RUST_STAIN", "OXIDATION", "CORRODED"],
            "ORNAMENT_INTACT": ["INTACT", "GOOD_ORNAMENT", "UNDAMAGED"],
            "REPAIRS": ["AFTER_REPAIR", "PATCH", "REPAIR_PATCH", "PLASTER_PATCH", "MORTAR_PATCH", "REPAIRED_AREA"],
            "TEXT_OR_IMAGES": ["GRAFFITI", "GRAFFITTI", "STICKER", "STICKERS", "POSTER", "FLYER", "DECAL", "AD", "PAINTED_TEXT", "NUMBERING", "TAG"]
        }
        with open(args.dump_aliases_template, "w", encoding="utf-8") as f:
            json.dump(template, f, ensure_ascii=False, indent=2)
        print(f"[OK] aliases template written to: {args.dump_aliases_template}")
        raise SystemExit

    return args


# ---------- helpers ----------
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


class TopKCheckpointManager:
    def __init__(self, directory: str, prefix: str, k: int = 5):
        self.directory = directory
        self.prefix = prefix
        self.k = k
        self._saved = []  # list of (loss, path)

    def _format_metrics_suffix(self, metrics: dict, loss: float) -> str:
        parts = [
            f"loss_{loss:.4f}",
        ]
        order = [
            ("pixel_acc", "pixelacc"),
            ("miou", "miou"),
            ("f1_macro", "f1"),
        ]
        for key, tag in order:
            if key in metrics and metrics[key] is not None:
                parts.append(f"{tag}_{metrics[key]:.4f}")
        return "_".join(parts)

    def save(self, state_dict: dict, epoch: int, loss: float, metrics: dict, extra=None):
        suffix = self._format_metrics_suffix(metrics, loss)
        fname = f"{self.prefix}_ep{epoch:03d}_{suffix}.pt"
        path = os.path.join(self.directory, fname)
        payload = {
            "state_dict": state_dict,
            "epoch": int(epoch),
            "val_loss": float(loss),
            "metrics": metrics,
        }
        if extra:
            for key, value in extra.items():
                if value is not None:
                    payload[key] = value
        torch.save(payload, path)
        self._saved.append((loss, path))
        self._saved.sort(key=lambda x: x[0])
        while len(self._saved) > self.k:
            _, drop_path = self._saved.pop(-1)
            try:
                os.remove(drop_path)
            except OSError:
                pass


def _infer_num_classes_from_state_dict(state_dict):
    min_out = None
    for key, tensor in state_dict.items():
        if not hasattr(tensor, "shape"):
            continue
        if getattr(tensor, "ndim", 0) == 0:
            continue
        if not key.endswith("weight"):
            continue
        bias_key = key[:-6] + "bias"
        if bias_key not in state_dict:
            continue
        out_ch = int(tensor.shape[0])
        if out_ch <= 1:
            continue
        if min_out is None or out_ch < min_out:
            min_out = out_ch
    if min_out is None:
        return None
    return max(0, min_out - 1)


def _default_class_order(num_classes):
    base = [DEFAULT_CATEGORIES[k] for k in sorted(DEFAULT_CATEGORIES.keys())]
    return base[:num_classes]


def evaluate_segmentation_full(model, dl, device, criterion):
    model.eval()
    losses = []
    hist = None
    K = None
    with torch.no_grad():
        for x, y, _ in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            out = model(x)["out"]
            if out.shape[-2:] != y.shape[-2:]:
                out = F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(out, y)
            losses.append(float(loss.item()))
            pred = out.argmax(1)
            if K is None:
                K = out.shape[1]
            h = seg_hist_np(pred.cpu().numpy(), y.cpu().numpy(), K)
            hist = h if hist is None else (hist + h)
    avg_loss = float(np.mean(losses)) if losses else float("inf")
    metrics = seg_metrics_from_hist(hist, ignore_background=True) if hist is not None else {
        "pixel_acc": 0.0,
        "miou": 0.0,
        "f1_macro": 0.0,
    }
    return avg_loss, metrics, K


def _mask_to_color(mask, class_names):
    color = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
    for idx, name in enumerate(class_names, start=1):
        color[mask == idx] = hex_to_bgr(LS_PALETTE.get(name, "#FF00FF"))
    return color


def _export_seg_split(seg_index: SegIndex, idxs, split_name: str, out_root: str,
                      size: int, norm_mode: str, alpha: float = 0.6):
    split_root = os.path.join(out_root, split_name)
    img_dir = os.path.join(split_root, "images")
    mask_dir = os.path.join(split_root, "masks")
    combo_dir = os.path.join(split_root, "combo")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(combo_dir, exist_ok=True)

    tf = _build_tf(train=(split_name == "train"), size=size, norm_mode=norm_mode, include_normalize=False)
    meta = []
    for i, idx in enumerate(tqdm(idxs, desc=f"Prepare {split_name}", leave=False)):
        rec = seg_index.items[idx]
        img_bgr = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tfed = tf(image=img_rgb, mask=rec["mask"])
        img_tf = tfed["image"]
        mask_tf = tfed["mask"].astype(np.uint8)

        base = os.path.splitext(os.path.basename(rec["path"]))[0]
        safe_base = base.replace(" ", "_")
        name = f"{i:05d}_{safe_base}"

        img_path = os.path.join(img_dir, name + ".png")
        mask_path = os.path.join(mask_dir, name + ".png")
        combo_path = os.path.join(combo_dir, name + ".png")

        cv2.imwrite(img_path, cv2.cvtColor(img_tf, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(mask_path, mask_tf, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        color_map = _mask_to_color(mask_tf, seg_index.classes)
        save_overlay(img_tf, color_map, combo_path, alpha=alpha, blur=0,
                     add_legend=True, class_names=seg_index.classes, palette=LS_PALETTE)

        meta.append({
            "source_path": rec["path"],
            "image": os.path.relpath(img_path, split_root),
            "mask": os.path.relpath(mask_path, split_root),
            "combo": os.path.relpath(combo_path, split_root),
        })

    meta_path = os.path.join(split_root, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": seg_index.classes,
            "items": meta,
        }, f, ensure_ascii=False, indent=2)


def seg_prepare(args):
    random.seed(args.prep_seed)
    np.random.seed(args.prep_seed)
    torch.manual_seed(args.prep_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.prep_seed)

    os.makedirs(args.prep_out_dir, exist_ok=True)
    print(f"[i] подготовленный датасет будет сохранён в: {args.prep_out_dir}")

    seg_index = SegIndex(args.images_dir, args.coco_json, class_aliases=args.class_aliases)
    tr_idx, va_idx = seg_index.split_by_images(val_ratio=args.val_ratio, seed=42)
    print(f"[i] подготовка train={len(tr_idx)}  val={len(va_idx)}")

    _export_seg_split(seg_index, tr_idx, "train", args.prep_out_dir, args.seg_img_size,
                      norm_mode=args.prep_norm_mode, alpha=args.vis_max_alpha)
    _export_seg_split(seg_index, va_idx, "val", args.prep_out_dir, args.seg_img_size,
                      norm_mode=args.prep_norm_mode, alpha=args.vis_max_alpha)

    if args.test_dir and os.path.isdir(args.test_dir):
        test_json = args.test_coco_json if args.test_coco_json else args.coco_json
        try:
            if os.path.abspath(args.test_dir) == os.path.abspath(args.images_dir) and \
               os.path.abspath(test_json) == os.path.abspath(args.coco_json):
                # test совпадает с train — не дублируем
                pass
            else:
                seg_index_test = SegIndex(args.test_dir, test_json, class_aliases=args.class_aliases)
                test_idxs = list(range(len(seg_index_test.items)))
                print(f"[i] подготовка test={len(test_idxs)}")
                _export_seg_split(seg_index_test, test_idxs, "test", args.prep_out_dir, args.seg_img_size,
                                  norm_mode=args.prep_norm_mode, alpha=args.vis_max_alpha)
        except Exception as exc:
            print(f"[!] не удалось подготовить тестовый набор: {exc}")

    print("[OK] подготовка датасетов завершена")


# ---------- train loops ----------
def clip_train(args, device):
    index = TileIndex(args.images_dir, args.coco_json,
                      tile_size=args.tile_size, stride=args.stride,
                      cover_thr=args.cover_thr, keep_empty=args.keep_empty,
                      class_aliases=args.class_aliases)
    tr_idx, va_idx = index.split_by_images(val_ratio=args.val_ratio, seed=42)
    tr_idx = duplicate_positive_indices(index, tr_idx, args.pos_dup)

    ds_tr = TilesDataset(index, tr_idx, augment=True,  img_size=args.img_size)
    ds_va = TilesDataset(index, va_idx, augment=False, img_size=args.img_size)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = CLIPHead(model_name=args.clip_model, pretrained=args.clip_ckpt,
                     n_classes=len(index.classes), device=device,
                     head_type=args.head_type, head_hidden=args.head_hidden, head_dropout=args.head_dropout).to(device)
    init_head_from_text(model.head, index.classes, model_name=args.clip_model, pretrained=args.clip_ckpt, device=device)

    pos_w = compute_pos_weight(dl_tr, len(index.classes)).to(device)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt   = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = stamp_out_dir(args.out_dir)
    best = -1.0
    for ep in range(1, args.epochs+1):
        model.train()
        losses = []
        for x, y, _ in tqdm(dl_tr, desc=f"CLIP Train {ep}/{args.epochs}", leave=False):
            x = x.to(device); y = y.to(device)
            logits, _ = model(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        _, macro, _, _ = evaluate_clip(model, dl_va, device, n_classes=len(index.classes), progress_desc=f"CLIP Val {ep}")
        score = macro["ap"]
        print(f"[i] ep{ep}: loss={np.mean(losses):.4f} | mAP={score:.4f}")
        if score > best:
            best = score
            os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
            torch.save({"state_dict": model.state_dict()}, os.path.join(out_dir, "checkpoints", "clip_head.pt"))
    print("[OK] CLIP tile head trained.")
    return


def _eval_segmentation(model, dl, device, K_hint=None):
    """Вычисляет confusion-matrix по всему даталоадеру и возвращает (hist, K)."""
    model.eval()
    hist = None
    K = K_hint
    with torch.no_grad():
        # если не знаем K — возьмём с первого батча
        for bi, (x, y, _) in enumerate(dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            out = model(x)["out"]
            if out.shape[-2:] != y.shape[-2:]:
                out = F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
            if K is None:
                K = out.shape[1]
            pred = out.argmax(1)
            h = seg_hist_np(pred.cpu().numpy(), y.cpu().numpy(), K)
            hist = h if hist is None else (hist + h)
    return hist, K


def seg_train(args, device):
    out_dir = stamp_out_dir(args.out_dir)
    print(f"[i] results will be saved to: {out_dir}")
    seg_index = SegIndex(args.images_dir, args.coco_json, class_aliases=args.class_aliases)
    tr_idx_orig, va_idx = seg_index.split_by_images(val_ratio=args.val_ratio, seed=42)
    tr_idx = []
    for _ in range(max(1, args.seg_dup)):
        tr_idx += tr_idx_orig

    print(f"[i] seg train images: {len(tr_idx_orig)} (dup x{args.seg_dup} -> {len(tr_idx)}) | val images: {len(va_idx)}")
    ds_tr = SegDataset(seg_index, tr_idx, train=True,  size=args.seg_img_size, norm_mode="imagenet")
    ds_va = SegDataset(seg_index, va_idx, train=False, size=args.seg_img_size, norm_mode="imagenet")
    dl_tr = DataLoader(ds_tr, batch_size=args.seg_batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.seg_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = get_seg_model(args.seg_model, seg_index.num_classes, weights_tag=args.seg_weights).to(device)
    model = maybe_compile(model, args.compile, mode=args.compile_mode, backend=args.compile_backend)

    total, trainable = count_params(model)
    print(f"[i] Seg model params: total={total:,.0f}  trainable={trainable:.0f}")

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_mgr = TopKCheckpointManager(ckpt_dir, prefix="seg", k=5)

    for ep in range(1, args.epochs+1):
        model.train(); losses = []; seen = 0
        for x, m, _ in tqdm(dl_tr, desc=f"Seg Train {ep}/{args.epochs}", leave=False):
            x = x.to(device); m = m.to(device).long()
            out = model(x)["out"]
            if out.shape[-2:] != m.shape[-2:]:
                out = F.interpolate(out, size=m.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(out, m)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item())); seen += x.size(0)
        avg_loss = (sum(losses) / max(1, len(losses)))
        print(f"[ep {ep:03d}] train_loss={avg_loss:.4f}")

        # validation metrics
        val_loss, m, K = evaluate_segmentation_full(model, dl_va, device, criterion)
        print(f"[ep {ep:03d}] val: loss={val_loss:.4f}  pixel_acc={m['pixel_acc']:.4f}  mIoU={m['miou']:.4f}  F1={m['f1_macro']:.4f}")

        # optional train metrics on a fraction
        frac = max(0.0, min(1.0, getattr(args, "train_miou_ratio", 0.0)))
        if frac > 0.0:
            max_batches = int(frac * len(dl_tr)) or 1
            model.eval(); hist_tr = None; bcount = 0
            with torch.no_grad():
                for x, y, _ in dl_tr:
                    x = x.to(device); y = y.to(device).long()
                    out = model(x)["out"]
                    if out.shape[-2:] != y.shape[-2:]:
                        out = F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    pred = out.argmax(1)
                    h = seg_hist_np(pred.cpu().numpy(), y.cpu().numpy(), K)
                    hist_tr = h if hist_tr is None else (hist_tr + h)
                    bcount += 1
                    if bcount >= max_batches:
                        break
            mt = seg_metrics_from_hist(hist_tr, ignore_background=True)
            tag = "all" if frac >= 1.0 else f"{int(frac*100)}%"
            print(f"[ep {ep:03d}] train[{tag}]: mIoU={mt['miou']:.4f}  F1={mt['f1_macro']:.4f}")

        ckpt_mgr.save(model.state_dict(), ep, val_loss, m, extra={"classes": seg_index.classes})


def ovseg_train(args, device):
    out_dir = stamp_out_dir(args.out_dir)
    print(f"[i] results will be saved to: {out_dir}")
    seg_index = SegIndex(args.images_dir, args.coco_json, class_aliases=args.class_aliases)

    tr_idx_orig, va_idx = seg_index.split_by_images(val_ratio=args.val_ratio, seed=42)
    tr_idx = []
    for _ in range(max(1, args.seg_dup)):
        tr_idx += tr_idx_orig
    print(f"[i] seg train images: {len(tr_idx_orig)} (dup x{args.seg_dup} -> {len(tr_idx)}) | val images: {len(va_idx)}")

    ds_tr = SegDataset(seg_index, tr_idx, train=True,  size=args.seg_img_size, norm_mode="clip")
    ds_va = SegDataset(seg_index, va_idx, train=False, size=args.seg_img_size, norm_mode="clip")
    dl_tr = DataLoader(ds_tr, batch_size=args.seg_batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=args.seg_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = OvSegModel(args.ovseg_model, args.ovseg_ckpt, seg_index.num_classes,
                       freeze_backbone=args.ovseg_freeze_backbone,
                       decoder_channels=args.ovseg_decoder_ch,
                       decoder_dropout=args.ovseg_decoder_dropout,
                       device=device).to(device)
    model = maybe_compile(model, args.compile, mode=args.compile_mode, backend=args.compile_backend)

    # param counters (чистые числа)
    try:
        n_backbone = sum(p.numel() for p in model.visual.parameters())
    except Exception:
        n_backbone = 0
    try:
        n_head = sum(p.numel() for p in model.decoder.parameters())
    except Exception:
        n_head = sum(p.numel() for p in model.parameters()) - n_backbone
    print(f"[i] OVSeg params: backbone={n_backbone}  head={n_head}  total={n_backbone + n_head}")

    dec_params = list(model.decoder.parameters())
    bb_params  = [p for p in model.visual.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": dec_params, "lr": args.ovseg_lr_decoder, "weight_decay": 1e-4},
            {"params": bb_params,  "lr": args.ovseg_lr_backbone, "weight_decay": 0.0},
        ]
    )
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_mgr = TopKCheckpointManager(ckpt_dir, prefix="ovseg", k=5)

    for ep in range(1, args.epochs+1):
        model.train(); losses = []; seen = 0
        for x, m, _ in tqdm(dl_tr, desc=f"OVSeg Train {ep}/{args.epochs}", leave=False):
            x = x.to(device); m = m.to(device).long()
            out = model(x)["out"]
            if out.shape[-2:] != m.shape[-2:]:
                out = F.interpolate(out, size=m.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(out, m)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item())); seen += x.size(0)
        avg_loss = (sum(losses) / max(1, len(losses)))
        print(f"[ep {ep:03d}] train_loss={avg_loss:.4f}")

        # validation metrics
        val_loss, m, K = evaluate_segmentation_full(model, dl_va, device, criterion)
        print(f"[ep {ep:03d}] val: loss={val_loss:.4f}  pixel_acc={m['pixel_acc']:.4f}  mIoU={m['miou']:.4f}  F1={m['f1_macro']:.4f}")

        # optional train metrics on a fraction
        frac = max(0.0, min(1.0, getattr(args, "train_miou_ratio", 0.0)))
        if frac > 0.0:
            max_batches = int(frac * len(dl_tr)) or 1
            model.eval(); hist_tr = None; bcount = 0
            with torch.no_grad():
                for x, y, _ in dl_tr:
                    x = x.to(device); y = y.to(device).long()
                    out = model(x)["out"]
                    if out.shape[-2:] != y.shape[-2:]:
                        out = F.interpolate(out, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    pred = out.argmax(1)
                    h = seg_hist_np(pred.cpu().numpy(), y.cpu().numpy(), K)
                    hist_tr = h if hist_tr is None else (hist_tr + h)
                    bcount += 1
                    if bcount >= max_batches:
                        break
            mt = seg_metrics_from_hist(hist_tr, ignore_background=True)
            tag = "all" if frac >= 1.0 else f"{int(frac*100)}%"
            print(f"[ep {ep:03d}] train[{tag}]: mIoU={mt['miou']:.4f}  F1={mt['f1_macro']:.4f}")

        ckpt_mgr.save(model.state_dict(), ep, val_loss, m, extra={"classes": seg_index.classes})


# ---------- inference ----------
@torch.no_grad()
def clip_infer_on_dir(args, device):
    assert args.ckpt, "--ckpt is required for infer"
    index = TileIndex(args.images_dir if args.images_dir else args.test_dir,
                      args.coco_json, tile_size=args.tile_size, stride=args.stride,
                      cover_thr=args.cover_thr, keep_empty=True, class_aliases=args.class_aliases)

    model = CLIPHead(model_name=args.clip_model, pretrained=args.clip_ckpt,
                     n_classes=len(index.classes), device=device,
                     head_type=args.head_type, head_hidden=args.head_hidden, head_dropout=args.head_dropout).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["state_dict"])
    model.eval()

    out_dir = args.out_dir or os.path.join(DEFAULT_OUT_DIR, "clip_infer")
    os.makedirs(out_dir, exist_ok=True)
    out_idx   = os.path.join(out_dir, "clip_idx");   os.makedirs(out_idx, exist_ok=True)
    out_color = os.path.join(out_dir, "clip_color"); os.makedirs(out_color, exist_ok=True)
    out_over  = os.path.join(out_dir, "clip_over");  os.makedirs(out_over, exist_ok=True)
    out_comp  = os.path.join(out_dir, "clip_comp");  os.makedirs(out_comp, exist_ok=True)

    class_names = index.classes
    kernel_full = make_cosine_kernel(args.tile_size)

    test_dir = args.test_dir if args.test_dir else args.images_dir
    img_paths = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                img_paths.append(os.path.join(root, f))
    img_paths.sort()

    for path in tqdm(img_paths, desc="CLIP infer"):
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        if src is None:
            print(f"[!] cannot read {path}")
            continue
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        H, W = src.shape[:2]
        acc = np.zeros((len(class_names), H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)

        batch_tiles, coords = [], []
        for yy in range(0, max(1, H - args.tile_size + 1), args.stride):
            for xx in range(0, max(1, W - args.tile_size + 1), args.stride):
                raw = src[yy:yy+args.tile_size, xx:xx+args.tile_size]   # может быть "урезанным"
                hh, ww = raw.shape[:2]
                # для модели всё равно приводим к img_size
                tile = cv2.resize(raw, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA)
                tile = (tile.astype(np.float32)/255.0 - np.array(CLIP_MEAN)[None,None,:]) / np.array(CLIP_STD)[None,None,:]
                tile = torch.from_numpy(tile.transpose(2,0,1)).float()
                batch_tiles.append(tile); coords.append((xx, yy, ww, hh))
                if len(batch_tiles) >= args.batch_size:
                    probs = infer_batch_clip(model, batch_tiles, device)
                    for p, (xx2, yy2, ww2, hh2) in zip(probs, coords):
                        k = kernel_full[:hh2, :ww2]
                        acc[:, yy2:yy2+hh2, xx2:xx2+ww2] += p[:, None, None] * k
                        cnt[yy2:yy2+hh2, xx2:xx2+ww2] += k
                    batch_tiles, coords = [], []

        if batch_tiles:
            probs = infer_batch_clip(model, batch_tiles, device)
            for p, (xx2, yy2, ww, hh) in zip(probs, coords):
                k = kernel_full[:hh, :ww]
                acc[:, yy2:yy2+hh, xx2:xx2+ww] += p[:, None, None] * k
                cnt[yy2:yy2+hh, xx2:xx2+ww] += k

        cnt[cnt == 0] = 1.0
        acc /= cnt[None, :, :]

        save_composite_overlay(
            src, acc, class_names,
            os.path.join(out_comp, os.path.basename(path)),
            palette=LS_PALETTE, vis_thr=args.vis_thr, alpha=args.vis_max_alpha, blur=args.vis_blur,
            add_legend=args.vis_legend
        )

        base = os.path.splitext(os.path.basename(path))[0]
        idx_path   = os.path.join(out_idx,   base + ".png")
        color_path = os.path.join(out_color, base + ".png")
        over_path  = os.path.join(out_over,  base + ".jpg")
        _, color_map = save_index_and_color_maps(acc, class_names, idx_path, color_path,
                                                 vis_thr=args.vis_thr, palette=LS_PALETTE, add_legend=args.vis_legend)
        save_overlay(src, color_map, over_path, alpha=args.vis_max_alpha, blur=args.vis_blur,
                     add_legend=args.vis_legend, class_names=class_names, palette=LS_PALETTE)


def seg_infer(args, device, use_ovseg=False):
    assert args.ckpt, "--ckpt is required for seg_infer/ovseg_infer"
    payload = torch.load(args.ckpt, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    class_names = None
    num_classes = None
    if isinstance(payload, dict):
        ckpt_classes = payload.get("classes")
        if ckpt_classes:
            class_names = list(ckpt_classes)
            num_classes = len(class_names)

    seg_index = None
    if class_names is None:
        try:
            seg_index = SegIndex(args.images_dir if args.images_dir else args.test_dir,
                                 args.coco_json, class_aliases=args.class_aliases)
            class_names = seg_index.classes
            num_classes = seg_index.num_classes
        except AssertionError as exc:
            print(f"[!] cannot build SegIndex from dataset: {exc}")
        except FileNotFoundError as exc:
            print(f"[!] cannot build SegIndex from dataset: {exc}")
        except Exception as exc:
            print(f"[!] cannot build SegIndex from dataset ({type(exc).__name__}): {exc}")

    if class_names is None:
        try:
            coco_classes = load_coco_class_order(args.coco_json, args.class_aliases)
            if coco_classes:
                class_names = coco_classes
                num_classes = len(class_names)
                print("[i] class order restored from COCO categories")
        except Exception as exc:
            print(f"[!] failed to load class order from COCO: {exc}")

    if num_classes is None:
        num_classes = _infer_num_classes_from_state_dict(state_dict)
        if num_classes:
            print(f"[i] inferred {num_classes} classes from checkpoint weights")

    if num_classes is None or num_classes <= 0:
        raise RuntimeError("Unable to determine number of segmentation classes for inference.")

    if class_names is None or len(class_names) != num_classes:
        class_names = _default_class_order(num_classes)
        print("[i] falling back to default class names order")

    if use_ovseg:
        model = OvSegModel(args.ovseg_model, args.ovseg_ckpt, num_classes,
                           freeze_backbone=False, decoder_channels=args.ovseg_decoder_ch,
                           decoder_dropout=args.ovseg_decoder_dropout, device=device).to(device)
    else:
        model = get_seg_model(args.seg_model, num_classes, weights_tag=None).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    out_dir = args.out_dir or os.path.join(DEFAULT_OUT_DIR, "seg_infer")
    os.makedirs(out_dir, exist_ok=True)
    out_idx   = os.path.join(out_dir, "seg_idx");   os.makedirs(out_idx, exist_ok=True)
    out_color = os.path.join(out_dir, "seg_color"); os.makedirs(out_color, exist_ok=True)
    out_over  = os.path.join(out_dir, "seg_over");  os.makedirs(out_over, exist_ok=True)

    test_dir = args.test_dir if args.test_dir else args.images_dir
    img_paths = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                img_paths.append(os.path.join(root, f))
    img_paths.sort()

    for path in tqdm(img_paths, desc="Seg infer"):
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        if src is None:
            print(f"[!] cannot read {path}")
            continue
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        H, W = src.shape[:2]

        long_side = max(H, W)
        scale = args.seg_img_size / long_side
        newH, newW = int(round(H*scale)), int(round(W*scale))
        img = cv2.resize(src, (newW, newH), interpolation=cv2.INTER_AREA)
        padH = args.seg_img_size - newH
        padW = args.seg_img_size - newW
        img = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_CONSTANT, value=(0,0,0))

        if use_ovseg:
            mean, std = np.array(CLIP_MEAN), np.array(CLIP_STD)
        else:
            mean, std = np.array(IMAGENET_MEAN), np.array(IMAGENET_STD)

        x = (img.astype(np.float32)/255.0 - mean[None,None,:]) / std[None,None,:]
        x = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            out = model(x)["out"]
            logits = F.interpolate(out, size=(newH, newW), mode="bilinear", align_corners=False)
            pred = logits.argmax(1)[0].detach().cpu().numpy()  # 0..C

        full = np.zeros((H, W), np.uint8)
        full[:newH, :newW] = pred

        base = os.path.splitext(os.path.basename(path))[0]
        idx_path   = os.path.join(out_idx,   base + ".png")
        color_path = os.path.join(out_color, base + ".png")
        over_path  = os.path.join(out_over,  base + ".jpg")

        C = num_classes
        acc = np.zeros((C, H, W), np.float32)
        for c in range(1, C+1):
            acc[c-1] = (full == c).astype(np.float32)
        _, color_map = save_index_and_color_maps(acc, class_names, idx_path, color_path,
                                                 vis_thr=0.5, palette=LS_PALETTE, add_legend=args.vis_legend)
        save_overlay(src, color_map, over_path, alpha=args.vis_max_alpha, blur=args.vis_blur,
                     add_legend=args.vis_legend, class_names=class_names, palette=LS_PALETTE)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "clip_train":
        clip_train(args, device); return
    if args.mode == "infer":
        clip_infer_on_dir(args, device); return
    if args.mode == "seg_train":
        seg_train(args, device); return
    if args.mode == "seg_infer":
        seg_infer(args, device, use_ovseg=False); return
    if args.mode == "ovseg_train":
        ovseg_train(args, device); return
    if args.mode == "ovseg_infer":
        seg_infer(args, device, use_ovseg=True); return
    if args.mode == "seg_prepare":
        seg_prepare(args); return


if __name__ == "__main__":
    main()
