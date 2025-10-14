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
from datasets.datasets import TileIndex, TilesDataset, SegIndex, SegDataset
from model_zoo.models import CLIPHead, init_head_from_text, OvSegModel, get_seg_model

torch.set_float32_matmul_precision('high')


def parse_args():
    ap = argparse.ArgumentParser("Facade defects (CLIP tiles + OVSeg + torchvision)")
    # common
    ap.add_argument("--mode", choices=["infer","clip_train","seg_train","seg_infer","ovseg_train","ovseg_infer"], default="infer")
    ap.add_argument("--images_dir", type=str, default="")
    ap.add_argument("--coco_json",  type=str, default="")
    ap.add_argument("--test_dir",   type=str, default="")
    ap.add_argument("--out_dir",    type=str, default="results")
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

    args = ap.parse_args()

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

    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

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
        hist_val, K = _eval_segmentation(model, dl_va, device)
        m = seg_metrics_from_hist(hist_val, ignore_background=True)
        print(f"[ep {ep:03d}] val: pixel_acc={m['pixel_acc']:.4f}  mIoU={m['miou']:.4f}  F1={m['f1_macro']:.4f}")

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

        torch.save({"state_dict": model.state_dict()}, os.path.join(out_dir, "checkpoints", f"seg_ep{ep:03d}.pt"))


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

    os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)

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
        hist_val, K = _eval_segmentation(model, dl_va, device)
        m = seg_metrics_from_hist(hist_val, ignore_background=True)
        print(f"[ep {ep:03d}] val: pixel_acc={m['pixel_acc']:.4f}  mIoU={m['miou']:.4f}  F1={m['f1_macro']:.4f}")

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

        torch.save({"state_dict": model.state_dict()}, os.path.join(out_dir, "checkpoints", f"ovseg_ep{ep:03d}.pt"))


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

    os.makedirs(args.out_dir, exist_ok=True)
    out_idx   = os.path.join(args.out_dir, "clip_idx");   os.makedirs(out_idx, exist_ok=True)
    out_color = os.path.join(args.out_dir, "clip_color"); os.makedirs(out_color, exist_ok=True)
    out_over  = os.path.join(args.out_dir, "clip_over");  os.makedirs(out_over, exist_ok=True)
    out_comp  = os.path.join(args.out_dir, "clip_comp");  os.makedirs(out_comp, exist_ok=True)

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
    seg_index = SegIndex(args.images_dir if args.images_dir else args.test_dir,
                         args.coco_json, class_aliases=args.class_aliases)

    if use_ovseg:
        model = OvSegModel(args.ovseg_model, args.ovseg_ckpt, seg_index.num_classes,
                           freeze_backbone=False, decoder_channels=args.ovseg_decoder_ch,
                           decoder_dropout=args.ovseg_decoder_dropout, device=device).to(device)
    else:
        model = get_seg_model(args.seg_model, seg_index.num_classes, weights_tag=None).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd["state_dict"])
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    out_idx   = os.path.join(args.out_dir, "seg_idx");   os.makedirs(out_idx, exist_ok=True)
    out_color = os.path.join(args.out_dir, "seg_color"); os.makedirs(out_color, exist_ok=True)
    out_over  = os.path.join(args.out_dir, "seg_over");  os.makedirs(out_over, exist_ok=True)

    # имена классов в порядке каналов (1..C — классы, 0 — фон)
    class_names = seg_index.classes

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

        C = seg_index.num_classes
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


if __name__ == "__main__":
    main()
