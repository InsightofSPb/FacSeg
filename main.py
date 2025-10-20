import os, json, argparse, random, sys, re, math
from collections import Counter
import numpy as np
import cv2
from tqdm.auto import tqdm
from typing import Dict, Optional
# ensure project root on sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import *
from datasets.datasets import *
from model_zoo.models import *

torch.set_float32_matmul_precision('high')


DEFAULT_IMAGES_DIR = "/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/train"
DEFAULT_COCO_JSON = "/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/annotations/result_coco.json"
DEFAULT_TEST_DIR = "/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/test"
DEFAULT_OUT_DIR = "/home/sasha/Facade_segmentation/results"
DEFAULT_PREPARED_DIR = "/home/sasha/Facade_segmentation/prepared"


_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _count_image_files(directory: str, extensions=_IMAGE_EXTS) -> int:
    total = 0
    low_exts = tuple(ext.lower() for ext in extensions)
    for root, _, files in os.walk(directory):
        for name in files:
            if name.lower().endswith(low_exts):
                total += 1
    return total


def parse_args():
    ap = argparse.ArgumentParser("Facade defects (OVSeg)")
    ap.add_argument("--mode", choices=["ovseg_train", "ovseg_infer", "seg_prepare"], default="ovseg_train")
    ap.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--coco_json",  type=str, default=DEFAULT_COCO_JSON)
    ap.add_argument("--test_dir",   type=str, default=DEFAULT_TEST_DIR)
    ap.add_argument("--test_coco_json", type=str, default=DEFAULT_COCO_JSON)
    ap.add_argument("--out_dir",    type=str, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--tiles-train",
        action="append",
        default=[],
        metavar="PATH",
        help="Path to a tiled dataset root (images/+masks/) for training."
             " Can be provided multiple times.",
    )
    ap.add_argument(
        "--tiles-val",
        action="append",
        default=[],
        metavar="PATH",
        help="Optional tiled dataset roots to use for validation."
             " When omitted the training tiles are split via --val_ratio.",
    )
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--compile_mode", default="default")
    ap.add_argument("--compile_backend", default=None)
    ap.add_argument("--val_ratio", type=float, default=0.25)
    ap.add_argument("--class_aliases", type=str, default="")
    ap.add_argument("--vis_legend", action="store_true", help="Draw legend on saved visuals")

    ap.add_argument("--vis_blur",   type=int, default=0)
    ap.add_argument("--vis_max_alpha", type=float, default=0.6)
    ap.add_argument("--dump_aliases_template", type=str, default="", help="Path to write a sample aliases.json and exit")
    ap.add_argument("--aug_dump_dir", type=str, default="",
                    help="Каталог для сохранения примеров аугментаций (train)")
    ap.add_argument("--aug_dump_limit", type=int, default=0,
                    help="Сколько аугментированных семплов сохранить (0=выкл)")
    ap.add_argument("--aug_config", type=str, default="",
                    help="Путь к JSON с переопределениями аугментаций (prepare/train)")
    # ovseg
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--ovseg_img_size", type=int, default=512)
    ap.add_argument("--ovseg_batch_size", type=int, default=2)
    ap.add_argument("--ovseg_dup", type=int, default=20)
    ap.add_argument("--ovseg_model", default="ViT-B-16")
    ap.add_argument("--ovseg_ckpt",  default="openai")
    ap.add_argument("--ovseg_freeze_backbone", action="store_true")
    ap.add_argument("--ovseg_lr_decoder", type=float, default=3e-4)
    ap.add_argument("--ovseg_lr_backbone", type=float, default=1e-5)
    ap.add_argument("--ovseg_decoder_ch", type=int, default=256)
    ap.add_argument("--ovseg_decoder_dropout", type=float, default=0.1)
    ap.add_argument("--train_miou_ratio", type=float, default=0.25,
                    help="Доля train для оценки mIoU/F1 после эпохи (0..1, 0=пропустить, 1=всё)")
    ap.add_argument("--val_checks_per_epoch", type=int, default=1,
                    help="Сколько раз запускать валидацию на эпоху (>=1)")
    # checkpoints
    ap.add_argument("--ckpt", type=str, default="", help="Path to load checkpoint for infer / resume")
    ap.add_argument("--prep_out_dir", type=str, default=DEFAULT_PREPARED_DIR,
                    help="Куда выгружать подготовленные датасеты (seg_prepare)")
    ap.add_argument("--prep_norm_mode", choices=["clip", "imagenet"], default="imagenet",
                    help="Какой набор аугментаций использовать при подготовке (влияет на нормировку при трене)")
    ap.add_argument("--prep_seed", type=int, default=42, help="Сид для аугментаций в seg_prepare")

    args = ap.parse_args()

    for attr in ["images_dir", "coco_json", "test_dir", "test_coco_json", "out_dir", "prep_out_dir", "class_aliases", "ckpt", "aug_dump_dir", "aug_config"]:
        val = getattr(args, attr, "")
        if isinstance(val, str):
            setattr(args, attr, os.path.expanduser(val.strip()))

    args.tiles_train = [os.path.expanduser(p.strip()) for p in (args.tiles_train or [])]
    args.tiles_val = [os.path.expanduser(p.strip()) for p in (args.tiles_val or [])]

    args.aug_config_data = {}
    if args.aug_config:
        with open(args.aug_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError("aug_config JSON должен содержать объект верхнего уровня")
        args.aug_config_data = cfg

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



class TopKCheckpointManager:
    def __init__(self, directory: str, prefix: str, k: int = 5,
                 mode: str = "min", saved: Optional[list] = None):
        self.directory = directory
        self.prefix = prefix
        self.k = k
        mode = mode.lower().strip()
        if mode not in {"min", "max"}:
            raise ValueError(f"Unsupported mode '{mode}', expected 'min' or 'max'")
        self.mode = mode
        self._saved = list(saved) if saved is not None else []  # list of (score, path)
        self._saved.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        if len(self._saved) > self.k:
            self._saved = self._saved[:self.k]

    def _format_metrics_suffix(self, metrics: dict, score: float) -> str:
        parts = [
            f"loss_{score:.4f}",
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

    def save(self, state_dict: dict, epoch: int, score: float, metrics: dict, extra=None):
        suffix = self._format_metrics_suffix(metrics, score)
        fname = f"{self.prefix}_ep{epoch:03d}_{suffix}.pt"
        path = os.path.join(self.directory, fname)
        payload = {
            "state_dict": state_dict,
            "epoch": int(epoch),
            "val_loss": float(score),
            "metrics": metrics,
        }
        if extra:
            for key, value in extra.items():
                if value is not None:
                    payload[key] = value
        torch.save(payload, path)
        self._saved.append((score, path))
        reverse = (self.mode == "max")
        self._saved.sort(key=lambda x: x[0], reverse=reverse)
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

def _generate_crop_sizes(split_name: str, count: int, size_min: int, size_max: int, rng: random.Random):
    count = max(0, int(count))
    if count == 0:
        return []
    size_min = max(32, int(size_min))
    size_max = max(size_min, int(size_max))
    split_name = (split_name or "train").lower()
    if split_name == "train":
        return [int(rng.randint(size_min, size_max)) for _ in range(count)]
    if count == 1:
        return [int(round((size_min + size_max) / 2.0))]
    vals = np.linspace(size_min, size_max, count)
    return [int(round(v)) for v in vals]


def _sample_crop_box(split_name: str, H: int, W: int, side: int, rng: random.Random):
    side = int(max(1, min(side, H, W)))
    split_name = (split_name or "train").lower()
    if split_name == "train":
        y0 = 0 if H == side else rng.randint(0, H - side)
        x0 = 0 if W == side else rng.randint(0, W - side)
    else:  # center crop for val/test
        y0 = max(0, (H - side) // 2)
        x0 = max(0, (W - side) // 2)
    return int(x0), int(y0), int(side), int(side)


def _resize_image_and_mask(image: np.ndarray, mask: np.ndarray, size: int):
    interp_img = cv2.INTER_AREA if image.shape[0] > size or image.shape[1] > size else cv2.INTER_LINEAR
    resized_image = cv2.resize(image, (size, size), interpolation=interp_img)
    resized_mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return resized_image, resized_mask
def _export_seg_split(seg_index: SegIndex, idxs, split_name: str, out_root: str,
                      size: int, norm_mode: str, alpha: float = 0.6, crop_conf=None,
                      rng: Optional[random.Random] = None, prepare_aug_config=None):
    split_root = os.path.join(out_root, split_name)
    img_dir = os.path.join(split_root, "images")
    mask_dir = os.path.join(split_root, "masks")
    combo_dir = os.path.join(split_root, "combo")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(combo_dir, exist_ok=True)

    rng = rng or random.Random()
    crop_conf = crop_conf or {}
    crop_count = max(0, int(crop_conf.get("count", 1)))
    crop_min = int(crop_conf.get("min_size", size))
    crop_max = int(crop_conf.get("max_size", size))
    tf = build_prepare_tf(split_name, norm_mode, include_normalize=False,
                          aug_config=prepare_aug_config or {})
    tf_pipeline = [type(t).__name__ for t in getattr(tf, "transforms", [])]
    meta = []
    duplicate_groups = {}
    mismatched = []
    sample_counter = 0
    for idx in tqdm(idxs, desc=f"Prepare {split_name}", leave=False):
        rec = seg_index.items[idx]
        img_bgr = cv2.imread(rec["path"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = rec["mask"].astype(np.uint8)

        base = os.path.splitext(os.path.basename(rec["path"]))[0]
        safe_base = base.replace(" ", "_")
        group_key = f"{safe_base}__{idx:05d}"

        sizes = _generate_crop_sizes(split_name, crop_count, crop_min, crop_max, rng)
        duplicate_groups[group_key] = {
            "source_path": rec["path"],
            "items": [],
            "expected_count": int(len(sizes)),
        }

        H, W = mask.shape[:2]
        generated_here = 0
        for crop_idx, desired_side in enumerate(sizes):
            x0, y0, crop_w, crop_h = _sample_crop_box(split_name, H, W, desired_side, rng)
            x1 = min(W, x0 + crop_w)
            y1 = min(H, y0 + crop_h)
            crop_img = img_rgb[y0:y1, x0:x1]
            crop_mask = mask[y0:y1, x0:x1]
            if crop_img.size == 0 or crop_mask.size == 0:
                continue

            resized_img, resized_mask = _resize_image_and_mask(crop_img, crop_mask, size)
            tfed = tf(image=resized_img, mask=resized_mask)
            img_tf = tfed["image"]
            mask_tf = tfed["mask"].astype(np.uint8)
            if img_tf.dtype != np.uint8:
                img_tf = np.clip(img_tf, 0, 255).astype(np.uint8)

            name = f"{sample_counter:07d}_{safe_base}_c{crop_idx:02d}_s{crop_w}"
            img_path = os.path.join(img_dir, name + ".png")
            mask_path = os.path.join(mask_dir, name + ".png")
            combo_path = os.path.join(combo_dir, name + ".png")

            cv2.imwrite(img_path, cv2.cvtColor(img_tf, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 3])
            cv2.imwrite(mask_path, mask_tf, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            color_map = _mask_to_color(mask_tf, seg_index.classes)
            save_overlay(img_tf, color_map, combo_path, alpha=alpha, blur=0,
                         add_legend=True, class_names=seg_index.classes, palette=LS_PALETTE)

            rel_img = os.path.relpath(img_path, split_root)
            rel_mask = os.path.relpath(mask_path, split_root)
            rel_combo = os.path.relpath(combo_path, split_root)
            meta.append({
                "source_path": rec["path"],
                "group_key": group_key,
                "source_index": int(idx),
                "crop_index": int(crop_idx),
                "crop_bbox": [int(x0), int(y0), int(x1), int(y1)],
                "crop_size": [int(crop_h), int(crop_w)],
                "image": rel_img,
                "mask": rel_mask,
                "combo": rel_combo,
            })
            duplicate_groups[group_key]["items"].append({
                "crop_index": int(crop_idx),
                "image": rel_img,
                "mask": rel_mask,
                "combo": rel_combo,
            })
            sample_counter += 1
            generated_here += 1

        duplicate_groups[group_key]["actual_count"] = int(generated_here)
        if generated_here != len(sizes):
            mismatched.append((rec["path"], len(sizes), generated_here))

    actual_total = len(meta)
    expected_total = sum(info["expected_count"] for info in duplicate_groups.values())
    if actual_total != expected_total:
        raise RuntimeError(
            f"[!] Split {split_name}: expected {expected_total} crops but generated {actual_total}."
        )
    if mismatched:
        details = ", ".join([f"{os.path.basename(p)} (exp={exp}, got={got})" for p, exp, got in mismatched])
        raise RuntimeError(f"[!] Split {split_name}: crop mismatch for: {details}")

    def _count_png(path):
        return len([f for f in os.listdir(path) if f.lower().endswith(".png")])

    imgs_saved = _count_png(img_dir)
    masks_saved = _count_png(mask_dir)
    combos_saved = _count_png(combo_dir)
    if not (imgs_saved == masks_saved == combos_saved == actual_total):
        raise RuntimeError(
            f"[!] Split {split_name}: file count mismatch (img={imgs_saved}, mask={masks_saved}, combo={combos_saved}, meta={actual_total})"
        )

    groups_serializable = []
    for key, info in duplicate_groups.items():
        items_sorted = sorted(info["items"], key=lambda x: x["crop_index"])
        groups_serializable.append({
            "group_key": key,
            "source_path": info["source_path"],
            "items": items_sorted,
            "expected_count": int(info["expected_count"]),
            "actual_count": int(info.get("actual_count", len(info["items"]))),
        })
    meta_path = os.path.join(split_root, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "classes": seg_index.classes,
            "items": meta,
            "duplicates": groups_serializable,
            "config": {
                "split": split_name,
                "target_size": int(size),
                "norm_mode": norm_mode,
                "crop": {
                    "count": int(crop_count),
                    "min_size": int(crop_min),
                    "max_size": int(crop_max),
                },
                "augmentation_pipeline": tf_pipeline,
                "expected_items": int(expected_total),
                "actual_items": int(actual_total),
            },
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

    train_crop_conf = {
        "count": args.prep_train_crop_count,
        "min_size": args.prep_train_crop_min,
        "max_size": args.prep_train_crop_max,
    }
    val_crop_conf = {
        "count": args.prep_val_crop_count,
        "min_size": args.prep_val_crop_min,
        "max_size": args.prep_val_crop_max,
    }
    test_crop_conf = {
        "count": args.prep_test_crop_count,
        "min_size": args.prep_test_crop_min,
        "max_size": args.prep_test_crop_max,
    }

    prepare_cfg = (args.aug_config_data or {}).get("prepare", {})
    _export_seg_split(seg_index, tr_idx, "train", args.prep_out_dir, args.ovseg_img_size,
                      norm_mode=args.prep_norm_mode, alpha=args.vis_max_alpha,
                      prepare_aug_config=prepare_cfg)
    _export_seg_split(seg_index, va_idx, "val", args.prep_out_dir, args.ovseg_img_size,
                        norm_mode=args.prep_norm_mode, alpha=args.vis_max_alpha,
                        prepare_aug_config=prepare_cfg)

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
                _export_seg_split(seg_index_test, test_idxs, "test", args.prep_out_dir, args.ovseg_img_size,
                                  norm_mode=args.prep_norm_mode, alpha=args.vis_max_alpha,
                                  prepare_aug_config=prepare_cfg)
        except Exception as exc:
            print(f"[!] не удалось подготовить тестовый набор: {exc}")

    print("[OK] подготовка датасетов завершена")



def ovseg_train(args, device):
    out_dir = stamp_out_dir(args.out_dir)
    print(f"[i] results will be saved to: {out_dir}")
    using_tiles = bool(getattr(args, "tiles_train", None))
    if using_tiles:
        seg_index = MaskTilesIndex(args.tiles_train, class_aliases=args.class_aliases)
        val_index = seg_index
        if args.tiles_val:
            val_index = MaskTilesIndex(args.tiles_val, class_aliases=args.class_aliases)
            tr_idx_orig = list(range(len(seg_index.items)))
            va_idx = list(range(len(val_index.items)))
        else:
            tr_idx_orig, va_idx = seg_index.split_by_images(val_ratio=args.val_ratio, seed=42)
    else:
        seg_index = SegIndex(args.images_dir, args.coco_json, class_aliases=args.class_aliases)
        val_index = seg_index
        tr_idx_orig, va_idx = seg_index.split_by_images(val_ratio=args.val_ratio, seed=42)

    total_indexed = len(seg_index.items)
    print(
        f"[i] dataset built: {total_indexed} images with masks across "
        f"{len(seg_index.classes)} classes"
    )

    if using_tiles:
        per_root = getattr(seg_index, "per_root_counts", {})
        if per_root:
            print("[i] per-tileset image counts:")
            for name, count in sorted(per_root.items()):
                print(f"      {name}: {count}")
        dropped_empty = getattr(seg_index, "dropped_empty", {})
        for name, count in sorted(dropped_empty.items()):
            if count > 0:
                print(f"[i] dropped {count} empty tiles from {name} (keep_empty=False)")
    else:
        if os.path.isdir(args.images_dir):
            try:
                total_files = _count_image_files(args.images_dir)
                print(
                    f"[i] source directory contains {total_files} image files; "
                    f"{total_indexed} matched the annotations"
                )
            except Exception as exc:
                print(f"[i] warning: unable to count files in {args.images_dir}: {exc}")

        buckets = Counter()
        for rec in seg_index.items:
            try:
                rel = os.path.relpath(rec["path"], args.images_dir)
            except ValueError:
                rel = os.path.basename(rec["path"])
            rel = rel.replace("\\", "/")
            parts = [p for p in rel.split("/") if p]
            bucket = parts[0] if len(parts) > 1 else "."
            buckets[bucket] += 1

        if len(buckets) > 1:
            print("[i] per-subdirectory image counts:")
            for name, count in sorted(buckets.items()):
                label = name if name != "." else "(root)"
                print(f"      {label}: {count}")

    if val_index is seg_index:
        print(
            f"[i] split summary: train={len(tr_idx_orig)} | val={len(va_idx)} "
            f"(val_ratio={args.val_ratio:.2f})"
        )
    else:
        print(
            f"[i] split summary: train tiles={len(tr_idx_orig)} | val tiles={len(va_idx)} "
            "(external validation roots)"
        )
    tr_idx = []
    for _ in range(max(1, args.ovseg_dup)):
        tr_idx += tr_idx_orig
    print(f"[i] training images: {len(tr_idx_orig)} (dup x{args.ovseg_dup} -> {len(tr_idx)}) | val images: {len(va_idx)}")

    dataset_cfg = (args.aug_config_data or {}).get("dataset", {})
    ds_tr = SegDataset(
        seg_index,
        tr_idx,
        train=True,
        size=args.ovseg_img_size,
        norm_mode="clip",
        aug_dump_dir=args.aug_dump_dir,
        aug_dump_limit=args.aug_dump_limit,
        aug_config=dataset_cfg,
    )
    ds_va = SegDataset(
        val_index,
        va_idx,
        train=False,
        size=args.ovseg_img_size,
        norm_mode="clip",
        aug_config=dataset_cfg,
    )
    dl_tr = DataLoader(ds_tr, batch_size=args.ovseg_batch_size, shuffle=True,  num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    dl_va = DataLoader(ds_va, batch_size=args.ovseg_batch_size, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    train_meta = getattr(seg_index, "meta_by_path", {})
    val_meta = getattr(val_index, "meta_by_path", train_meta)
    train_root_hint = args.images_dir if not using_tiles else ""
    val_root_hint = train_root_hint
    if using_tiles:
        roots = getattr(seg_index, "root_infos", {})
        if roots:
            first = next(iter(roots.values()))
            train_root_hint = str(first.get("images_dir", ""))
        val_roots = getattr(val_index, "root_infos", roots)
        if val_roots:
            first_val = next(iter(val_roots.values()))
            val_root_hint = str(first_val.get("images_dir", ""))

    def _resolve_rel_path(path_str: str, meta_map: Dict[str, Dict[str, str]], root_hint: str = "") -> str:
        info = meta_map.get(path_str)
        if info:
            rel = info.get("relative_path")
            dataset = info.get("dataset")
            if rel:
                rel = rel.replace("\\", "/")
                if dataset and dataset not in {"", "."}:
                    return f"{dataset}/{rel}"
                return rel
        if root_hint:
            try:
                rel = os.path.relpath(path_str, root_hint)
                return rel.replace("\\", "/")
            except Exception:
                pass
        return path_str

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
    ckpt_dir_train = os.path.join(ckpt_dir, "best_train_loss")
    ckpt_dir_val = os.path.join(ckpt_dir, "best_val_miou")
    os.makedirs(ckpt_dir_train, exist_ok=True)
    os.makedirs(ckpt_dir_val, exist_ok=True)
    ckpt_mgr_train = TopKCheckpointManager(ckpt_dir_train, prefix="ovseg", k=5, mode="min")
    ckpt_mgr_val = TopKCheckpointManager(ckpt_dir_val, prefix="ovseg", k=5, mode="max")

    vis_root = os.path.join(out_dir, "vis")
    os.makedirs(vis_root, exist_ok=True)
    vis_count = max(1, min(args.epochs, 10))
    vis_epochs = sorted({int(round(ep)) for ep in np.linspace(1, args.epochs, num=vis_count)})
    vis_epochs = [ep for ep in vis_epochs if 1 <= ep <= args.epochs]

    for ep in range(1, args.epochs+1):
        model.train(); losses = []; seen = 0

        steps_total = max(1, len(dl_tr))
        checks = max(1, int(getattr(args, "val_checks_per_epoch", 1)))
        check_points = {steps_total}
        if len(dl_tr) > 0 and checks > 1:
            for idx in range(1, checks):
                step = math.ceil(idx * len(dl_tr) / checks)
                check_points.add(min(steps_total, max(1, step)))
        next_checks = iter(sorted(check_points))
        next_check = next(next_checks, None)
        last_val = None

        for step, (x, m, _) in enumerate(
            tqdm(dl_tr, desc=f"OVSeg Train {ep}/{args.epochs}", leave=False), start=1
        ):
            x = x.to(device); m = m.to(device).long()
            out = model(x)["out"]
            if out.shape[-2:] != m.shape[-2:]:
                out = F.interpolate(out, size=m.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(out, m)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item())); seen += x.size(0)

            while next_check is not None and step >= next_check:
                val_loss, m_val, K = evaluate_segmentation_full(model, dl_va, device, criterion)
                print(
                    f"[ep {ep:03d}] val@{step}/{steps_total}: "
                    f"loss={val_loss:.4f}  pixel_acc={m_val['pixel_acc']:.4f}  "
                    f"mIoU={m_val['miou']:.4f}  F1={m_val['f1_macro']:.4f}"
                )
                last_val = (val_loss, m_val, K)
                next_check = next(next_checks, None)

        if len(dl_tr) == 0:
            val_loss, m_val, K = evaluate_segmentation_full(model, dl_va, device, criterion)
            print(
                f"[ep {ep:03d}] val@0/{steps_total}: "
                f"loss={val_loss:.4f}  pixel_acc={m_val['pixel_acc']:.4f}  "
                f"mIoU={m_val['miou']:.4f}  F1={m_val['f1_macro']:.4f}"
            )
            last_val = (val_loss, m_val, K)

        avg_loss = (sum(losses) / max(1, len(losses)))
        print(f"[ep {ep:03d}] train_loss={avg_loss:.4f}")

        if last_val is None:
            val_loss, m_val, K = evaluate_segmentation_full(model, dl_va, device, criterion)
            print(
                f"[ep {ep:03d}] val@{steps_total}/{steps_total}: "
                f"loss={val_loss:.4f}  pixel_acc={m_val['pixel_acc']:.4f}  "
                f"mIoU={m_val['miou']:.4f}  F1={m_val['f1_macro']:.4f}"
            )
        else:
            val_loss, m_val, K = last_val

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

        extra_payload = {"classes": seg_index.classes}
        ckpt_mgr_train.save(model.state_dict(), ep, avg_loss, m_val, extra=extra_payload)
        ckpt_mgr_val.save(model.state_dict(), ep, m_val['miou'], m_val, extra=extra_payload)

        if ep in vis_epochs:
            has_val_samples = len(dl_va) > 0
            has_train_samples = len(dl_tr) > 0
            if not has_val_samples and not has_train_samples:
                continue

            ep_dir = os.path.join(vis_root, f"ep{ep:03d}")
            os.makedirs(ep_dir, exist_ok=True)
            ep_idx_dir = os.path.join(ep_dir, "idx")
            ep_color_dir = os.path.join(ep_dir, "color")
            ep_over_dir = os.path.join(ep_dir, "overlay")
            if has_val_samples:
                os.makedirs(ep_idx_dir, exist_ok=True)
                os.makedirs(ep_color_dir, exist_ok=True)
                os.makedirs(ep_over_dir, exist_ok=True)

            train_gt_dir = os.path.join(ep_dir, "train_gt")
            train_pred_dir = os.path.join(ep_dir, "train_pred")
            if has_train_samples:
                os.makedirs(train_gt_dir, exist_ok=True)
                os.makedirs(train_pred_dir, exist_ok=True)

            mean = np.array(CLIP_MEAN, dtype=np.float32)
            std = np.array(CLIP_STD, dtype=np.float32)
            model.eval()
            with torch.no_grad():
                if has_val_samples:
                    max_vis_batches = min(2, max(1, len(dl_va)))
                    saved = 0
                    for b_idx, (x_va, _, paths) in enumerate(dl_va):
                        if b_idx >= max_vis_batches:
                            break
                        logits = model(x_va.to(device))["out"]
                        if logits.shape[-2:] != x_va.shape[-2:]:
                            logits = F.interpolate(logits, size=x_va.shape[-2:], mode="bilinear", align_corners=False)
                        probs = torch.softmax(logits, dim=1)
                        for i in range(x_va.size(0)):
                            img = x_va[i].detach().cpu().numpy().transpose(1, 2, 0)
                            img = (img * std[None, None, :]) + mean[None, None, :]
                            img = np.clip(img, 0.0, 1.0)
                            img_rgb = (img * 255.0).astype(np.uint8)

                            acc = probs[i].detach().cpu().numpy()

                            raw_path = paths[i]
                            rel = _resolve_rel_path(raw_path, val_meta, val_root_hint)
                            safe_base = re.sub(r"[^0-9a-zA-Z._/-]", "_", rel)
                            safe_base = safe_base.replace(os.sep, "__").replace("/", "__")
                            safe_base = safe_base.replace("..", "__")
                            safe_base = os.path.splitext(safe_base)[0]
                            fname = f"{saved:04d}__{safe_base}"

                            idx_path = os.path.join(ep_idx_dir, fname + ".png")
                            color_path = os.path.join(ep_color_dir, fname + ".png")
                            over_path = os.path.join(ep_over_dir, fname + ".jpg")

                            _, color_map = save_index_and_color_maps(acc, seg_index.classes, idx_path, color_path,
                                                                     vis_thr=0.5, palette=LS_PALETTE, add_legend=True)
                            save_overlay(img_rgb, color_map, over_path, alpha=args.vis_max_alpha, blur=args.vis_blur,
                                         add_legend=True, class_names=seg_index.classes, palette=LS_PALETTE)
                            saved += 1

                if has_train_samples:
                    max_train_batches = min(2, max(1, len(dl_tr)))
                    saved_train = 0
                    for b_idx, (x_tr_vis, m_tr_vis, paths_tr) in enumerate(dl_tr):
                        if b_idx >= max_train_batches:
                            break
                        logits_tr = model(x_tr_vis.to(device))["out"]
                        if logits_tr.shape[-2:] != x_tr_vis.shape[-2:]:
                            logits_tr = F.interpolate(logits_tr, size=x_tr_vis.shape[-2:], mode="bilinear", align_corners=False)
                        probs_tr = torch.softmax(logits_tr, dim=1)
                        preds_tr = probs_tr.argmax(1).cpu().numpy().astype(np.uint8)
                        masks_tr = m_tr_vis.cpu().numpy().astype(np.uint8)

                        for i in range(x_tr_vis.size(0)):
                            raw_path = paths_tr[i]
                            rel = _resolve_rel_path(raw_path, train_meta, train_root_hint)
                            safe_base = re.sub(r"[^0-9a-zA-Z._/-]", "_", rel)
                            safe_base = safe_base.replace(os.sep, "__").replace("/", "__")
                            safe_base = safe_base.replace("..", "__")
                            safe_base = os.path.splitext(safe_base)[0]
                            fname = f"{saved_train:04d}__{safe_base}"

                            gt_mask = masks_tr[i].copy()
                            gt_mask[gt_mask == 255] = 0
                            gt_color = _mask_to_color(gt_mask, seg_index.classes)
                            pred_color = _mask_to_color(preds_tr[i], seg_index.classes)

                            gt_path = os.path.join(train_gt_dir, fname + ".png")
                            pred_path = os.path.join(train_pred_dir, fname + ".png")

                            cv2.imwrite(gt_path, gt_color)
                            cv2.imwrite(pred_path, pred_color)
                            saved_train += 1
            model.train()

def ovseg_infer(args, device):
    assert args.ckpt, "--ckpt is required for ovseg_infer"
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
            if args.tiles_train or args.tiles_val:
                roots = args.tiles_train or args.tiles_val
                seg_index = MaskTilesIndex(roots, class_aliases=args.class_aliases)
            else:
                seg_index = SegIndex(
                    args.images_dir if args.images_dir else args.test_dir,
                    args.coco_json,
                    class_aliases=args.class_aliases,
                )
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

    model = OvSegModel(args.ovseg_model, args.ovseg_ckpt, num_classes,
                       freeze_backbone=False, decoder_channels=args.ovseg_decoder_ch,
                       decoder_dropout=args.ovseg_decoder_dropout, device=device).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    base_out_dir = args.out_dir if args.out_dir else os.path.join(DEFAULT_OUT_DIR, "ovseg_infer")
    out_dir = stamp_out_dir(base_out_dir)
    out_idx   = os.path.join(out_dir, "ovseg_idx");   os.makedirs(out_idx, exist_ok=True)
    out_color = os.path.join(out_dir, "ovseg_color"); os.makedirs(out_color, exist_ok=True)
    out_over  = os.path.join(out_dir, "ovseg_over");  os.makedirs(out_over, exist_ok=True)

    test_dir = args.test_dir if args.test_dir else args.images_dir
    img_paths = []
    for root, _, files in os.walk(test_dir):
        for f in files:
            if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):
                img_paths.append(os.path.join(root, f))
    img_paths.sort()

    for path in tqdm(img_paths, desc="OVSeg infer"):
        src = cv2.imread(path, cv2.IMREAD_COLOR)
        if src is None:
            print(f"[!] cannot read {path}")
            continue
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        H, W = src.shape[:2]

        long_side = max(H, W)
        scale = args.ovseg_img_size / long_side
        newH, newW = int(round(H*scale)), int(round(W*scale))
        img = cv2.resize(src, (newW, newH), interpolation=cv2.INTER_AREA)
        padH = args.ovseg_img_size - newH
        padW = args.ovseg_img_size - newW
        img = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_CONSTANT, value=(0,0,0))

        mean, std = np.array(CLIP_MEAN), np.array(CLIP_STD)


        x = (img.astype(np.float32)/255.0 - mean[None,None,:]) / std[None,None,:]
        x = torch.from_numpy(x.transpose(2,0,1)).unsqueeze(0).float().to(device)

        with torch.no_grad():
            logits = model(x)["out"]

            logit_h, logit_w = logits.shape[-2:]
            if padH > 0 or padW > 0:
                valid_h = logit_h - int(round(logit_h * padH / args.ovseg_img_size))
                valid_w = logit_w - int(round(logit_w * padW / args.ovseg_img_size))
                valid_h = max(1, min(valid_h, logit_h))
                valid_w = max(1, min(valid_w, logit_w))
                logits = logits[..., :valid_h, :valid_w]

            logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
            pred = logits.argmax(1)[0].detach().cpu().numpy()  # 0..C

        base = os.path.splitext(os.path.basename(path))[0]
        idx_path   = os.path.join(out_idx,   base + ".png")
        color_path = os.path.join(out_color, base + ".png")
        over_path  = os.path.join(out_over,  base + ".jpg")# классы 1..C

        base = os.path.splitext(os.path.basename(path))[0]
        idx_path   = os.path.join(out_idx,   base + ".png")
        color_path = os.path.join(out_color, base + ".png")
        over_path  = os.path.join(out_over,  base + ".jpg")

        total_channels = num_classes + 1
        acc = np.zeros((total_channels, H, W), np.float32)
        for c in range(total_channels):
            acc[c] = (pred == c).astype(np.float32)
        _, color_map = save_index_and_color_maps(acc, class_names, idx_path, color_path,
                                                 vis_thr=0.5, palette=LS_PALETTE, add_legend=args.vis_legend)
        save_overlay(src, color_map, over_path, alpha=args.vis_max_alpha, blur=args.vis_blur,
                     add_legend=args.vis_legend, class_names=class_names, palette=LS_PALETTE)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "ovseg_train":
        ovseg_train(args, device); return
    if args.mode == "ovseg_infer":
        ovseg_infer(args, device)
        return
    if args.mode == "seg_prepare":
        seg_prepare(args); return


if __name__ == "__main__":
    main()
