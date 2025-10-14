import os
import sys
import json
import glob
import argparse
from copy import deepcopy
from collections import defaultdict

import numpy as np
import cv2

# progress bar (мягкая зависимость)
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# --- add project root to sys.path (на случай запуска из tools/) ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
# ------------------------------------------------------------------

from utils.utils import (
    DEFAULT_CATEGORIES,
    LS_PALETTE,
    _norm_name,
    load_aliases_json,
    hex_to_bgr,
)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def poly_to_mask(poly, H, W):
    """poly: list[[x,y], ...] → бинарная маска (uint8 {0,1})"""
    mask = np.zeros((H, W), np.uint8)
    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def draw_poly_overlay(img_bgr, poly, color_bgr, label=None):
    """Рисует на BGR-изображении."""
    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_bgr, [pts], isClosed=True, color=color_bgr, thickness=2)
    if label:
        x, y = pts.reshape(-1, 2)[0]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        y0 = max(0, y - th - 6)
        cv2.rectangle(img_bgr, (x, y0), (x + tw + 8, y0 + th + 6), color_bgr, thickness=-1)
        cv2.putText(
            img_bgr, label, (x + 4, y0 + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), thickness=1, lineType=cv2.LINE_AA
        )


def build_default_dacl_aliases():
    """
    Резервный словарь маппинга (нормализованные ключи) -> наш канон.
    При необходимости расширяй под реальные классы DACL10K.
    """
    m = {
        "crack": "CRACK",
        "spalling": "SPALLING",
        "delamination": "DELAMINATION",
        "missingelement": "MISSING_ELEMENT",
        "missing_part": "MISSING_ELEMENT",
        "waterstain": "WATER_STAIN",
        "efflorescence": "EFFLORESCENCE",
        "corrosion": "CORROSION",
        "repairs": "REPAIRS",
        "repair": "REPAIRS",
        "ornament": "ORNAMENT_INTACT",
        "ornament_intact": "ORNAMENT_INTACT",
        "graffiti": "TEXT_OR_IMAGES",
        "text": "TEXT_OR_IMAGES",
        "poster": "TEXT_OR_IMAGES",
        "sticker": "TEXT_OR_IMAGES",
        # часто встречающиеся:
        "peeling_paint": "SPALLING",
        "damp": "WATER_STAIN",
    }
    out = {}
    for k, v in m.items():
        out[_norm_name(k)] = v
    return out


def map_class_to_canonical(raw_name, aliases_tbl, dacl_fallback):
    """
    Вернёт каноническое имя (как в DEFAULT_CATEGORIES) либо None.
    1) Сначала проверяем пользовательские алиасы (aliases.json): alias -> canonical (norm).
    2) Потом fallback-таблицу.
    """
    n = _norm_name(raw_name)
    if n in aliases_tbl:
        canon_norm = aliases_tbl[n]  # напр. "crack"
        for k in sorted(DEFAULT_CATEGORIES.keys()):
            if _norm_name(DEFAULT_CATEGORIES[k]) == canon_norm:
                return DEFAULT_CATEGORIES[k]
        return None
    if n in dacl_fallback:
        return dacl_fallback[n]
    return None


def _derive_image_basename_and_ext(json_path: str):
    """
    DACL10K аннотации часто называются "<basename>.<img_ext>.json" (foo.jpg.json).
    Вернёт (base, forced_img_ext_or_None).
    """
    base_json = os.path.splitext(os.path.basename(json_path))[0]  # срезали .json
    base = base_json
    forced_img_ext = None
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        if base_json.lower().endswith(ext):
            base = base_json[: -len(ext)]
            forced_img_ext = ext
            break
    return base, forced_img_ext


def process_one(json_path, images_dir, include_set, aliases_tbl, dacl_fallback,
                out_matched, out_unmatched, save_filtered_json=True):
    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    H = int(ann["size"]["height"])
    W = int(ann["size"]["width"])

    # имя исходника
    base, forced_img_ext = _derive_image_basename_and_ext(json_path)

    # поиск картинки
    img_path = None
    search_exts = [e for e in (forced_img_ext,) if e] + [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    for ext in search_exts:
        p = os.path.join(images_dir, base + ext)
        if os.path.isfile(p):
            img_path = p
            break
    if img_path is None:
        tried = [base + e for e in search_exts]
        print(f"[!] image not found for {json_path} | tried: {tried[:3]}{'...' if len(tried) > 3 else ''}")
        return 0, 0

    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"[!] cannot read image: {img_path}")
        return 0, 0

    matched_objs = []         # список (obj_deepcopy, canon_name)
    unmatched_polys = []      # список (poly, raw_name)

    for obj in ann.get("objects", []):
        if obj.get("geometryType") != "polygon":
            continue
        raw_name = obj.get("classTitle", "")
        poly = obj.get("points", {}).get("exterior", [])
        if not poly or len(poly) < 3:
            continue
        canon = map_class_to_canonical(raw_name, aliases_tbl, dacl_fallback)
        if canon is not None and canon in include_set:
            o = deepcopy(obj)
            o["classTitle"] = canon  # переписываем на канон
            matched_objs.append((o, canon))
        else:
            unmatched_polys.append((poly, raw_name))

    # --- matched: сохраняем исходник, маску и фильтрованный JSON ---
    if matched_objs:
        include_list = sorted(list(include_set))
        name_to_idx = {nm: i + 1 for i, nm in enumerate(include_list)}
        mask = np.zeros((H, W), np.uint8)  # 0=фон, 1..C

        for obj, canon in matched_objs:
            idx = name_to_idx[canon]
            poly = obj.get("points", {}).get("exterior", [])
            pm = poly_to_mask(poly, H, W)    # {0,1}
            mask[pm > 0] = idx               # присваиваем индекс класса

        # создаём папки
        ensure_dir(os.path.join(out_matched, "images"))
        ensure_dir(os.path.join(out_matched, "masks"))
        if save_filtered_json:
            ensure_dir(os.path.join(out_matched, "ann"))

        # пишем изображение и маску
        cv2.imwrite(os.path.join(out_matched, "images", base + ".jpg"), img_bgr)
        cv2.imwrite(os.path.join(out_matched, "masks",  base + ".png"),  mask)

        # пишем отфильтрованный JSON (в исходном формате, только нужные объекты)
        if save_filtered_json:
            filtered = {
                "description": ann.get("description", ""),
                "tags": ann.get("tags", []),
                "size": {"height": int(H), "width": int(W)},
                "objects": [o for (o, _) in matched_objs],
                # полезная служебная инфа:
                "meta": {
                    "note": "Filtered to canonical classes",
                    "canonical_classes": include_list,
                },
            }
            with open(os.path.join(out_matched, "ann", base + ".json"), "w", encoding="utf-8") as fw:
                json.dump(filtered, fw, ensure_ascii=False)

    # --- unmatched: рисуем оверлей с подписями для ручного просмотра ---
    if unmatched_polys:
        vis_bgr = img_bgr.copy()
        palette_names = list(DEFAULT_CATEGORIES.values())
        for i, (poly, raw_name) in enumerate(unmatched_polys):
            pal_name = palette_names[i % len(palette_names)]
            color = hex_to_bgr(LS_PALETTE.get(pal_name, "#FF00FF"))
            draw_poly_overlay(vis_bgr, poly, color, label=str(raw_name))
        ensure_dir(os.path.join(out_unmatched, "overlays"))
        cv2.imwrite(os.path.join(out_unmatched, "overlays", base + ".jpg"), vis_bgr)

    return int(bool(matched_objs)), int(bool(unmatched_polys))


def main():
    ap = argparse.ArgumentParser("DACL10K -> фильтр под наши классы + визуализация остальных")
    ap.add_argument("--images_dir", required=True, help="Папка с изображениями DACL10K")
    ap.add_argument("--ann_dir", required=True, help="Папка с JSON-аннотациями (по одному на изображение)")
    ap.add_argument("--out_matched", required=True, help="Куда класть совпавшие (images/, masks/, ann/)")
    ap.add_argument("--out_unmatched", required=True, help="Куда класть оверлеи с прочими классами")
    ap.add_argument("--aliases_json", default="", help="aliases.json (опционально)")
    ap.add_argument(
        "--only_classes",
        nargs="*",
        default=None,
        help="Список наших канонических классов; по умолчанию — все DEFAULT_CATEGORIES",
    )
    ap.add_argument("--no_json", action="store_true", help="Не сохранять фильтрованный JSON (по умолчанию сохраняем)")
    args = ap.parse_args()

    aliases_tbl = load_aliases_json(args.aliases_json) if args.aliases_json else {}
    dacl_fallback = build_default_dacl_aliases()

    include_set = set(args.only_classes) if args.only_classes else set(DEFAULT_CATEGORIES.values())
    os.makedirs(args.out_matched, exist_ok=True)
    os.makedirs(args.out_unmatched, exist_ok=True)

    jsons = sorted(glob.glob(os.path.join(args.ann_dir, "*.json")))
    m_cnt = u_cnt = 0

    pbar = tqdm(jsons, total=len(jsons), desc="DACL10K", unit="img")
    for jp in pbar:
        m, u = process_one(
            jp, args.images_dir, include_set, aliases_tbl, dacl_fallback,
            args.out_matched, args.out_unmatched, save_filtered_json=not args.no_json
        )
        m_cnt += m
        u_cnt += u
        pbar.set_postfix_str(f"matched={m_cnt} unmatched={u_cnt}")

    print(f"[i] done. matched images: {m_cnt}, unmatched overlays: {u_cnt}")


if __name__ == "__main__":
    main()