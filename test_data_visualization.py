import os, json, math, argparse, hashlib
from collections import defaultdict, Counter
import numpy as np
import cv2
from tqdm.auto import tqdm

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

def color_from_name(name):
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:6], 16)
    return (64 + (h & 0xFF) % 192, 64 + ((h >> 8) & 0xFF) % 192, 64 + ((h >> 16) & 0xFF) % 192)

def safe_path(images_dir, coco_file_name):
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

def draw_polygons_bgr(img_bgr, polygons, label, alpha=0.45):
    overlay = img_bgr.copy()
    r,g,b = color_from_name(label)
    color_bgr = (b,g,r)

    for poly in polygons:
        if not poly: continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        H, W = img_bgr.shape[:2]
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
        ipts = pts.astype(np.int32)
        cv2.fillPoly(overlay, [ipts], color_bgr)
        cx, cy = int(ipts[:,0].mean()), int(ipts[:,1].mean())
        cv2.putText(overlay, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, dst=img_bgr)
    return img_bgr

def annotate_and_save(img_bgr, anns, cat_map, save_path):
    labels_here = []
    for a in anns:
        if a.get("iscrowd", 0) == 1: 
            continue
        label = cat_map.get(a["category_id"], f"cat_{a['category_id']}")
        seg = a.get("segmentation", [])
        if isinstance(seg, list) and seg and isinstance(seg[0], list):
            img_bgr = draw_polygons_bgr(img_bgr, seg, label, alpha=0.45)
            labels_here.append(label)

    ext = os.path.splitext(save_path)[1].lower()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(save_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    elif ext == ".png":
        cv2.imwrite(save_path, img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(save_path, img_bgr)
    return labels_here, img_bgr

def make_mosaic(tiles, cols, tile_width, pad=8, bg_color=(255,255,255)):
    """tiles: list of BGR images (different sizes allowed)"""
    if not tiles: return None
    resized = []
    for im in tiles:
        h, w = im.shape[:2]
        scale = tile_width / float(w)
        new = cv2.resize(im, (tile_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        resized.append(new)

    rows = math.ceil(len(resized) / cols)
    mosaic_rows = []
    idx = 0
    for r in range(rows):
        row_imgs = resized[idx:idx+cols]
        idx += cols
        max_h = max(im.shape[0] for im in row_imgs)
        padded = []
        for im in row_imgs:
            h, w = im.shape[:2]
            if h < max_h:
                pad_bottom = max_h - h
                im = cv2.copyMakeBorder(im, 0, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=bg_color)
            padded.append(im)
        with_gaps = []
        for j, im in enumerate(padded):
            with_gaps.append(im)
            if j < len(padded)-1:
                with_gaps.append(np.full((max_h, pad, 3), bg_color, np.uint8))
        row = np.hstack(with_gaps)
        mosaic_rows.append(row)
        if r < rows-1:
            mosaic_rows.append(np.full((pad, row.shape[1], 3), bg_color, np.uint8))
    mosaic = np.vstack(mosaic_rows)
    return mosaic

def main(images_dir, coco_json, out_mosaic, cols, tile_width, annotated_fmt):
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    if "categories" in coco and coco["categories"]:
        cat_map = {c["id"]: c.get("name", f"cat_{c['id']}") for c in coco["categories"]}
    else:
        cat_map = DEFAULT_CATEGORIES.copy()

    anns_by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    images = [im for im in coco.get("images", []) if im["id"] in anns_by_img]
    if not images:
        raise RuntimeError("В COCO нет изображений с аннотациями.")

    ds_root = os.path.dirname(images_dir.rstrip("/"))
    annotated_dir = os.path.join(ds_root, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    tiles_for_mosaic = []
    for im in tqdm(images, desc="Annotating"):
        src_path = safe_path(images_dir, im["file_name"])
        if src_path is None:
            print(f"[!] не нашёл файл: {im['file_name']} в {images_dir}")
            continue

        img_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"[!] не удалось прочитать: {src_path}")
            continue

        base_tail = im["file_name"].split("-", 1)[-1] if "-" in im["file_name"] else im["file_name"]
        dst_name = os.path.splitext(base_tail)[0] + (".png" if annotated_fmt == "png" else ".jpg")
        dst_path = os.path.join(annotated_dir, dst_name)

        labels_here, annotated_bgr = annotate_and_save(img_bgr.copy(), anns_by_img[im["id"]], cat_map, dst_path)

        if labels_here:
            lab_counts = Counter(labels_here)
            legend = ", ".join([f"{k}×{v}" if v > 1 else k for k, v in lab_counts.items()])
            cv2.rectangle(annotated_bgr, (8, 8), (8 + 10*len(legend), 40), (255,255,255), thickness=-1)
            cv2.putText(annotated_bgr, legend, (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)

        tiles_for_mosaic.append(annotated_bgr)

    if tiles_for_mosaic:
        mosaic = make_mosaic(tiles_for_mosaic, cols=cols, tile_width=tile_width, pad=8, bg_color=(255,255,255))
        os.makedirs(os.path.dirname(out_mosaic) or ".", exist_ok=True)
        cv2.imwrite(out_mosaic, mosaic, [cv2.IMWRITE_PNG_COMPRESSION, 3] if out_mosaic.lower().endswith(".png") else [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[OK] Мозаика сохранена: {out_mosaic}")
        print(f"[OK] Индивидуальные изображения: {annotated_dir}")
    else:
        print("[!] Нет тайлов для мозаики (возможно, не найден ни один исходный файл).")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/images", help="Папка с исходными изображениями (без uuid-префиксов)")
    ap.add_argument("--coco_json", default="/home/sasha/Facade_segmentation/datasets/Chernyshevskaya/annotations/result_coco.json", help="Путь к COCO JSON (Label Studio → Brush labels to COCO)")
    ap.add_argument("--out", default="coco_mosaic.png", help="Куда сохранить мозаичный PNG/JPG")
    ap.add_argument("--cols", type=int, default=4, help="Сколько столбцов в коллаже")
    ap.add_argument("--tile_width", type=int, default=1100, help="Ширина одного тайла в коллаже, px")
    ap.add_argument("--annotated_fmt", choices=["jpg","png"], default="png", help="Формат сохранения одиночных картинок")
    args = ap.parse_args()
    main(args.images_dir, args.coco_json, args.out, args.cols, args.tile_width, args.annotated_fmt)
