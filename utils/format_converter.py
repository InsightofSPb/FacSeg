import sys, os, pathlib
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
dst = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else src / "out").resolve()
dst.mkdir(parents=True, exist_ok=True)

heic_exts = {".heic", ".HEIC"}
files = [p for p in src.rglob("*") if p.suffix in heic_exts]

for p in files:
    rel = p.relative_to(src)
    out_dir = (dst / rel.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (p.stem + ".jpg")

    img = Image.open(p)
    exif = img.info.get("exif")
    img = img.convert("RGB")
    img.save(out_file, "JPEG", quality=90, optimize=True, exif=exif)
    print("Saved:", out_file)
