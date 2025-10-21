import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Сравнивает список изображений в выгрузке Label Studio "
            "с файлами в raw каталоге и train/val/test после конвертации."
        )
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("datasets/annotations/result_coco_94.json"),
        help="Путь к COCO-аннотации из Label Studio.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("datasets/raw_images"),
        help="Каталог с исходными изображениями.",
    )
    parser.add_argument(
        "--converted-root",
        type=Path,
        default=Path("lposs/data/facade_damage"),
        help="Каталог, куда convert_labelstudio_facade.py разложил split'ы.",
    )
    return parser.parse_args()


_LABEL_STUDIO_PREFIX = re.compile(r"^[0-9a-f]{8,}-")


def _sanitise_name(name: str) -> str:
    base = Path(name).name
    if _LABEL_STUDIO_PREFIX.match(base):
        return base.split("-", 1)[1]
    return base


def _collect_present(paths: Iterable[Path]) -> set[str]:
    present: set[str] = set()
    for path in paths:
        if not path.is_file():
            continue
        present.add(_sanitise_name(path.name))
        present.add(path.name)
    return present


def _load_annotation_names(ann_path: Path) -> set[str]:
    with ann_path.open("r", encoding="utf-8") as fh:
        coco = json.load(fh)
    return {
        _sanitise_name(img["file_name"])
        for img in coco.get("images", [])
    }


def _ensure_exists(path: Path, description: str) -> bool:
    if path.exists():
        return True
    print(f"[!] {description} не найден: {path}")
    return False


def main() -> int:
    args = _parse_args()

    if not _ensure_exists(args.annotations, "Файл аннотации"):
        return 1
    if not _ensure_exists(args.raw_dir, "Каталог raw изображений"):
        return 1
    if not _ensure_exists(args.converted_root, "Каталог с train/val/test"):
        return 1

    annotated = _load_annotation_names(args.annotations)
    raw_present = _collect_present(args.raw_dir.iterdir())
    split_present = _collect_present(
        args.converted_root.glob("images/*/*")
        if (args.converted_root / "images").exists()
        else []
    )

    missing_in_raw = sorted(annotated - raw_present)
    missing_in_splits = sorted(annotated - split_present)

    print(f"Всего в аннотации: {len(annotated)}")
    print(f"Есть в raw_images: {len(raw_present & annotated)}")
    print(f"Есть в split'ах: {len(split_present & annotated)}")

    if missing_in_raw:
        print("\nНет среди исходных изображений:")
        print("\n".join(missing_in_raw))

    if missing_in_splits:
        print("\nОтсутствуют в train/val/test:")
        print("\n".join(missing_in_splits))

    split_counts: Counter[str] = Counter()
    images_root = args.converted_root / "images"
    if images_root.exists():
        for split_dir in images_root.glob("*"):
            if not split_dir.is_dir():
                continue
            files = {
                _sanitise_name(p.name)
                for p in split_dir.iterdir()
                if p.is_file()
            }
            split_counts[split_dir.name] = len(files & annotated)

    if split_counts:
        print("\nРазмеры сплитов (только файлы из аннотации):")
        for split, count in sorted(split_counts.items()):
            print(f"{split}: {count}")
    else:
        print(
            "\nНе найден каталог с изображениями split'ов. "
            "Убедитесь, что запускали convert_labelstudio_facade.py."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())