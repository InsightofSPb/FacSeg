# FacSeg utilities

## Dataset label summary helper

Use the `tools/dataset_label_summary.py` script to inspect annotation labels in
JSON (e.g., DACL10K) or XML (e.g., Prova) datasets, generate per-class
visualisations, and optionally emit a JSON template for remapping classes.

### Basic usage

```bash
python tools/dataset_label_summary.py <path-to-annotation-folder>
```

The command recursively searches the folder for `.json` and `.xml` files, prints
all discovered labels, how many annotation files each label appears in, and up
to ten sample filenames per class. When a matching image is found the script can
also render preview images with the selected class highlighted.

### Useful options

* `--samples-per-class N` &mdash; change how many sample files to show per class.
* `--mapping-template path/to/output.json` &mdash; write an empty mapping template
  (keys are discovered labels, values are left blank for you to fill in).
* `--extensions .json .xml` &mdash; override the set of annotation file extensions
  that will be parsed.
* `--image-root /path/to/images` &mdash; point the script at the directory that
  stores source images (defaults to the annotation folder).
* `--image-extensions .jpg .png` &mdash; customise the list of extensions used when
  resolving image paths.
* `--viz-output previews/` &mdash; store up to *N* (see `--samples-per-class`) image
  previews per class with the corresponding geometry rendered on top of the
  source photo.

### Examples

List classes for the DACL10K annotations and create a mapping template:

```bash
python tools/dataset_label_summary.py /path/to/dacl10k/annotations \
  --mapping-template tools/mappings/dacl10k_to_facseg.json
```

Check labels in the Prova dataset (XML annotations):

```bash
python tools/dataset_label_summary.py /path/to/prova/annotations
```

To generate quick-look previews for every class:

```bash
python tools/dataset_label_summary.py /path/to/dacl10k/annotations \
  --image-root /path/to/dacl10k/images \
  --viz-output previews \
  --samples-per-class 10
```

The samples printed for each label can be used to manually inspect
representative files and verify class mappings, while the rendered previews
provide immediate visual confirmation of the annotation quality.

### Predefined mapping dictionaries

Ready-to-use mappings that collapse raw dataset labels into the FacSeg target
classes live under `tools/mappings/`:

* `dacl10k_to_facseg.json` &mdash; remaps the 19 DACL10K labels into the seven
  categories we keep for training (e.g. both `spalling` and `hollowareas`
  become `SPALLING`).
* `prova_to_facseg.json` &mdash; translates Pascal VOC class names from the Prova
  dataset (bounding boxes) into the same FacSeg categories.
* `portrait_spalling_cracks_values.json` &mdash; maps foreground pixel values in the
  `portrait_spalling_cracks` masks to the `SPALLING` category.
* `facade_damage_values.json` &mdash; mirrors the pixel indices produced by
  `convert_labelstudio_facade.py`, mapping them to the grouped FacSeg training
  labels (`background`, `DAMAGE`, `WATER_STAIN`, `ORNAMENT_INTACT`, `REPAIRS`,
  `TEXT_OR_IMAGES`). Use this file whenever you tile the exported masks.

Pass these JSON files to the tiling helper via `--class-mapping` or
`--mask-value-mapping` to reuse the agreed mapping without rebuilding it from
scratch.

## Filtering raw datasets before tiling

`tools/filter_dataset_by_mapping.py` scans annotations or masks, keeps only the
files that contain classes present in your mapping, and copies/symlinks the
matching images (and masks/annotations) into a clean output folder. This lets
you prune large datasets before tiling so you do not waste time on irrelevant
photos.

```bash
python tools/filter_dataset_by_mapping.py \
  /path/to/dacl10k/train/img \
  /path/to/dacl10k_filtered/train \
  --dataset-type json-polygons \
  --annotations-dir /path/to/dacl10k/train/ann \
  --class-mapping tools/mappings/dacl10k_to_facseg.json \
  --copy-mode symlink \
  --manifest /path/to/dacl10k_filtered/train/summary.json
```

For Prova (Pascal VOC XML boxes):

```bash
python tools/filter_dataset_by_mapping.py \
  /path/to/prova/images \
  /path/to/prova_filtered \
  --dataset-type xml-bboxes \
  --annotations-dir /path/to/prova/annotations \
  --class-mapping tools/mappings/prova_to_facseg.json
```

For portrait_spalling_cracks (mask PNGs):

```bash
python tools/filter_dataset_by_mapping.py \
  /path/to/portrait/images \
  /path/to/portrait_filtered \
  --dataset-type mask-png \
  --masks-dir /path/to/portrait/masks \
  --mask-value-mapping tools/mappings/portrait_spalling_cracks_values.json
```

Pass `--dry-run` to only report how many files would be retained. Use
`--copy-mode symlink` if you prefer lightweight symlinks over physical copies.

## Unified tiling & augmentation helper

`tools/prepare_dataset_tiles.py` tiles raw images and annotations into
segmentation-ready crops with optional augmentations and QA overlays. It
supports three dataset flavours:

* `json-polygons` &mdash; polygon annotations such as DACL10K (`.json`).
* `xml-bboxes` &mdash; Pascal VOC style bounding boxes such as Prova (`.xml`).
* `mask-png` &mdash; paired image/mask datasets where each mask encodes class IDs as
  pixel values (e.g., `portrait_spalling_cracks`).

### What the script does

1. Resolves companion images/annotations depending on the dataset type.
2. Converts raw annotations into class-index masks using the supplied mapping
   (`--class-mapping` or `--mask-value-mapping`).
3. Splits every image/mask pair into tiles using the requested tile size,
   stride, padding, and foreground filters.
4. Optionally applies offline augmentations (`cutout`, `cutblur`, `cutmix`) and
   writes augmented copies alongside the base tiles.
5. Saves crops to `output/images/` and `output/masks/`, renders QA overlays when
   `--overlay-dir` is given, and (optionally) records tile statistics in a JSON
   manifest (`--metadata`).

Each run prints a concise configuration summary up front and reports how many
tiles were produced per class when finished, so you can immediately check that
the mapping and thresholds are correct.

### Required inputs per dataset type

| `--dataset-type` | Must provide | Notes |
|------------------|--------------|-------|
| `json-polygons`  | `--annotations-dir` with `.json` files<br>`--class-mapping` | Rasterises polygons/rectangles into masks. |
| `xml-bboxes`     | `--annotations-dir` with Pascal VOC `.xml` files<br>`--class-mapping` | Converts bounding boxes into filled mask regions. |
| `mask-png`       | `--masks-dir` with mask images<br>`--mask-value-mapping` | Reads existing masks and remaps pixel values. |

OpenCV (`opencv-python`) is required because the script relies on it for image
IO and geometric rasterisation.

### Commonly used flags

* `--tile-size H W` / `--stride SY SX` &mdash; control crop size and overlap.
* `--pad` &mdash; pad smaller images instead of skipping them.
* `--min-coverage` &mdash; minimum fraction of foreground pixels a tile must contain
  (use `--keep-empty` to keep the filtered-out crops).
* `--overlay-dir path/` &mdash; store blended QA previews that colourise every class
  in the generated mask.
* `--augmentations cutout cutmix` &mdash; enable one or more offline augmentations
  (`--augmentations-per-tile` duplicates each augmentation that many times).
* `--metadata manifest.json` &mdash; save run statistics (tile counts, pixel counts,
  configuration) for later auditing.

### Example: tiling DACL10K polygons

```bash
python tools/prepare_dataset_tiles.py \
  /path/to/dacl10k/images \
  /path/to/output/dacl10k_tiles \
  --dataset-type json-polygons \
  --annotations-dir /path/to/dacl10k/annotations \
  --class-mapping tools/mappings/dacl10k_to_facseg.json \
  --tile-size 1024 1024 \
  --stride 768 768 \
  --min-coverage 0.01 \
  --pad \
  --augmentations cutout \
  --augmentations-per-tile 2 \
  --overlay-dir /path/to/output/dacl10k_tiles/overlays \
  --metadata /path/to/output/dacl10k_tiles/manifest.json
```

### Example: converting Prova bounding boxes

```bash
python tools/prepare_dataset_tiles.py \
  /path/to/prova/images \
  /path/to/output/prova_tiles \
  --dataset-type xml-bboxes \
  --annotations-dir /path/to/prova/annotations \
  --class-mapping tools/mappings/prova_to_facseg.json \
  --tile-size 512 512 \
  --augmentations cutout cutblur \
  --overlay-dir /path/to/output/prova_tiles/overlays
```

### Example: portrait_spalling_cracks mask dataset

```bash
python tools/prepare_dataset_tiles.py \
  /path/to/portrait/images \
  /path/to/output/portrait_tiles \
  --dataset-type mask-png \
  --masks-dir /path/to/portrait/masks \
  --mask-value-mapping tools/mappings/portrait_spalling_cracks_values.json \
  --tile-size 768 768 \
  --augmentations cutmix \
  --overlay-dir /path/to/output/portrait_tiles/overlays
```

### Example: Label Studio facade masks (mask-png)

```bash
python -m lposs.tools.convert_labelstudio_facade \
  /path/to/labelstudio/result_coco.json \
  /path/to/source/images \
  --output-root /path/to/output/facade_damage \
  --auto-split train=0.8 val=0.2 \
  --seed 42

python tools/prepare_dataset_tiles.py \
  /path/to/output/facade_damage/images/train \
  /path/to/output/facade_damage_tiles/train \
  --dataset-type mask-png \
  --masks-dir /path/to/output/facade_damage/masks/train \
  --mask-value-mapping tools/mappings/facade_damage_values.json \
  --tile-size 1024 1024 \
  --stride 768 768 \
  --pad \
  --augmentations cutout cutmix cutblur \
  --augmentations-per-tile 2 \
  --overlay-dir /path/to/output/facade_damage_tiles/train/overlays \
  --metadata /path/to/output/facade_damage_tiles/train/manifest.json
```

The script writes cropped images to `output_dir/images/`, masks to
`output_dir/masks/`, and (optionally) overlays to the directory specified via
`--overlay-dir`. A manifest with aggregated tile statistics can be written via
`--metadata`. The closing summary printed to stdout shows the number of tiles
generated for each target class and reminds you where the artefacts were stored.

### Quick recipe: from filtering to training

The commands below mirror the exact folders you mentioned so you can go from
raw DACL10K annotations + personal facade imagery to a train/val dataset that
matches the model’s `1024×1024` training scale【F:lposs/segmentation/configs/_base_/datasets/facade_damage.py†L37-L61】.

1. **Filter DACL10K to the agreed FacSeg classes** (train/val similarly) so only
   images containing mapped defects remain:

   ```bash
   python tools/filter_dataset_by_mapping.py \
     /home/sasha/Facade_segmentation/datasets/dacl10k/train/img \
     /home/sasha/Facade_segmentation/datasets/filtered_dacl10k/train \
     --dataset-type json-polygons \
     --annotations-dir /home/sasha/Facade_segmentation/datasets/dacl10k/train/ann \
     --class-mapping tools/mappings/dacl10k_to_facseg.json \
     --copy-mode symlink
   ```

2. **Tile the filtered DACL10K split** with overlap and offline augmentations.
   A `1024×1024` tile size keeps the same scale the training pipeline expects,
   while a `768` stride ensures large images are fully covered with 25 % overlap:

   ```bash
   python tools/prepare_dataset_tiles.py \
     /home/sasha/Facade_segmentation/datasets/filtered_dacl10k/train/images \
     /home/sasha/Facade_segmentation/tiles/dacl10k/train \
     --dataset-type json-polygons \
     --annotations-dir /home/sasha/Facade_segmentation/datasets/filtered_dacl10k/train/annotations \
     --class-mapping tools/mappings/dacl10k_to_facseg.json \
     --tile-size 1024 1024 \
     --stride 768 768 \
     --pad \
     --min-coverage 0.01 \
     --augmentations cutout cutmix cutblur \
     --augmentations-per-tile 1 \
     --overlay-dir /home/sasha/Facade_segmentation/tiles/dacl10k/train/overlays \
     --metadata /home/sasha/Facade_segmentation/tiles/dacl10k/train/manifest.json
   ```

3. **Tile your own / internet facades** once their masks are prepared (pixel
   IDs mapped via `portrait_spalling_cracks_values.json` if they follow the same
   scheme). Adjust the `--masks-dir` to wherever the aligned PNG masks reside:

   ```bash
   python tools/prepare_dataset_tiles.py \
     /home/sasha/Facade_segmentation/datasets/raw_images \
     /home/sasha/Facade_segmentation/tiles/raw_facades \
     --dataset-type mask-png \
     --masks-dir /home/sasha/Facade_segmentation/datasets/raw_masks \
     --mask-value-mapping tools/mappings/portrait_spalling_cracks_values.json \
     --tile-size 1024 1024 \
     --stride 768 768 \
     --pad \
     --augmentations cutout cutmix cutblur \
     --overlay-dir /home/sasha/Facade_segmentation/tiles/raw_facades/overlays \
     --metadata /home/sasha/Facade_segmentation/tiles/raw_facades/manifest.json
   ```

4. **Launch training** once the tiled outputs are copied/merged into
   `data/facade_damage/images/{train,val}` and `masks/{train,val}` (see previous
   section). From the project root (`FacSeg/`) run the grouped configuration,
   which collapses the damage subclasses into a single `DAMAGE` label by
   default:

   ```bash
   cd /path/to/FacSeg
   python main.py --mode lposs_train --config lposs/configs/facade_grouped.yaml \
     training.dataset.data_root=/home/sasha/Facade_segmentation/dataset_final \
     --out_dir results/facade_grouped_run1
   ```

   The `--mode lposs_train` flag switches the CLI to the LPOSS/Hydra training
   pipeline (supplying `--config` without `--mode` also auto-selects it). Any
   additional trailing arguments such as `training.max_epochs=200` or
   `evaluate.task=[]` are forwarded to Hydra as overrides. The familiar
   `--epochs` and `--out_dir` options continue to work and map to
   `training.max_epochs` and `output` respectively for convenience. When tile
   folders are supplied via `--tiles-train` (and optionally `--tiles-val` or
   `--val_ratio`), the CLI materialises them into a temporary dataset under
   `<out_dir>/_lposs_tiles_dataset` and automatically overrides
   `training.dataset.data_root`. Batch size and duplication hints from
   `--ovseg_batch_size` / `--ovseg_dup` are also forwarded to the Hydra config
   when explicitly provided, so existing OVSeg commands continue to work after
   switching `--mode` to `lposs_train`.

   Before training you can double-check the merged tiles with
   `tools/preview_tiled_dataset.py`. It samples a handful of image/mask pairs
   from the specified roots and writes side-by-side overlays:

   ```bash
   python tools/preview_tiled_dataset.py \
     /home/sasha/Facade_segmentation/tiles/dacl10k/train \
     /home/sasha/Facade_segmentation/tiles/facade_damage/train \
     --num-samples 10 --output /home/sasha/Facade_segmentation/tiles/preview
   ```

Adjust the destination folders if you prefer different aggregation points for
train/val tiles. Repeat steps 1–3 for each split (e.g. replace `train` with
`val`) before triggering the final training command.

## DACL10K tiling helper

Use `tools/prepare_dacl10k_tiles.py` to convert filtered DACL10K images and
annotations into segmentation-ready tiles. The script reads the original JSON
annotations, applies your class mapping, rasterises the polygons into mask
indices, and writes cropped image/mask pairs.

```bash
python tools/prepare_dacl10k_tiles.py \
  /path/to/dacl10k/images \
  /path/to/dacl10k/annotations \
  /path/to/output/tiles \
  --class-mapping tools/mappings/dacl10k_to_facseg.json \
  --tile-size 1024 1024 \
  --stride 768 768 \
  --min-coverage 0.01 \
  --pad \
  --metadata /path/to/output/tiles/manifest.json
```

Key options:

* `--class-mapping` — required JSON mapping generated via the label summary
  helper. Empty values mean “ignore this class”.
* `--tile-size` and `--stride` — control the crop size and overlap.
* `--min-coverage` — drop tiles with less than the specified fraction of
  foreground pixels (unless `--keep-empty` is set).
* `--pad` — pad source images that are smaller than the tile size instead of
  skipping them.
* `--metadata` — write a manifest with tiling statistics and class counts.

The resulting folder contains `images/` and `masks/` subdirectories with
aligned crops that can be plugged into the existing segmentation pipeline.
