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
  --mapping-template dacl10k_label_mapping.json
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

Key capabilities:

* Applies a user-provided class mapping (raw label &rarr; target label) or mask
  value mapping (pixel value &rarr; target label).
* Cuts large images into tiles with configurable size, stride, overlap, and
  optional padding for smaller inputs.
* Optionally keeps tiles with no foreground (`--keep-empty`) or filters them via
  `--min-coverage`.
* Generates QA overlays for every saved crop (`--overlay-dir`).
* Offline augmentations: `cutout`, `cutblur`, and `cutmix`, each of which can
  produce multiple augmented copies per tile (`--augmentations-per-tile`).

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

The script writes cropped images to `output_dir/images/`, masks to
`output_dir/masks/`, and (optionally) overlays to the directory specified via
`--overlay-dir`. A manifest with aggregated tile statistics can be written via
`--metadata`.

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
  --class-mapping dacl10k_label_mapping.json \
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
