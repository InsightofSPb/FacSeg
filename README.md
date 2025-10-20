# FacSeg utilities

## Dataset label summary helper

Use the `tools/dataset_label_summary.py` script to inspect annotation labels in
JSON (e.g., DACL10K) or XML (e.g., Prova) datasets and optionally emit a JSON
template for remapping classes.

### Basic usage

```bash
python tools/dataset_label_summary.py <path-to-annotation-folder>
```

The command recursively searches the folder for `.json` and `.xml` files, prints
all discovered labels, how many annotation files each label appears in, and up
to ten sample filenames per class.

### Useful options

* `--samples-per-class N` &mdash; change how many sample files to show per class.
* `--mapping-template path/to/output.json` &mdash; write an empty mapping template
  (keys are discovered labels, values are left blank for you to fill in).
* `--extensions .json .xml` &mdash; override the set of annotation file extensions
  that will be parsed.

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

If your dataset stores annotations in a different subfolder, point the script
directly to that directory. The samples printed for each label can be used to
manually inspect representative files and verify class mappings.