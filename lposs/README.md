# LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation

This repository contains the code for the paper Vladan Stojnić, Yannis Kalantidis, Jiří Matas, Giorgos Tolias, ["LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation"](http://arxiv.org/abs/2503.19777), In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

<div align="center">
    
[![arXiv](https://img.shields.io/badge/arXiv-2503.19777-b31b1b.svg)](http://arxiv.org/abs/2503.19777) [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md-dark.svg)](https://huggingface.co/spaces/stojnvla/LPOSS)

</div>

## Demo

The demo of our method is available at [<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png" height=20px> huggingface spaces](https://huggingface.co/spaces/stojnvla/LPOSS).

## Setup

Setup the conda environment:
```
# Create conda environment
conda create -n lposs python=3.9
conda activate lposs
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```
Install MMCV and MMSegmentation:
```
pip install -U openmim
mim install mmengine    
mim install "mmcv-full==1.6.0"
mim install "mmsegmentation==0.27.0"
```
Install additional requirements:
```
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install kornia cupy-cuda11x ftfy omegaconf open_clip_torch==2.26.1 hydra-core wandb
```

## Datasets

We use 8 benchmark datasets: PASCAL VOC20, PASCAL Context59, COCO-Object, PASCAL VOC, PASCAL Context, COCO-Stuff, Cityscapes, and ADE20k.

To run the evaluation, download and set up PASCAL VOC, PASCAL Context, COCO-Stuff164k, Cityscapes, and ADE20k datasets following ["MMSegmentation"](https://mmsegmentation.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html) data preparation document.

COCO-Object dataset uses only object classes from COCO-Stuff164k dataset by collecting instance segmentation annotations. Run the following command to convert instance segmentation annotations to semantic segmentation annotations:

```
python tools/convert_coco.py data/coco_stuff164k/ -o data/coco_stuff164k/
```

## Running

The provided code can be run using follwing commands:

LPOSS:
```
torchrun main_eval.py lposs.yaml --dataset {voc, coco_object, context, context59, coco_stuff, voc20, ade20k, cityscapes} [--measure_boundary]
```

LPOSS+:
```
torchrun main_eval.py lposs_plus.yaml --dataset {voc, coco_object, context, context59, coco_stuff, voc20, ade20k, cityscapes} [--measure_boundary]
```

### Facade damage utilities

The `tools` sub-package contains scripts that streamline quantitative evaluation and interpretability studies for the St. Petersburg facade dataset:

* `python -m lposs.tools.evaluate_facade_models --output-dir ./outputs/facade_eval --finetuned-checkpoint <path/to/finetuned.ckpt>` evaluates both the stock and fine-tuned models on the selected splits, storing:
  * full metric dumps (including per-class IoU, macro F1, aggregated damage IoU, and the new `damageDetectionRecall`, `damageDetectionPrecision`, `damageDetectionF1`, and `damageMislabelRate` that treat any correctly detected damage—regardless of the subtype—as a partial hit),
  * raw and normalised confusion matrices in CSV and PNG form,
  * a consolidated `summary.json` for quick comparisons.

  Pass `--skip-stock` to omit the base model, `--stock-checkpoint` to load an explicit checkpoint for the pre-finetuned weights, and `--splits train val` to evaluate multiple splits in one run.

* `python -m lposs.tools.analyse_facade_representations --output-dir ./outputs/facade_analysis --finetuned-checkpoint <path/to/finetuned.ckpt> --indices 0 12 27` compares internal representations between model variants using Grad-CAM and Integrated Gradients. For each sampled tile it saves:
  * per-model segmentation overlays,
  * per-class attribution heatmaps for the requested methods (defaulting to damage classes),
  * an `analysis_summary.json` capturing prompt configuration, class probabilities, and average attribution intensities.

  Custom text prompts can be explored either inline (`--prompt SPALLING="spalling у колонны"`) or via named JSON/YAML prompt sets supplied through `--prompt-set custom=path/to/prompts.json`. The script re-instantiates each model for every prompt set so that stock and fine-tuned variants can be compared side by side under identical textual conditions.

## Citation

```
@InProceedings{stojnic2025_lposs,
    author    = {Stojni\'c, Vladan and Kalantidis, Yannis and Matas, Ji\v{r}\'i  and Tolias, Giorgos},
    title     = {LPOSS: Label Propagation Over Patches and Pixels for Open-vocabulary Semantic Segmentation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2025}
}
```

## Acknowledgments

This repository is based on ["CLIP-DINOiser: Teaching CLIP a few DINO tricks for Open-Vocabulary Semantic Segmentation"](https://github.com/wysoczanska/clip_dinoiser). Thanks to the authors!
