# Dataset configuration for facade damage segmentation.

_base_ = ["../custom_import.py"]

dataset_type = "FacadeDamageDataset"
data_root = "./data/facade_damage"

classes = (
    "background",
    "CRACK",
    "SPALLING",
    "DELAMINATION",
    "MISSING_ELEMENT",
    "WATER_STAIN",
    "EFFLORESCENCE",
    "CORROSION",
    "ORNAMENT_INTACT",
    "REPAIRS",
    "TEXT_OR_IMAGES",
)

palette = [
    [0, 0, 0],
    [229, 57, 53],
    [30, 136, 229],
    [67, 160, 71],
    [251, 140, 0],
    [142, 36, 170],
    [253, 216, 53],
    [0, 172, 193],
    [158, 158, 158],
    [78, 158, 158],
    [142, 126, 71],
]

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

meta_keys = [
    "ori_shape",
    "img_shape",
    "pad_shape",
    "scale_factor",
    "flip",
    "flip_direction",
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="FacadeAugment", brightness=0.15, contrast=0.25, blur_prob=0.2),
    dict(type="Resize", img_scale=(1024, 1024), keep_ratio=True),
    dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_semantic_seg"]),
    dict(type="Collect", keys=["img", "gt_semantic_seg"], meta_keys=meta_keys),
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="ToTensor", keys=["gt_semantic_seg"]),
            dict(type="Collect", keys=["img", "gt_semantic_seg"], meta_keys=meta_keys),
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"], meta_keys=meta_keys),
        ],
    ),
]

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/train",
        ann_dir="masks/train",
        classes=classes,
        palette=palette,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/val",
        ann_dir="masks/val",
        classes=classes,
        palette=palette,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/test",
        ann_dir="masks/test",
        classes=classes,
        palette=palette,
        pipeline=test_pipeline,
    ),
)

test_cfg = dict(mode="slide", stride=(256, 256), crop_size=(512, 512))
