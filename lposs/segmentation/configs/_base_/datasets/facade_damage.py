# Dataset configuration for facade damage segmentation.

_base_ = ["../custom_import.py"]

dataset_type = "FacadeDamageDataset"
data_root = "./data/facade_damage"

classes = (
    "background",
    "DAMAGE",
    "WATER_STAIN",
    "ORNAMENT_INTACT",
    "REPAIRS",
    "TEXT_OR_IMAGES",
)

palette = [
    [0, 0, 0],
    [229, 57, 53],
    [142, 36, 170],
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
    dict(
        type="FacadeAugment",
        brightness=0.2,
        contrast=0.3,
        saturation=0.25,
        hue=0.02,
        gamma=0.15,
        blur_prob=0.2,
        noise_std=0.02,
        perspective_prob=0.2,
        perspective_scale=0.05,
    ),
    dict(
        type="PhotoMetricDistortion",
        brightness_delta=20,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=10,
    ),
    dict(type="RandomRotate", prob=0.5, degree=10, pad_val=0, seg_pad_val=255),
    dict(type="Resize", img_scale=(1024, 1024), ratio_range=(0.8, 1.2), keep_ratio=True),
    dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(
        type="RandomCutOut",
        prob=0.3,
        n_holes=(1, 3),
        cutout_shape=(64, 64),
        seg_fill_in=255,
    ),
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
