configs/_base_/datasets/facade_damage.py
# Dataset configuration for facade damage segmentation.

_base_ = ["../custom_import.py"]

dataset_type = "FacadeDamageDataset"
data_root = "./data/facade_damage"

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
    dict(type="Collect", keys=["img", "gt_semantic_seg"], meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'scale_factor'])
]

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="MultiScaleFlipAug", img_scale=(1024, 1024), flip=False,
         transforms=[
             dict(type="Resize", keep_ratio=True),
             dict(type="Normalize", **img_norm_cfg),
             dict(type="ImageToTensor", keys=["img"]),
             dict(type="ToTensor", keys=["gt_semantic_seg"]),
             dict(type="Collect", keys=["img", "gt_semantic_seg"], meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'scale_factor'])
         ])
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="MultiScaleFlipAug", img_scale=(1024, 1024), flip=False,
         transforms=[
             dict(type="Resize", keep_ratio=True),
             dict(type="Normalize", **img_norm_cfg),
             dict(type="ImageToTensor", keys=["img"]),
             dict(type="Collect", keys=["img"], meta_keys=['ori_shape', 'img_shape', 'pad_shape', 'scale_factor'])
         ])
]

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/train",
        ann_dir="masks/train",
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/val",
        ann_dir="masks/val",
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images/test",
        ann_dir="masks/test",
        pipeline=test_pipeline,
    ),
)

test_cfg = dict(mode="slide", stride=(256, 256), crop_size=(512, 512))
