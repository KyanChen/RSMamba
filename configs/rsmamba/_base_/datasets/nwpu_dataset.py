# dataset settings
dataset_type = 'RSClsDataset'

data_preprocessor = dict(
    num_classes=45,
    # RGB format normalization parameters
    mean=[93.847266, 97.20128819, 87.66547902],
    std=[37.10216765, 34.61650149, 33.70191772],
    # convert image from BGR to RGB
    to_rgb=True,
)
img_size = 224
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=img_size,
        crop_ratio_range=(0.4, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file='train.txt',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file='val.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(
    type='SingleLabelMetric',
    num_classes=45,
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator
