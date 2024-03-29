_base_ = [
    '_base_/rsmamba_default_runtime.py',
    '_base_/datasets/uc_bs64_pil_resize_autoaug.py',
    # '_base_/datasets/uc_dataset.py',
    '_base_/schedules/uc_schedule.py',
]

work_dir = 'work_dirs/rsmamba_uc_h'

data_root = '/path_to_data/rsmamba/data/UCMerced_LandUse'
code_root = '/path_to_data/rsmamba/datainfo/UC'

batch_size = 16
train_cfg = dict(max_epochs=500, val_interval=20)


vis_backends = [dict(type='LocalVisBackend'),
                # dict(type='WandbVisBackend', init_kwargs=dict(project='rsmamba', group='UC', name='rsmamba_uc_h'))
                ]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

num_classes = 21
data_preprocessor = dict(
    num_classes=num_classes,
)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='RSMamba',
        arch='h',
        pe_type='learnable',
        path_type='forward_reverse_shuffle_gate',
        cls_position='none',  # 'head', 'tail', 'head_tail', 'middle', 'none'
        out_type='avg_featmap',
        img_size=224,
        patch_size=16,
        drop_rate=0.,
        patch_cfg=dict(stride=8),
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ]),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=320,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)


train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_name='UC',
        data_root=data_root,
        ann_file=code_root+'/train.txt',

    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_name='UC',
        data_root=data_root,
        ann_file=code_root+'/val.txt',
    )
)
test_dataloader = val_dataloader
