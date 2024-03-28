# optimizer
# In ClassyVision, the lr is set to 0.003 for bs4096.
# In this implementation(bs2048), lr = 0.003 / 4096 * (32bs * 64gpus) = 0.0015
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0005, weight_decay=0.05),
)

# learning policy
warmup_epochs = 10
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        eta_min=1e-5,
        by_epoch=True,
        begin=warmup_epochs
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=500, val_interval=20)
val_cfg = dict()
test_cfg = dict()
