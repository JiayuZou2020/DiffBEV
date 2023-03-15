_base_ = [
    '../_base_/models/pyva_swin.py','../_base_/datasets/kittiodometry.py',
    '../_base_/default_runtime.py','../_base_/schedules/schedule_20k.py'
]

backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='new_pyva_BEVSegmentor',
    pretrained="/mnt/cfs/algorithm/junjie.huang/models/swin_tiny_patch4_window7_224.pth",
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(3,),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
        pretrain_style='official',
        output_missing_index_as_none=True),
    transformer = dict(type='v4_Pyva_transformer',size = (32,32),back='swin'),
    decode_head=dict(
        type='PyramidHeadKitti',
        num_classes=1,
        align_corners=True,
        priors=[0.04]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole',output_type='iou',positive_thred=0.5))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False, with_calib_kittiodometry=True, imdecode_backend='pyramid'),
    dict(type='Resize', img_scale=(1024, 1024), resize_gt=False, keep_ratio=False),
    dict(type='RandomFlipKitti', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg', 'calib')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False, with_calib_kittiodometry=True, imdecode_backend='pyramid'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False, resize_gt=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg', 'gt_semantic_seg', 'calib')),
        ])
]

data = dict(samples_per_gpu=4, workers_per_gpu=2,
            train=dict(pipeline=train_pipeline),
            test=dict(pipeline=test_pipeline))
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00018,
    betas=(0.9, 0.999))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
