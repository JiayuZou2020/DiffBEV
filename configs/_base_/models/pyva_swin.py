# model settings
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='new_pyva_BEVSegmentor',
    # enter your path of the swin transformer tiny weight
    pretrained="YOUR_PATH/swin_tiny_patch4_window7_224.pth",
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
        type='PyramidHead',
        num_classes=14,
        align_corners=True),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole',output_type='iou',positive_thred=0.5))
