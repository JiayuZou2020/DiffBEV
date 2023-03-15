# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import mmcv
from mmseg.apis import inference_segmentor, init_segmentor

def test_test_time_augmentation_on_cpu():
    config_file = '../configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    print('Loading configs...')
    config = mmcv.Config.fromfile(config_file)
    print('Finish loading configs...')
    # Remove pretrain model download for testing
    config.model.pretrained = None
    # Replace SyncBN with BN to inference on CPU
    norm_cfg = dict(type='BN', requires_grad=True)
    config.model.backbone.norm_cfg = norm_cfg
    config.model.decode_head.norm_cfg = norm_cfg
    config.model.auxiliary_head.norm_cfg = norm_cfg

    # Enable test time augmentation
    config.data.test.pipeline[1].flip = True

    checkpoint_file = None
    print('Loading model...')
    model = init_segmentor(config, checkpoint_file, device='cpu')
    print('Finish evaluation ...')
    img = mmcv.imread(
        osp.join(osp.dirname(__file__), './data/color.jpg'), 'color')
    result = inference_segmentor(model, img)
    # result is a list, len(result) = 1,result[0].shape = (288, 512)
    assert result[0].shape == (288, 512)

