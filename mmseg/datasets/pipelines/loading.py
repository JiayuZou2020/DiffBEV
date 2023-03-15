# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from cv2 import decomposeProjectionMatrix
import cv2
import mmcv
import numpy as np
import json
import torch
import torchvision
from PIL import Image

from ..builder import PIPELINES
import os
# from tools import heatmap_vis

@PIPELINES.register_module()
class LoadImageFromFile(object):
    """Load an image from file.

    Required keys are "img_prefix" and "img_info" (a dict that must contain the
    key "filename"). Added or updated keys are "filename", "img", "img_shape",
    "ori_shape" (same as `img_shape`), "pad_shape" (same as `img_shape`),
    "scale_factor" (1.0) and "img_norm_cfg" (means=0 and stds=1).

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'cv2'
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 imdecode_backend='cv2'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('img_prefix') is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img_bytes = self.file_client.get(filename)
        # using cv2 to read image, so the image format is H,W,C
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        results['ori_filename'] = results['img_info']['filename']
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32},'
        repr_str += f"color_type='{self.color_type}',"
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str

def decode_binary_labels(labels,nclass):
    bits = torch.pow(2,torch.arange(nclass))
    return (labels & bits.view(-1,1,1))>0

@PIPELINES.register_module()
class LoadAnnotations(object):
    """Load annotations for semantic segmentation.

    Args:
        reduce_zero_label (bool): Whether reduce all label value by 1.
            Usually used for datasets where 0 is background label.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
        imdecode_backend (str): Backend for :func:`mmcv.imdecode`. Default:
            'pillow'
    """

    def __init__(self,
                reduce_zero_label=False,
                file_client_args=dict(backend='disk'),
                imdecode_backend='pillow',
                with_calib=False,
                with_calib_kittiraw=False,
                with_calib_kittiodometry=False,
                with_calib_kittiobject=False):
        self.reduce_zero_label = reduce_zero_label
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.imdecode_backend = imdecode_backend
        self.with_calib = with_calib
        self.with_calib_kittiraw = with_calib_kittiraw
        self.with_calib_kittiodometry = with_calib_kittiodometry
        self.with_calib_kittiobject = with_calib_kittiobject
        # enter your path of calibration matrix of each dataset
        if self.with_calib:
            self.nuscenes = json.load(open('YOUR_PATH/nuscenes/calib.json','r'))
        if self.with_calib_kittiraw:
            self.kittiraw = json.load(open('YOUR_PATH/kitti_raw/calib.json','r'))
        if self.with_calib_kittiodometry:
            self.kittiodometry = json.load(open('YOUR_PATH/kitti_odometry/calib.json','r'))
        if self.with_calib_kittiobject:
            self.kittiobject = json.load(open('YOUR_PATH/kitti_object/calib.json','r'))
    
    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmseg.CustomDataset`.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        if results.get('seg_prefix', None) is not None:
            filename = osp.join(results['seg_prefix'],
                                results['ann_info']['seg_map'])
        else:
            filename = results['ann_info']['seg_map']
        if self.imdecode_backend=='pyramid':
            encoded_labels = torchvision.transforms.functional.to_tensor(Image.open(filename)).long()
            # decode to binary labels,the data type of gt_semantic_seg is bool,i.e. 0 or 1, gt_semantic_seg is numpy array
            if self.with_calib:
                gt_semantic_seg = decode_binary_labels(encoded_labels,15).numpy()
            if self.with_calib_kittiraw or self.with_calib_kittiodometry or self.with_calib_kittiobject:
                # only one class for kitti dataset
                gt_semantic_seg = np.zeros((2,196,200)).astype(np.bool)
                gt_semantic_seg[0,...] = cv2.imread(filename,cv2.IMREAD_GRAYSCALE)
                gt_semantic_seg[0,...] = cv2.flip(gt_semantic_seg[0,...].astype(np.uint8),0).astype(np.bool)
                gt_semantic_seg[-1,...] = cv2.imread("./mask_vis.png",cv2.IMREAD_GRAYSCALE).astype(np.bool)
                gt_semantic_seg[-1,...] = np.invert(gt_semantic_seg[-1,...])
            gt_semantic_seg[-1,...] = np.invert(gt_semantic_seg[-1,...])
        else:
            img_bytes = self.file_client.get(filename)
            gt_semantic_seg = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend=self.imdecode_backend).squeeze().astype(np.uint8)
        # modify if custom classes
        if results.get('label_map', None) is not None:
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg == old_id] = new_id
        # reduce zero_label
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        results['gt_semantic_seg'] = gt_semantic_seg
        results['seg_fields'].append('gt_semantic_seg')
        if self.with_calib:
            token = osp.basename(filename).split('.')[0]
            intrinsics = torch.tensor(self.nuscenes[token])
            intrinsics[0] *= 800 / results['img_shape'][1]
            intrinsics[1] *= 600 /results['img_shape'][0]
            results['calib'] = intrinsics
        if self.with_calib_kittiraw:
            token = osp.basename(filename).split('.')[0]
            intrinsics = torch.tensor(self.kittiraw[token])
            intrinsics[0] *= 1024 / results['img_shape'][1]
            intrinsics[1] *= 1024 /results['img_shape'][0]
            results['calib'] = intrinsics
        if self.with_calib_kittiodometry:
            token = osp.basename(filename).split('.')[0]
            intrinsics = torch.tensor(self.kittiodometry[token])
            intrinsics[0] *= 1024 / results['img_shape'][1]
            intrinsics[1] *= 1024 /results['img_shape'][0]
            results['calib'] = intrinsics
        if self.with_calib_kittiobject:
            token = osp.basename(filename).split('.')[0]
            intrinsics = torch.tensor(self.kittiobject[token])
            intrinsics[0] *= 1024 / results['img_shape'][1]
            intrinsics[1] *= 1024 /results['img_shape'][0]
            results['calib'] = intrinsics
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(reduce_zero_label={self.reduce_zero_label},'
        repr_str += f"imdecode_backend='{self.imdecode_backend}')"
        return repr_str
