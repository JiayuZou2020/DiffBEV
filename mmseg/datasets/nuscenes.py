import os.path as osp
import os

import mmcv
import torch
import json
import numpy as np
from PIL import Image

from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset
from tqdm import trange

def covert_color(input):
    str1 = input[1:3]
    str2 = input[3:5]
    str3 = input[5:7]
    r = int('0x' + str1, 16)
    g = int('0x' + str2, 16)
    b = int('0x' + str3, 16)
    return (r, g, b)

# visualize_map_mask中的color_map对应的RGB为
# RGB_value = [(166, 206, 227),(31, 120, 180),(178, 223, 138),(51, 160, 44),(251, 154, 153),
#              (227, 26, 28),(253, 191, 111),(255, 127, 0),(202, 178, 214),(106, 61, 154),
#              (126, 119, 46),(0, 255, 0),(0, 0, 255),(0, 255, 255),(48, 48, 48)]
def visualize_map_mask(map_mask):
    color_map = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99',
                 '#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a',
                 '#7e772e','#00ff00','#0000ff','#00ffff','#303030']
    ori_shape = map_mask.shape
    vis = np.zeros((ori_shape[1], ori_shape[2], 3),dtype=np.uint8)
    vis = vis.reshape(-1,3)
    map_mask = map_mask.reshape(ori_shape[0],-1)
    for layer_id in range(map_mask.shape[0]):
        keep = np.where(map_mask[layer_id,:])[0]
        for i in range(3):
            vis[keep, 2-i] = covert_color(color_map[layer_id])[i]
    return vis.reshape(ori_shape[1], ori_shape[2], 3)


@DATASETS.register_module()
class NuscenesDataset(CustomDataset):
    CLASSES = ('drivable_area', 'ped_crossing', 'walkway', 'carpark',
               'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
               'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51]]

    def __init__(self, **kwargs):
        super(NuscenesDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id):
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

            prog_bar.update()

        return result_files

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        assert isinstance(results, list), 'results must be a list'

        imgfile_prefix = osp.join(imgfile_prefix, 'vis')
        if not osp.exists(imgfile_prefix):
            os.makedirs(imgfile_prefix)
        print_log('\n Start formatting the result')

        for id in trange(len(results)):
            pred, gt, img_path = results[id]
            b,c,h,w = pred.shape
            assert pred.shape[0]==1 and gt.shape[0]==1
            pred = pred[0]
            gt = gt[0]
            gt[-1, ...] = np.invert(gt[-1, ...])
            pred = np.concatenate([pred, gt[-1,...][None,...]], axis=0)
            pred_vis = visualize_map_mask(pred)
            gt_vis = visualize_map_mask(gt)
            img = mmcv.imread(img_path, backend='cv2')
            img = mmcv.imresize(img,(int(float(img.shape[1])*h/float(img.shape[0])), h))
            vis = np.concatenate([img, pred_vis[::-1,...], gt_vis[::-1,]], axis=1)
            save_path = osp.join(imgfile_prefix, os.path.basename(img_path))
            mmcv.imwrite(vis, save_path)

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mIoUv1', 'mIoUv2']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        tp = torch.cat([res[0][None, ...] for res in results], dim=0) #N*C
        fp = torch.cat([res[1][None, ...] for res in results], dim=0) #N*C
        fn = torch.cat([res[2][None, ...] for res in results], dim=0) #N*C
        valids = torch.cat([res[3][None,...] for res in results],dim=0) #N*C
        for met in metric:
            if met=='mIoU':
                ious = tp.sum(0).float()/(tp.sum(0)+fp.sum(0)+fn.sum(0)).float()
                print_log('\nper class results (iou):', logger)
                for cid in range(len(self.CLASSES)):
                    print_log('%.04f:%s tp:%d fp:%d fn:%d' % (ious[cid], self.CLASSES[cid], tp.sum(0)[cid],fp.sum(0)[cid],fn.sum(0)[cid]), logger)
                print_log('%s: %.04f' % (met, ious.mean()), logger)
            elif met == 'mIoUv1':
                ious = tp.float() / (tp + fp + fn).float()
                print_log('\nper class results (iou):', logger)
                miou, valid_class = 0, 0
                for cid in range(len(self.CLASSES)):
                    iou_c = ious[:, cid][valids[:, cid]]
                    if iou_c.shape[0] > 0:
                        iou_c = iou_c.mean()
                        miou += iou_c
                        valid_class += 1
                    else:
                        iou_c = -1
                    print_log('%.04f:%s' % (iou_c, self.CLASSES[cid]), logger)
                print_log('%s: %.04f' % (met, miou / valid_class), logger)
            elif met == 'mIoUv2':
                ious = tp.sum(-1).float() / (tp.sum(-1) + fp.sum(-1) + fn.sum(-1)).float()
                print_log('\n%s: %.04f' % (met, ious.mean()), logger)
            else:
                assert False, 'nuknown metric type %s'%metric
