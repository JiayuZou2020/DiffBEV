import os.path as osp
import os

import mmcv
import torch
import numpy as np
from PIL import Image

from mmcv.utils import print_log
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

def visualize_map_mask(map_mask):
    color_map = ['#a6cee3','#303030']
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
class KittiOdometryDataset(CustomDataset):
    # only one class for kitti dataset
    # we should add ',' after 'drivable_area' if we have only one class so that CLASSES is a list
    CLASSES = ('drivable_area',)
    PALETTE = [[120,120,120], [8, 255, 51]]
    def __init__(self, **kwargs):
        super(KittiOdometryDataset, self).__init__(
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
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir for visualization.

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
               the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

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
        """Calculate the evaluate result according to the metric type.

            Args:
                results (list): Testing results of the dataset.
                metric (str | list[str]): Type of evalutate metric, mIoU is in consistent
                    with "Predicting Semantic Map Representations from Images with
                    Pyramid Occupancy Networks. CVPR2020", where per class fp,fn,tp are
                    calculated on the hold dataset first. mIOUv1 calculates the per
                    class iou in each image first and average the result between the
                    valid images (i.e. for class c, there is positive sample point in
                    this image). mIOUv2 calculates the per image iou first and average
                    the result between all images.
                logger (logging.Logger | None | str): Logger used for printing
                    related information during evaluation. Default: None.

            Returns:
                tuple: (result_files, tmp_dir), result_files is a list containing
                   the image paths, tmp_dir is the temporal directory created
                    for saving json/png files when img_prefix is not specified.
            """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mIoUv1', 'mIoUv2','mAP']
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
            elif met == 'mAP':
                ious = tp.sum(0).float()/(tp.sum(0)+fp.sum(0)).float()
                print_log('\nper class results (iou):', logger)
                for cid in range(len(self.CLASSES)):
                    print_log('%.04f:%s tp:%d fp:%d' % (ious[cid], self.CLASSES[cid], tp.sum(0)[cid],fp.sum(0)[cid]), logger)
                print_log('%s: %.04f' % (met, ious.mean()), logger)
            else:
                assert False, 'nuknown metric type %s'%metric
