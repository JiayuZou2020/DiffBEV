import numpy as np
from nuscenes.nuscenes import NuScenes
import cv2
nusc = NuScenes(version='v1.0-trainval', dataroot='Your_path/nuscenes', verbose=True)
camlist = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK']
save_path = 'Your_path/nuscenes/ann_bev_dir/depth/'
for sample in tqdm(nusc.sample):
    for cam in camlist:
        points, coloring, im = nusc.render_pointcloud_in_image(sample['token'], pointsensor_channel='LIDAR_TOP', camera_channel=cam)
        cam_token = sample['data'][cam]
        sparse_depth_img = np.zeros((im._size[1], im._size[0]), np.int8)
        sparse_depth_img[points[1].astype(np.int), points[0].astype(np.int)] = coloring.astype(np.int)
        cv2.imwrite(save_path+'%s.png' % cam_token, sparse_depth_img)
