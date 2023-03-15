from PIL import Image
import numpy as np

def colorize_mask(mask, palette):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))
def get_palette(category):
    if category == 'kitti_object':
        return kitti_object_palette
    elif category == 'argoverse':
        return argoverse_palette
    elif category == 'nuscenes':
        return nuscenes_palette
def get_class_names(category):
    if category == 'kitti_object':
        return kitti_object_class
    elif category == 'argoverse':
        return argoverse_class
    elif category == 'nuscenes':
        return nuscenes_class
 
kitti_object_palette = [ 
  255,  255,  255, # 0 bg
  238,  229,  102, # 1 vehicle
]
argoverse_palette = [255,  255,  255, # 0 bg
           190, 153, 153, # 1 drivable
            250, 170, 30, # 2 vehicle
           220, 220, 0, # 3 pedestrian
           107, 142, 35, # 4 large vehicle
           102, 102, 156, # 5 bicycle
           152, 251, 152, # 6 bus
           119, 11, 32, # 7 trailer
           244, 35, 232, # 8 motorcycle
          ]
nuscenes_palette =[ 255,  255,  255, # 0 background
  238,  229,  102,# 1 drivable
  250, 150,  50,# 2 ped crossing
  124,  99 , 34, # 3 walkway
  193 , 127,  15,# 4 carpark
  225,  96  ,18, # 5 car
  220  ,147 , 77, # 6 truck
  99 , 83  , 3, # 7 bus
  116 , 116 , 138,  # 8 trailer
  200  ,226 , 37, # 9 construct vehicle
  225 , 184 , 161, # 10 pedestrian
  142 , 172  ,248, # 11 motorcycle
  153 , 112 , 146, # 12 bicycle
  38  ,112 , 254, # 13 traffic zone
  229 , 30  ,141, # 14 barrier
]

kitti_object_class = ['background','vehicle']
argoverse_class = ['background','drivable','vehicle','pedestrian','large vehicle',
                   'bicycle','bus','trailer','motorcycle']
nuscenes_class = ['background','drivable','ped crossing','walkway','carpark',
                  'car','truck','bus','trailer','construct vehicle','pedestrian',
                  'motorcycle','bicycle','traffic zone','barrier']