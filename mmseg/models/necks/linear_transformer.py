import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import NECKS
import torch.nn.functional as F

@NECKS.register_module()
class TransformerLinear(BaseModule):
    def __init__(self, use_light=False, use_high_res=False, input_width=25, input_height=19, input_dim=768, output_width=100, output_height=98, output_dim=64):
        super(TransformerLinear, self).__init__()
        self.use_light = use_light
        self.use_hight_res = use_high_res
        if not use_high_res:
            self.input_width=input_width
            self.input_height=input_height
        else:
            self.input_width=32
            self.input_height=32
        
        self.input_dim=input_dim
        self.output_width=output_width
        self.output_height=output_height
        self.output_dim=output_dim
        if not self.use_light:
            self.tf = nn.Sequential(
                nn.Linear(self.input_width * self.input_height, int(0.2*(self.output_width * self.output_height))),
                nn.ReLU(),
                nn.Linear(int(0.2*self.output_width * self.output_height), self.output_width * self.output_height),
                nn.ReLU())
        else:
            self.tf = nn.Sequential(
                nn.Linear(self.input_width * self.input_height, int(0.05*self.output_width * self.output_height)),
                nn.ReLU(),
                nn.Linear(int(0.05*self.output_width * self.output_height), self.output_width * self.output_height),
                nn.ReLU())
        self.conv = nn.Sequential(nn.Conv2d(self.input_dim, self.output_dim, 1),
                                nn.ReLU())

    def forward(self, feature_maps, intrinstics):
        feature = feature_maps[3]
        n,c,h,w = feature.shape
        feature = feature.view(n, c, h*w)
        feature = self.tf(feature)
        feature = feature.view(n, c, self.output_height, self.output_width)
        feature = self.conv(feature)
        return feature

@NECKS.register_module()
class DepthTransformerLinear(BaseModule):
    def __init__(self, use_light=False, use_high_res=False, input_width=25, input_height=19, input_dim=768, output_width=100, output_height=98, output_dim=64,
                outdepth=False,depthsup=False):
        super(DepthTransformerLinear, self).__init__()
        self.use_light = use_light
        self.use_hight_res = use_high_res
        if not use_high_res:
            self.input_width=input_width
            self.input_height=input_height
        else:
            self.input_width=32
            self.input_height=32
        
        self.input_dim=input_dim
        self.output_width=output_width
        self.output_height=output_height
        self.output_dim=output_dim
        if not self.use_light:
            self.tf = nn.Sequential(
                nn.Linear(self.input_width * self.input_height, int(0.2*(self.output_width * self.output_height))),
                nn.ReLU(),
                nn.Linear(int(0.2*self.output_width * self.output_height), self.output_width * self.output_height),
                nn.ReLU())
        else:
            self.tf = nn.Sequential(
                nn.Linear(self.input_width * self.input_height, int(0.05*self.output_width * self.output_height)),
                nn.ReLU(),
                nn.Linear(int(0.05*self.output_width * self.output_height), self.output_width * self.output_height),
                nn.ReLU())
        self.conv = nn.Sequential(nn.Conv2d(self.input_dim, self.output_dim, 1),nn.ReLU())
        self.D = 49
        self.C = 64
        self.depthnet = nn.Conv2d(input_dim, self.D + self.C, kernel_size=1, padding=0)
        self.depth_output = nn.Conv2d(self.D, 1, kernel_size=1, padding=0)
        self.outdepth = outdepth
        self.depthsup = depthsup

    def get_depth_dist(self, x):
        return x.softmax(dim=1)
    
    def forward(self, feature_maps, intrinstics):
        # depthnet
        tmp_feature = feature_maps[3][:,:,:18,:25]   # feature.shape:bs,768,18,25
        tmp_feature = self.depthnet(tmp_feature)   # feature.shape:bs,113,18,25
        depth = self.get_depth_dist(tmp_feature[:, :self.D])     # depth.shape: bs,49,18,25
        depth_logit = tmp_feature[:, :self.D]

        feature = feature_maps[3]
        n,c,h,w = feature.shape
        feature = feature.view(n, c, h*w)
        feature = self.tf(feature)
        feature = feature.view(n, c, self.output_height, self.output_width)
        feature = self.conv(feature)
        if self.depthsup and self.outdepth:
            return torch.cat((feature, self.depth_output(F.interpolate(depth,size=(98,100)))),dim=1), depth_logit
        elif self.depthsup and not self.outdepth:
            return feature, depth_logit
        elif self.outdepth and not self.depthsup:
            return torch.cat((feature, self.depth_output(F.interpolate(depth,size=(98,100)))),dim=1)
        else:
            return feature
