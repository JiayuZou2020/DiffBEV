import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from functools import reduce
from operator import mul
from ..builder import NECKS

# generate grids in BEV
def _make_grid(resolution, extents):
    # Create a grid of cooridinates in the birds-eye-view
    x1, z1, x2, z2 = extents
    zz, xx = torch.meshgrid(torch.arange(z1, z2, resolution), torch.arange(x1, x2, resolution))

    return torch.stack([xx, zz], dim=-1)


class Resampler(nn.Module):
    def __init__(self, resolution, extents):
        super().__init__()

        # Store z positions of the near and far planes
        # extents[1]:zmin,extents[3]:zmax
        self.near = extents[1]
        self.far = extents[3]

        # Make a grid in the x-z plane
        self.grid = _make_grid(resolution, extents)

    def forward(self, features, calib):
        # Copy grid to the correct device
        self.grid = self.grid.to(features)

        # We ignore the image v-coordinate, and assume the world Y-coordinate
        # calib shape:[bs,3,3]-->[bs,2,3]-->[bs,2,2]-->[bs,1,1,2,2]
        calib = calib[:, [0, 2]][..., [0, 2]].view(-1, 1, 1, 2, 2)

        # Transform grid center locations into image u-coordinates
        cam_coords = torch.matmul(calib, self.grid.unsqueeze(-1)).squeeze(-1)

        # Apply perspective projection and normalize
        ucoords = cam_coords[..., 0] / cam_coords[..., 1]
        ucoords = ucoords / features.size(-1) * 2 - 1

        # Normalize z coordinates
        zcoords = (cam_coords[..., 1] - self.near) / (self.far - self.near) * 2 - 1

        # Resample 3D feature map
        # how to stack ucoords and zcoords,why clamp to [-1,1,1,1]
        grid_coords = torch.stack([ucoords, zcoords], -1).clamp(-1.1, 1.1)
        return F.grid_sample(features, grid_coords)


class DenseTransformer(nn.Module):
    def __init__(self, in_channels, channels, resolution, grid_extents,
                ymin, ymax, focal_length, groups=1):
        super().__init__()

        # Initial convolution to reduce feature dimensions
        self.conv = nn.Conv2d(in_channels, channels, 1)
        self.bn = nn.GroupNorm(16, channels)

        # Resampler transforms perspective features to BEV
        self.resampler = Resampler(resolution, grid_extents)

        # Compute input height based on region of image covered by grid
        self.zmin, zmax = grid_extents[1], grid_extents[3]
        self.in_height = math.ceil(focal_length * (ymax - ymin) / self.zmin)
        # self.ymid = 1
        self.ymid = (ymin + ymax) / 2

        # Compute number of output cells required
        self.out_depth = math.ceil((zmax - self.zmin) / resolution)

        # Dense layer which maps UV features to UZ
        self.fc = nn.Conv1d(
            channels * self.in_height, channels * self.out_depth, 1, groups=groups
        )
        
        self.out_channels = channels

    def forward(self, features, calib, *args):
        # Crop feature maps to a fixed input height
        features = torch.stack([self._crop_feature_map(fmap, cal)
                                for fmap, cal in zip(features, calib)])

        # Reduce feature dimension to minimize memory usage
        features = F.relu(self.bn(self.conv(features)))

        # Flatten height and channel dimensions
        B, C, _, W = features.shape
        flat_feats = features.flatten(1, 2)
        bev_feats = self.fc(flat_feats).view(B, C, -1, W)
        # Resample to orthographic grid
        return self.resampler(bev_feats, calib)

    def _crop_feature_map(self, fmap, calib):
        # Compute upper and lower bounds of visible region
        focal_length, img_offset = calib[1, 1:]
        vmid = self.ymid * focal_length / self.zmin + img_offset
        vmin = math.floor(vmid - self.in_height / 2)
        vmax = math.floor(vmid + self.in_height / 2)

        # Pad or crop input tensor to match dimensions
        return F.pad(fmap, [0, 0, -vmin, vmax - fmap.shape[-2]])


@NECKS.register_module()
class TransformerPyramid(BaseModule):
    def __init__(self, in_channels=256, channels=64, resolution= 0.25 * reduce(mul, [1, 2]),
                extents=[-25.0, 1.0, 25.0, 50.0], ymin=-2, ymax=4, focal_length=630.0):
        super().__init__()
        self.transformers = nn.ModuleList()
        for i in range(5):
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3)

            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3])
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]
            # Build transformers
            tfm = DenseTransformer(in_channels, channels, resolution,
                                subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)

    def forward(self, feature_maps, calib):
        bev_feats = list()
        # scale = 8,16,32,64,128
        # calib.shape = [bs,3,3]
        for i, fmap in enumerate(feature_maps):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2 ** i
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale

            # Apply orthographic transformation to each feature map separately
            bev_feats.append(self.transformers[i](fmap, calib_downsamp))
        return torch.cat(bev_feats[::-1], dim=-2)  # shape: [bs,64,98,100]

@NECKS.register_module()
class DepthTransformerPyramid(BaseModule):
    def __init__(self, in_channels=256, channels=64, resolution= 0.25 * reduce(mul, [1, 2]),
                extents=[-25.0, 1.0, 25.0, 50.0], ymin=-2, ymax=4, focal_length=630.0,
                outdepth=False,depthsup=False):
        super().__init__()
        self.transformers = nn.ModuleList()
        for i in range(5):
            # Scaled focal length for each transformer
            focal = focal_length / pow(2, i + 3)

            # Compute grid bounds for each transformer
            zmax = min(math.floor(focal * 2) * resolution, extents[3])
            zmin = math.floor(focal) * resolution if i < 4 else extents[1]
            subset_extents = [extents[0], zmin, extents[2], zmax]
            # Build transformers
            tfm = DenseTransformer(in_channels, channels, resolution,
                                subset_extents, ymin, ymax, focal)
            self.transformers.append(tfm)
        self.D = 49
        self.C = 64
        self.depthnet = nn.Conv2d(256, self.D + self.C, kernel_size=1, padding=0)
        self.depth_output = nn.Conv2d(self.D, 1, kernel_size=1, padding=0)
        self.outdepth = outdepth
        self.depthsup = depthsup
        
    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def forward(self, feature_maps, calib):
        # depthnet
        tmp_feature = feature_maps[3][:,:,:16,:16]   # feature.shape:bs,768,18,25
        tmp_feature = F.interpolate(tmp_feature,size=(18,25))
        tmp_feature = self.depthnet(tmp_feature)   # feature.shape:bs,113,18,25
        depth = self.get_depth_dist(tmp_feature[:, :self.D])     # depth.shape: bs,49,18,25
        depth_logit = tmp_feature[:, :self.D]
        
        bev_feats = list()
        for i, fmap in enumerate(feature_maps):
            # Scale calibration matrix to account for downsampling
            scale = 8 * 2 ** i
            calib_downsamp = calib.clone()
            calib_downsamp[:, :2] = calib[:, :2] / scale
            bev_feats.append(self.transformers[i](fmap, calib_downsamp))
        feature = torch.cat(bev_feats[::-1], dim=-2)
        if self.depthsup and self.outdepth:
            return torch.cat((feature, self.depth_output(F.interpolate(depth,size=(98,100)))),dim=1), depth_logit
        elif self.depthsup and not self.outdepth:
            return feature, depth_logit
        elif self.outdepth and not self.depthsup:
            return torch.cat((feature, self.depth_output(F.interpolate(depth,size=(98,100)))),dim=1)
        else:
            return feature