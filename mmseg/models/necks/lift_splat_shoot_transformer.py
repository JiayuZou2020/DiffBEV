import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import NECKS

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])
    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1
        val = gradx[back]
        return val, None, None

@NECKS.register_module()
class TransformerLiftSplatShoot(BaseModule):
    def __init__(self, use_high_res=False, downsample=32,  in_channels=768, bev_feature_channels=64, ogfH=600, ogfW=800, outdepth=False, depthsup=False,
                grid_conf=dict(dbound=[1, 50, 1], xbound=[-25,25,0.5], zbound=[1, 50, 0.5], ybound=[-10,10,20])):
        super(TransformerLiftSplatShoot, self).__init__()
        self.use_high_res = use_high_res
        self.grid_conf = grid_conf
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                self.grid_conf['ybound'],
                                self.grid_conf['zbound'],)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)
        self.downsample = downsample
        if not self.use_high_res:
            self.ogfH = ogfH
            self.ogfW = ogfW
        else:
            self.ogfH = 1024
            self.ogfW = 1024
        self.frustum = self.create_frustum()
        self.D ,_,_,_ = self.frustum.shape
        self.C = bev_feature_channels
        # by default, self.C = 64, self.D = 49
        self.depthnet = nn.Conv2d(in_channels, self.D + self.C, kernel_size=1, padding=0)
        self.use_quickcumsum = True
        self.outdepth = outdepth
        self.depthsup = depthsup

    def create_frustum(self):
        fH, fW = self.ogfH//self.downsample, self.ogfW//self.downsample
        depth_samples = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        num_depth, _, _ = depth_samples.shape
        x_samples = torch.linspace(0, self.ogfW - 1, fW, dtype=torch.float).view(1,1,fW).expand(num_depth,fH,fW)
        y_samples = torch.linspace(0, self.ogfH - 1, fH, dtype=torch.float).view(1,fH,1).expand(num_depth,fH,fW)

        # D x H x W x 3
        frustum = torch.stack((x_samples,y_samples,depth_samples), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrinstics):
        B = intrinstics.shape[0]
        D,H,W,C = self.frustum.shape
        points = self.frustum.view(1,D,H,W,-1).expand(B,D,H,W,C)
        points = torch.cat([points[:,:,:,:,:2]* points[:,:,:,:,2:3],
                            points[:,:,:,:,2:3]],4)
        combine = torch.inverse(intrinstics)
        points = combine.view(B,1,1,1,3,3).matmul(points.unsqueeze(-1)).squeeze(-1).view(B,1,D,H,W,-1)
        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)
        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                        device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B) \
                + geom_feats[:, 1] * (nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[1], nx[2], nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 1], geom_feats[:, 2], geom_feats[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def forward(self, feature_maps, intrinstics):
        # feature_maps[3].shape: bs,768,19,25
        geom = self.get_geometry(intrinstics)  # geom.shape: bs,1,49,18,25,3
        b,_,_,h,w,_  = geom.shape
        if self.downsample==32:
            feature = feature_maps[3][:,:,:h,:w]   # feature.shape:bs,768,18,25
        elif self.downsample==16:
            feature = feature_maps[0][:, :, :h, :w]
        else:
            assert False
        feature = self.depthnet(feature)   # feature.shape:bs,113,18,25
        depth = self.get_depth_dist(feature[:, :self.D])     # depth.shape: bs,49,18,25
        depth_logit = feature[:, :self.D]  # depth_logit.shape: bs,49,18,25
        new_feature = depth.unsqueeze(1) * feature[:, self.D:(self.D + self.C)].unsqueeze(2)  # new_feature.shape: bs,64,49,18,25
        feature = new_feature.view(b, 1, self.C, self.D, h, w)  # feature.shape: bs,1,64,49,18,25
        feature = feature.permute(0, 1, 3, 4, 5, 2)  # feature.shape: bs,1,49,18,25,64
        if self.depthsup and self.outdepth:
            return self.voxel_pooling(geom, torch.cat((feature, depth[:,None,...,None]), dim=-1)), depth_logit
        elif self.depthsup and not self.outdepth:
            return self.voxel_pooling(geom, feature), depth_logit
        elif self.outdepth and not self.depthsup:
            return self.voxel_pooling(geom, torch.cat((feature, depth[:,None,...,None]), dim=-1))
        return self.voxel_pooling(geom, feature)  # shape: [8, 64, 98, 100]