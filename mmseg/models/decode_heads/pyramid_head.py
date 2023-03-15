import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABCMeta
from mmcv.runner import BaseModule, force_fp32
from ..builder import HEADS
from ..losses import iou

def prior_uncertainty_loss(x, mask, priors):
    # priors shape: [14]-->[1,14,1,1]-->[bs,14,196,200]
    priors = x.new(priors).view(1, -1, 1, 1).expand_as(x)
    # F.binary_cross_entropy_with_logits(x, priors, reduce=False) return a tensor with the shape of x, i.e. [bs,14,196,200]
    xent = F.binary_cross_entropy_with_logits(x, priors, reduce=False)
    return (xent * (~mask).float().unsqueeze(1)).mean()

def balanced_binary_cross_entropy(logits, labels, mask, weights):
    weights = (logits.new(weights).view(-1, 1, 1) - 1) * labels.float() + 1.
    weights = weights * mask.unsqueeze(1).float()
    return F.binary_cross_entropy_with_logits(logits, labels.float(), weights)

class OccupancyCriterion(nn.Module):

    def __init__(self, priors=[0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 0.00189, 0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176],
                xent_weight=1., uncert_weight=0.001,
                weight_mode='sqrt_inverse'):
        super().__init__()

        self.xent_weight = xent_weight
        self.uncert_weight = uncert_weight

        self.priors = torch.tensor(priors)

        if weight_mode == 'inverse':
            self.class_weights = 1 / self.priors
        elif weight_mode == 'sqrt_inverse':
            self.class_weights = torch.sqrt(1 / self.priors)
        elif weight_mode == 'equal':
            self.class_weights = torch.ones_like(self.priors)
        else:
            raise ValueError('Unknown weight mode option: ' + weight_mode)

    def forward(self, logits, labels, mask, *args):
        self.class_weights = self.class_weights.to(logits)
        bce_loss = balanced_binary_cross_entropy(
            logits, labels, mask, self.class_weights)
        self.priors = self.priors.to(logits)
        uncert_loss = prior_uncertainty_loss(logits, mask, self.priors)

        return bce_loss * self.xent_weight + uncert_loss * self.uncert_weight


class LinearClassifier(nn.Conv2d):

    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes, 1)

    def initialise(self, prior):
        prior = torch.tensor(prior)
        self.weight.data.zero_()
        self.bias.data.copy_(torch.log(prior / (1 - prior)))


class TopdownNetwork(nn.Sequential):

    def __init__(self, in_channels, channels, layers=[6, 1, 1],
                strides=[1, 2, 2], blocktype='basic'):
        modules = list()
        self.downsample = 1
        for nblocks, stride in zip(layers, strides):
            # Add a new residual layer
            module = ResNetLayer(
                in_channels, channels, nblocks, 1 / stride, blocktype=blocktype)
            modules.append(module)

            # Halve the number of channels at each layer
            in_channels = module.out_channels
            channels = channels // 2
            self.downsample *= stride

        self.out_channels = in_channels

        super().__init__(*modules)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    # Fractional strides correspond to transpose convolution
    if stride < 1:
        stride = int(round(1 / stride))
        kernel_size = stride + 2
        padding = int((dilation * (kernel_size - 1) - stride + 1) / 2)
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size, stride, padding,
            output_padding=0, dilation=dilation, bias=False)
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=int(stride),
                    dilation=dilation, padding=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    # Fractional strides correspond to transpose convolution
    if int(1 / stride) > 1:
        stride = int(1 / stride)
        return nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=stride, stride=stride, bias=False)

    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=int(stride), bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.GroupNorm(16, planes)

        self.conv2 = conv3x3(planes, planes, 1, dilation)
        self.bn2 = nn.GroupNorm(16, planes)

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), nn.GroupNorm(16, planes))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out, inplace=True)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3(planes, planes, stride, dilation)
        self.bn2 = nn.GroupNorm(16, planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.GroupNorm(16, planes * self.expansion)

        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, stride),
                nn.GroupNorm(16, planes * self.expansion))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out

class ResNetLayer(nn.Sequential):

    def __init__(self, in_channels, channels, num_blocks, stride=1,
                dilation=1, blocktype='bottleneck'):

        # Get block type
        if blocktype == 'basic':
            block = BasicBlock
        elif blocktype == 'bottleneck':
            block = Bottleneck
        else:
            raise Exception("Unknown residual block type: " + str(blocktype))

        # Construct layers
        layers = [block(in_channels, channels, stride, dilation)]
        for _ in range(1, num_blocks):
            layers.append(block(channels * block.expansion, channels, 1, dilation))

        self.in_channels = in_channels
        self.out_channels = channels * block.expansion

        super(ResNetLayer, self).__init__(*layers)


@HEADS.register_module()
class PyramidHead(BaseModule, metaclass=ABCMeta):
    def __init__(self,num_classes,align_corners=True,in_channels=64, channels=128, layers=[4,4], strides=[1,2], **kwargs):
        super(PyramidHead, self).__init__(**kwargs)
        self.topdown = TopdownNetwork(blocktype='bottleneck',in_channels=in_channels, channels=channels, layers=layers, strides=strides)
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.classifier = LinearClassifier(self.topdown.out_channels, self.num_classes)

        self.classifier.initialise([0.44679, 0.02407, 0.14491, 0.02994, 0.02086, 0.00477, 0.00156, 0.00189,
                                    0.00084, 0.00119, 0.00019, 0.00012, 0.00031, 0.00176])
        self.criterion = OccupancyCriterion()
        
    def forward(self, inputs):
        td_feats = self.topdown(inputs)
        logits = self.classifier(td_feats)
        return logits

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        return self.forward(inputs)

    @force_fp32(apply_to=('seg_logit', ))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        seg_label = seg_label.squeeze(1).bool()
        loss = dict()
        loss['acc_seg'] = iou(seg_logit.detach().sigmoid()>0.5, seg_label[:,:-1,...], seg_label[:,-1,...])
        loss['loss_seg'] = self.criterion(seg_logit,seg_label[:,:-1,...],seg_label[:,-1,...])
        return loss
