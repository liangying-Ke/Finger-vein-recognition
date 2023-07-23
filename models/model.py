import torch
import torch.nn as nn
import torch.nn.functional as F
from .dbb import conv_bn, conv_bn_relu
from .coordatt import CoordAtt
from .module import *


def BasicConv3x3(in_channels, out_channels, stride=1):
    block = [
        conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
    ]
    return nn.Sequential(*block)


def DepthWiseConv2d3x3(in_channels, out_channels, stride=1):
    block = [
        conv_bn_relu(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
        conv_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
    ]
    return nn.Sequential(*block)


def MaxPooling(inplanes, filt_size=4):
    block = [
        circular_pad([0, 1, 0 ,1]),  
        nn.MaxPool2d(kernel_size=2, stride=1),
        ApsPool(inplanes, filt_size=filt_size, stride=2, return_poly_indices=False),
    ]
    return nn.Sequential(*block)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, CONV, inplanes, planes, stride=1, filter_size=4):
        super().__init__()
        self.stride = stride

        self.downsample = None
        if self.stride != 1 or inplanes != planes * self.expansion:
            self.downsample = [ApsPool(filt_size=filter_size, stride=self.stride, channels=inplanes, return_poly_indices = False)] if self.stride != 1 else []
            self.downsample += [conv_bn(inplanes, self.expansion * planes, kernel_size=1, stride=1)]
            self.downsample = nn.Sequential(*self.downsample)

        if self.stride > 1:
            self.aps_down = ApsPool(planes, filt_size=filter_size, stride=self.stride, return_poly_indices=True)

        self.block1 = CONV(inplanes, planes)
        self.block2 = CONV(planes, planes)
        self.ca = CoordAtt(planes, planes)
        
    def forward(self, x):
        identity = x

        out = self.block1(x)
        if self.stride > 1:
            out, indices = self.aps_down(out)

        out = self.block2(out)
        out = self.ca(out)
        
        if self.downsample:
            if self.stride > 1:
                identity = self.downsample({'inp': x, 'polyphase_indices': indices})
            else:
                identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class LightWeightedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.eps = 1e-3
        self.stem = nn.Sequential(
            BasicConv3x3(3, 16, stride=1),
            CoordAtt(16, 16),
            MaxPooling(16),
        )
        self.feature = nn.Sequential(
            BasicBlock(BasicConv3x3, 16, 32, stride=2),
            BasicBlock(BasicConv3x3, 32, 64, stride=2),
            BasicBlock(BasicConv3x3, 64, 128, stride=2),
            BasicBlock(BasicConv3x3, 128, 256, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

        self.kernel = nn.Parameter(torch.FloatTensor(256, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        B, C, H, W = x.shape
        
        out = self.stem(x)
        out = self.feature(out).reshape(B, -1)
        out = self.dropout(self.relu(self.bn(out)))

        out_norm = F.normalize(out, dim=1)
        kernel_norm = F.normalize(self.kernel, dim=0)
        cos_theta = torch.mm(out_norm, kernel_norm).clamp(-1+self.eps, 1-self.eps) 

        out = torch.mm(out, self.kernel)
        return cos_theta, out
