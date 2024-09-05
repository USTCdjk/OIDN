# 中国科学技术大学 工程科学学院
# USTC  戴俊康
# 开发时间 ：2023/3/30 21:05


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .common1 import fft2d, fftshift2d, gelu, pixel_shiffle, global_average_pooling2d

class FCALayer(nn.Module):
    def __init__(self, channel, reduction=16, size_psc=128):
        super(FCALayer, self).__init__()
        self.gamma = 0.8
        self.conv = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        absfft1 = fft2d(x, gamma=0.8)
        absfft1 = fftshift2d(absfft1, size_psc=128)
        absfft2 = self.conv(absfft1)
        W = self.relu(absfft2)
        W = self.avg_pool(W)
        W = self.fc1(W)
        W = self.relu(W)
        W = self.fc2(W)
        W = self.sigmoid(W)
        mul = x * W
        return mul


class FCAB(nn.Module):
    def __init__(self, channel, size_psc=128):
        super(FCAB, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.gelu = nn.GELU()
        self.fca = FCALayer(channel, size_psc=size_psc)
        #self.norm1 = nn.InstanceNorm2d(channel)
        #self.norm2 = nn.InstanceNorm2d(channel)
    def forward(self, x):
        conv = self.conv1(x)
        #conv = self.norm1(conv)
        conv = self.gelu(conv)
        conv = self.conv2(conv)
        #conv = self.norm2(conv)
        conv = self.gelu(conv)
        conv = self.fca(conv)
        out = x + conv
        return out

# class ResidualGroup(input, channel, size_psc=128):
#     conv = input
#     n_RCAB = 4
#     for _ in range(n_RCAB):
#         conv = FCAB(conv, channel=channel, size_psc=size_psc)
#     conv = conv + input
#     return conv

class ResidualGroup(nn.Module):
    def __init__(self, channel, size_psc=128):
        super(ResidualGroup, self).__init__()
        self.FCABs = nn.Sequential(
            FCAB(channel= channel,size_psc = size_psc),
            FCAB(channel=channel, size_psc=size_psc),
            FCAB(channel= channel,size_psc = size_psc),
            FCAB(channel= channel,size_psc = size_psc),
        )

    def forward(self, x):
        conv = self.FCABs(x)
        out = x + conv
        return out

class DFCAN16(nn.Module):
    def __init__(self, input_shape=9, scale=2, size_psc=128):
        super(DFCAN16, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_shape, out_channels=64, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.residual_groups = nn.Sequential(
            *[ResidualGroup(channel=64, size_psc=size_psc) for _ in range(4)]
        )
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64*(scale**2), kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale)
        # self.pixel_shuffle = nn.PixelShuffle(scale_factorupscale_factor=scale)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu(x)
        x = self.residual_groups(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.pixel_shuffle(x)
        x = self.conv3(x)
        # x = self.activation(x)
        return x
