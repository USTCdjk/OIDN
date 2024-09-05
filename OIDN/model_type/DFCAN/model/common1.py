# 中国科学技术大学 工程科学学院
# USTC  戴俊康
# 开发时间 ：2023/3/30 21:05


import torch
import torch.fft as fft
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F



def gelu(x):         #  gelu 激活函数也可以直接调用下面的
    cdf = 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0))))
    return x * cdf

def fft2d(input, gamma=0.1):    #快速2D傅里叶变换
    # 将输入的tensor维度按照指定顺序进行转换
    # temp = input.permute(0, 3, 1, 2)
    # 对转换后的tensor进行2维FFT变换，并使用0填充
    # fft = torch.fft.fftn(torch.complex(temp, torch.zeros_like(temp)), dim=(-2, -1))
    fft = torch.fft.fftn(torch.complex(input, torch.zeros_like(input)), dim=(-2, -1))
    # 将变换结果取绝对值并进行幂运算，加上一个小的数以防止出现0
    # absfft = torch.pow(torch.abs(fft)+1e-8, gamma)
    output = torch.pow(torch.abs(fft) + 1e-8, gamma)
    # 将结果的维度再次按照指定顺序进行转换
    # output = absfft.permute(0, 2, 3, 1)
    return output



def fft3d(input, gamma=0.1):        #快速3D傅里叶变换
    input = apodize3d(input, napodize=5)
    temp = torch.permute(input, (0, 4, 1, 2, 3))
    fft = torch.fft.fftn(torch.complex(temp, torch.zeros_like(temp)), dim=(-3, -2, -1))
    absfft = torch.pow(torch.abs(fft) + 1e-8, gamma)
    output = torch.permute(absfft, (0, 4, 1, 2, 3))
    return output



def fftshift2d(input, size_psc=128):   #对输入思维张量进行二维傅里叶变换位移操作
    bs, h, w, ch = input.size()
    fs11 = input[:, -h//2:h, -w//2:w, :]
    fs12 = input[:, -h//2:h, 0:w//2, :]
    fs21 = input[:, 0:h//2, -w//2:w, :]
    fs22 = input[:, 0:h//2, 0:w//2, :]
    output = torch.cat([torch.cat([fs11, fs21], dim=1), torch.cat([fs12, fs22], dim=1)], dim=2)
    output = torch.nn.functional.interpolate(output, size=(size_psc, size_psc), mode='bilinear', align_corners=False)
    return output


def fftshift3d(input, size_psc=64):
    bs, h, w, z, ch = input.shape
    fs111 = input[:, -h // 2:h, -w // 2:w, -z // 2 + 1:z, :]
    fs121 = input[:, -h // 2:h, 0:w // 2, -z // 2 + 1:z, :]
    fs211 = input[:, 0:h // 2, -w // 2:w, -z // 2 + 1:z, :]
    fs221 = input[:, 0:h // 2, 0:w // 2, -z // 2 + 1:z, :]
    fs112 = input[:, -h // 2:h, -w // 2:w, 0:z // 2 + 1, :]
    fs122 = input[:, -h // 2:h, 0:w // 2, 0:z // 2 + 1, :]
    fs212 = input[:, 0:h // 2, -w // 2:w, 0:z // 2 + 1, :]
    fs222 = input[:, 0:h // 2, 0:w // 2, 0:z // 2 + 1, :]
    output1 = torch.cat([torch.cat([fs111, fs211], dim=1), torch.cat([fs121, fs221], dim=1)], dim=2)
    output2 = torch.cat([torch.cat([fs112, fs212], dim=1), torch.cat([fs122, fs222], dim=1)], dim=2)
    output0 = torch.cat([output1, output2], dim=3)
    output = []
    for iz in range(z):
        output.append(torch.nn.functional.interpolate(output0[:, :, :, iz, :], size=(size_psc, size_psc), mode='nearest'))
    output = torch.stack(output, dim=3)
    return output


def apodize2d(img, napodize=10):
    bs, ny, nx, ch = img.get_shape().as_list()
    img_apo = img[:, napodize:ny-napodize, :, :]

    imageUp = img[:, 0:napodize, :, :]
    imageDown = img[:, ny-napodize:, :, :]
    diff = (imageDown[:, -1::-1, :, :] - imageUp) / 2
    l = np.arange(napodize)
    fact_raw = 1 - np.sin((l + 0.5) / napodize * np.pi / 2)
    fact = fact_raw.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(torch.float32)
    fact = fact.repeat([bs, 1, nx, ch])
    factor = diff * fact
    imageUp = torch.add(imageUp, factor)
    imageDown = torch.sub(imageDown, factor[:, -1::-1, :, :])
    img_apo = torch.cat([imageUp, img_apo, imageDown], dim=1)

    imageLeft = img_apo[:, :, 0:napodize, :]
    imageRight = img_apo[:, :, nx-napodize:, :]
    img_apo = img_apo[:, :, napodize:nx-napodize, :]
    diff = (imageRight[:, :, -1::-1, :] - imageLeft) / 2
    fact = fact_raw.unsqueeze(0).unsqueeze(1).unsqueeze(-1).to(torch.float32)
    fact = fact.repeat([bs, ny, 1, ch])
    factor = diff * fact
    imageLeft = torch.add(imageLeft, factor)
    imageRight = torch.sub(imageRight, factor[:, :, -1::-1, :])
    img_apo = torch.cat([imageLeft, img_apo, imageRight], dim=2)

    return img_apo


def apodize3d(img, napodize=5):
    bs, ny, nx, nz, ch = img.shape
    img_apo = img[:, napodize:ny-napodize, :, :, :]

    imageUp = img[:, 0:napodize, :, :, :]
    imageDown = img[:, ny-napodize:, :, :, :]
    diff = (imageDown[:, torch.arange(napodize - 1, -1, -1), :, :, :] - imageUp) / 2
    l = torch.arange(napodize, dtype=torch.float32)
    fact_raw = 1 - torch.sin((l + 0.5) / napodize * np.pi / 2)
    fact = fact_raw.view(1, -1, 1, 1, 1).to(img.device)
    fact = fact.repeat(bs, 1, nx, nz, ch)
    factor = diff * fact
    imageUp = imageUp + factor
    imageDown = imageDown - factor[:, torch.arange(napodize - 1, -1, -1), :, :, :]
    img_apo = torch.cat([imageUp, img_apo, imageDown], dim=1)

    imageLeft = img_apo[:, :, 0:napodize, :, :]
    imageRight = img_apo[:, :, nx-napodize:, :, :]
    img_apo = img_apo[:, :, napodize:nx-napodize, :, :]
    diff = (imageRight[:, :, torch.arange(napodize - 1, -1, -1), :, :] - imageLeft) / 2
    fact = fact_raw.view(1, 1, -1, 1, 1).to(img.device)
    fact = fact.repeat(bs, ny, 1, nz, ch)
    factor = diff * fact
    imageLeft = imageLeft + factor
    imageRight = imageRight - factor[:, :, torch.arange(napodize - 1, -1, -1), :, :]
    img_apo = torch.cat([imageLeft, img_apo, imageRight], dim=2)

    return img_apo


def pixel_shiffle(layer_in, scale):       #  定义像素偏移函数
    return torch.nn.functional.pixel_shuffle(layer_in, scale_factor=scale)


def global_average_pooling2d(layer_in):   # 定义 2D 全局平均池化函数
    return torch.nn.functional.adaptive_avg_pool2d(layer_in, (1, 1))
    # return torch.mean(layer_in, dim=(2, 3), keepdim=True)


def global_average_pooling3d(layer_in):     #   定义 3D 全局平均池化函数
    return torch.mean(layer_in, dim=(2, 3, 4), keepdim=True)


def conv_block2d(input, channel_size):
    # 定义 2D 卷积层，并对输入进行卷积操作，输出通道数不变
    conv = nn.Conv2d(channel_size[0], channel_size[0], kernel_size=3, padding=1)(input)
    # 对卷积输出进行 LeakyReLU 激活操作
    conv = nn.LeakyReLU(negative_slope=0.1)(conv)
    # 定义 2D 卷积层，并对上一层的输出进行卷积操作，输出通道数为 channel_size[1]
    conv = nn.Conv2d(channel_size[0], channel_size[1], kernel_size=3, padding=1)(conv)
    # 对卷积输出进行 LeakyReLU 激活操作
    conv = nn.LeakyReLU(negative_slope=0.1)(conv)
    return conv


def conv_block3d(input, channel_size):
    # 定义 3D 卷积层，并对输入进行卷积操作，输出通道数不变
    conv = nn.Conv3d(channel_size[0], channel_size[0], kernel_size=3, padding=1)(input)
    # 对卷积输出进行 LeakyReLU 激活操作
    conv = nn.LeakyReLU(negative_slope=0.1)(conv)
    # 定义 3D 卷积层，并对上一层的输出进行卷积操作，输出通道数为 channel_size[1]
    conv = nn.Conv3d(channel_size[0], channel_size[1], kernel_size=3, padding=1)(conv)
    # 对卷积输出进行 LeakyReLU 激活操作
    conv = nn.LeakyReLU(negative_slope=0.1)(conv)
    return conv
