# 中国科学技术大学 工程科学学院
# USTC  戴俊康
# 开发时间 ：2023/3/30 21:05

import torch
from torch import nn
import pytorch_msssim
from models.common import fft2d

device = torch.device("cpu")

def norm(x : torch.Tensor)->torch.Tensor:
    # return ((x-x.min())/(x.max()-x.min()+1e-7)).clip(0 , 1)
    return x/65535.0

def l2_penalty(w):
    return (w**2).sum() / 2

def loss_mse_ssim(y_true, y_pred):
    ssim_para = 1e-1
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    # x = norm(x)
    # y = norm(y)
    MSE_loss = nn.MSELoss()
    MSE_loss = MSE_loss.to(device)
    # SSIMloss = SSIM()
    # SSIMloss = SSIMloss.to(device)

    mse_loss = mse_para * MSE_loss(x, y)
    ssim_loss = ssim_para * (1 - pytorch_msssim.ssim(x, y))
    mse_ssim_loss = mse_loss + ssim_loss

    return mse_ssim_loss

def CharbonnierLoss(y_true, y_pred):
    epsilon = 1e-3
    epsilon2 = epsilon*epsilon

    y_true_fft = fft2d(y_true)
    y_pred_fft = fft2d(y_pred)
    yt_real = y_true_fft.real
    yt_imag = y_true_fft.imag
    yp_real = y_pred_fft.real
    yp_imag = y_pred_fft.imag
    real = torch.sqrt(torch.pow((yt_real - yp_real), 2) + epsilon2)
    imag = torch.sqrt(torch.pow((yt_imag - yp_imag), 2) + epsilon2)
    value = real + imag
    return torch.mean(value)


def loss_mae_mse(y_true, y_pred):
    mae_para = 0.2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = norm(x)
    y = norm(y)

    MAEloss = nn.L1Loss()
    MSEloss = nn.MSELoss()


    mae_loss = mae_para * MAEloss(x, y)
    mse_loss = mse_para * MSEloss(x, y)

    mae_mse_loss = mae_loss + mse_loss

    return mae_mse_loss