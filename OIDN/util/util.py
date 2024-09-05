from __future__ import print_function
import os
import torch
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse


def tensor2im(input_image, imtype=np.uint16):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 65535
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path, quality=100)  # added by Mia (quality)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def prctile_norm(x, min_prc=0, max_prc=100):
    # np.percentile(x, prc)表示求取百分之prc的分位数，np.percentile(x, 0)就是取x中最小值
    # np.percentile(x, 100)就是取x中最大值
    # 这里的归一化就是(x - np.min(x)) / (np.max(x) - np.min(x) + 1e-7)
    y = (x - np.percentile(x, min_prc)) / (np.percentile(x, max_prc) - np.percentile(x, min_prc) + 1e-7)
    return y

def torch_minmax_norm(x):

    y = (x - torch.min(x)) / (torch.max(x) - torch.min(x) + 1e-7)
    y[torch.where(y > 1)] = 1
    y[torch.where(y < 0)] = 0
    return y
    

def max_min_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))






def img_comp(gt, pr, mses=None, nrmses=None, psnrs=None, ssims=None):
    if ssims is None:
        ssims = []
    if psnrs is None:
        psnrs = []
    if nrmses is None:
        nrmses = []
    if mses is None:
        mses = []
    gt, pr = np.squeeze(gt), np.squeeze(pr)
    gt = gt.astype(np.float32)
    pr = pr.astype(np.float32)
    #pr = prctile_norm(pr.astype(np.float32))
    #gt = np.uint8(gt* 255)
    #print(np.shape(pr))
    #pr = np.uint8(prctile_norm(pr) * 255)
    if gt.ndim == 2:
        n = 1
        gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
        pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
    else:
        n = np.size(gt, 0)
    #print(np.shape(gt))
    #print(np.shape(pr))
    for i in range(n):
        mses.append(compare_mse(np.squeeze(gt[i]), prctile_norm(pr[i])))
        nrmses.append(
            compare_nrmse(np.squeeze(gt[i]), np.squeeze(pr[i]), normalization='min-max'))
        psnrs.append(compare_psnr(np.squeeze(gt[i]), np.squeeze(pr[i])))
        ssims.append(compare_ssim(np.squeeze(gt[i]), np.squeeze(pr[i])))

    return np.mean(psnrs), np.mean(ssims), np.mean(mses), np.mean(nrmses)

def img_comp2(gt, pr, mses=None, nrmses=None, psnrs=None, ssims=None):
    if ssims is None:
        ssims = []
    if psnrs is None:
        psnrs = []
    if nrmses is None:
        nrmses = []
    if mses is None:
        mses = []
    gt, pr = np.squeeze(gt), np.squeeze(pr)
    gt = gt.astype(np.float32)
    pr = pr.astype(np.float32)
    if gt.ndim == 2:
        n = 1
        gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
        pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
    else:
        n = np.size(gt, 0)
    for i in range(n):
        mses.append(compare_mse(np.squeeze(gt[i]), prctile_norm(pr[i])))
        nrmses.append(
            compare_nrmse(np.squeeze(gt[i]), np.squeeze(pr[i]), normalization='min-max'))
        psnrs.append(compare_psnr(np.squeeze(gt[i]), np.squeeze(pr[i]),data_range=1.))
        ssims.append(compare_ssim(np.squeeze(gt[i]), np.squeeze(pr[i])))

    return np.mean(psnrs), np.mean(ssims), np.mean(mses), np.mean(nrmses)


'''def img_comp(gt, pr, mses=None, nrmses=None, psnrs=None, ssims=None):
    if ssims is None:
        ssims = []
    if psnrs is None:
        psnrs = []
    if nrmses is None:
        nrmses = []
    if mses is None:
        mses = []
    gt, pr = np.squeeze(gt), np.squeeze(pr)
    gt = gt.astype(np.float32)
    pr = pr.astype(np.float32)
    if gt.ndim == 2:
        n = 1
        gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
        pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
    else:
        n = np.size(gt, 0)
    for i in range(n):
        mses.append(compare_mse(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
        nrmses.append(
            compare_nrmse(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i])), normalization='min-max'))
        psnrs.append(compare_psnr(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))
        ssims.append(compare_ssim(prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))))

    return np.mean(psnrs), np.mean(ssims), np.mean(mses), np.mean(nrmses)'''
