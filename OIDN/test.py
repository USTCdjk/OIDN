import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import tifffile
from caffe2.python.python_op_test import CustomError
from torch.utils.data import DataLoader
from BioSR_dataset import Dataset

from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import normalized_root_mse as compare_nrmse

from argparse import ArgumentParser
from model_type.DFCAN.model.DFCAN16 import DFCAN16
from model_type.APCAN.model.APCAN_1 import APCAN
from model_type.OIDN.model.OIDN import OIDN
from util.util import img_comp

def test(args, test_loader):
    checkpoint = torch.load(args.chinkpoint_for_test)
    if args.model_name=='APCAN':
       print("Testing model: APCAN")
       model = APCAN().cuda()
       model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model_name=='DFCAN':
       print("Testing model: DFCAN")
       model = DFCAN16().cuda()
       model.load_state_dict(checkpoint['model_state_dict'])
    elif args.model_name == 'OIDN':
       print("Testing model: OIDN")
       model = OIDN().cuda()
       model.load_state_dict(checkpoint['model_state_dict'])
    else:
       raise CustomError(f"Error: the model is not implemented!!!!")

    print("start testing>>>>>>>>>") 
    psnr_val, ssim_val, mse_val, nrmses_val, infertime =evaluation(args, test_loader, model)
    print(f'Eval_PSNR: {psnr_val:.4f}, Eval_SSIM: {ssim_val:.4f}, Eval_mse: {mse_val:.4f}, Eval_nrmses: {nrmses_val:.4f}')
    print(f'time: {infertime:.4f}')

def evaluation(args, test_loader, model):
    model.eval()
    with torch.no_grad():
        eval_loss = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        mse_sum =0.0
        nrmses_sum = 0.0
        for eval_idx,(origin_input,origin_gt, norm_input, norm_gt) in enumerate(test_loader):
            val_input=norm_input.cuda()
            val_gt=norm_gt.cuda()
            st=time.time()
            val_output =model(val_input,1)
            infertime=time.time()-st
            val_output = val_output[0].cpu().detach().numpy().transpose(0, 2, 3, 1)
            val_output = np.clip(val_output,0,1)
            val_gt = val_gt.cpu().numpy().transpose(0, 2, 3, 1)
            for inp, out in zip(val_output, val_gt):
                psnrs, ssims, mses, nrmses = img_comp(inp, out)
                psnr_sum += psnrs
                ssim_sum += ssims
                mse_sum += mses
                nrmses_sum += nrmses
            save_images(val_output, eval_idx,args.results_path)
        psnr_val = psnr_sum /len(test_loader)
        ssim_val = ssim_sum /len(test_loader)
        mse_val = mse_sum /len(test_loader)
        nrmses_val = nrmses_sum /len(test_loader)
     
    return psnr_val, ssim_val, mse_val, nrmses_val,infertime

def save_images(output_image, eval_idx,output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_image=output_image[0,:,:,0]*65535
    output_image=output_image.astype('uint16')
    formatted_number = "{:03d}".format(eval_idx)
    tifffile.imsave(output_path+str(formatted_number)+'.tif', output_image)

