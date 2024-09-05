import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import tifffile
from caffe2.python.python_op_test import CustomError
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, MultiStepLR, ExponentialLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from BioSR_dataset import Dataset
from model_type.DFCAN.model.DFCAN16 import DFCAN16
from model_type.APCAN.model.APCAN_1 import APCAN
from model_type.OIDN.model.OIDN import OIDN
from util.util import img_comp
from loss.SSIM_loss import SSIM
from loss.Spectrum_Loss import SpectrumLoss
from util.loss import CharbonnierLoss
import matplotlib.pyplot as plt



def train(args, train_loader, eval_loader):

    # 初始化模型、损失函数和优化器
    if args.model_name=='APCAN':
       print("Training model: APCAN")
       model = APCAN().cuda()
    elif args.model_name=='DFCAN':
       print("Training model: DFCAN")
       model = DFCAN16().cuda()
    elif args.model_name=='OIDN':
       print("Training model: OIDN")
       model = OIDN().cuda()
    else:
       raise CustomError(f"Error: the model is not implemented!!!!")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 设置 TensorBoard
    writer = SummaryWriter()
    # 设置断点继续训练的参数
    start_epoch = 0
    L1_loss = nn.L1Loss()
    ssim_loss = SSIM()
    spectrum_Loss=SpectrumLoss()
    if os.path.exists(args.resume):
       checkpoint = torch.load(args.checkpoint_path)
       model.load_state_dict(checkpoint['model_state_dict'])
       optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
       start_epoch = checkpoint['epoch'] + 1
       print(f"Loaded checkpoint from epoch {start_epoch - 1}")

    # 设置学习率调整策略
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    print("start training>>>>>>>>>") 
    for epoch in range(start_epoch, args.nEpochs):
        running_loss = 0.0
        for i, (origin_inputs, origin_gt, norm_inputs, norm_gt) in enumerate(train_loader):
            input_images = norm_inputs.cuda()
            gt_images = norm_gt.cuda()

            model.train()
            output, psf, loss_psf = model(input_images, args.batch_size)

            loss = args.weight*L1_loss(output, gt_images)+(1-args.weight)*torch.abs(1-ssim_loss(output, gt_images))+ (1e-3) * loss_psf + 0.01 * CharbonnierLoss(gt_images, output)
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i+1) % args.test_frequency == 0:
           
                with torch.no_grad():
                     model.eval()
                     psnr_val, ssim_val, mse_val, nrmses_val=evaluation(eval_loader, model)
                     writer.add_scalar('psnr_val', psnr_val, epoch * len(train_loader) + i)
                     writer.add_scalar('ssim_val', ssim_val, epoch * len(train_loader) + i)
                     writer.add_scalar('mse_val', ssim_val, epoch * len(train_loader) + i)
                     writer.add_scalar('nrmses_val', ssim_val, epoch * len(train_loader) + i)
                writer.add_scalar('train_loss', running_loss / args.test_frequency, epoch)
                print(f'Epoch [{epoch + 1}/{args.nEpochs}], Step [{i + 1}/{len(train_loader)}], '
                      f'Eval Loss: {running_loss / args.test_frequency:.4f},'
                      f'Eval_PSNR: {psnr_val:.4f}, Eval_SSIM: {ssim_val:.4f}, Eval_mse: {mse_val:.4f}, Eval_nrmses: {nrmses_val:.4f}')
                f=open("test_1.txt","a+")
                f.write("Epoch: %d, PSNR: %.3f, SSIM---%f, MSE---%f, NRMSES---%f---"%(epoch, float(np.array(psnr_val)), float(np.array(ssim_val)), float(np.array(mse_val)),float(np.array(nrmses_val)))+"\n")
                f.close()
                running_loss = 0.0
        scheduler.step()
        # 保存断点
        if (epoch+1)%args.model_save_frequency==0:
           if not os.path.exists(args.checkpoint_path):
              os.makedirs(args.checkpoint_path)
           filename='chinkpoint' +  '_epoch' + str(epoch + 1) + '.pth.tar'
           save_path=os.path.join(args.checkpoint_path, filename)
           torch.save({
           'epoch': epoch,
           'model_state_dict': model.state_dict(),
           'optimizer_state_dict': optimizer.state_dict(),
           }, save_path)
           print(f'Saved checkpoint for epoch {epoch}')
    writer.close()
            

def evaluation(eval_loader, model):
    model.eval()
    eval_loss = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    mse_sum =0.0
    nrmses_sum = 0.0
    for eval_idx,(origin_input,origin_gt, norm_input, norm_gt) in enumerate(eval_loader):
        val_input=norm_input.cuda()
        val_gt=norm_gt.cuda()
        val_output=model(val_input,1)
        val_output = val_output[0].cpu().numpy().transpose(0, 2, 3, 1)
        val_gt = val_gt.cpu().numpy().transpose(0, 2, 3, 1)
        val_output = np.clip(val_output,0,1)
        for inp, out in zip(val_output, val_gt):
            psnrs, ssims, mses, nrmses = img_comp(out, inp)
            psnr_sum += psnrs
            ssim_sum += ssims
            mse_sum += mses
            nrmses_sum += nrmses

    psnr_val = psnr_sum /len(eval_loader) 
    ssim_val = ssim_sum /len(eval_loader) 
    mse_val = mse_sum /len(eval_loader) 
    nrmses_val = nrmses_sum /len(eval_loader) 
        
    return psnr_val, ssim_val, mse_val, nrmses_val


