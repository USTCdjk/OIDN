import torch
import torch.nn as nn
class SpectrumLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(SpectrumLoss, self).__init__()
        self.alpha = alpha

    def forward(self, output, target):
        # 对模型输出和目标进行二维傅里叶变换
        output_fft = torch.fft.fftn(output, dim=(2, 3), norm='ortho')
        target_fft = torch.fft.fftn(target, dim=(2, 3), norm='ortho')
        output_fft_shifted = torch.fft.fftshift(output_fft,dim=(2, 3))
        target_fft_shifted = torch.fft.fftshift(target_fft,dim=(2, 3))
        
        output_fft_shifted_real=torch.real(output_fft_shifted)
        output_fft_shifted_imag=torch.imag(output_fft_shifted)
        target_fft_shifted_real=torch.real(target_fft_shifted)
        target_fft_shifted_imag=torch.imag(target_fft_shifted)
        # 分别计算实部和虚部之间的差异作为损失，并加入约束
        real_loss = torch.mean(torch.abs(output_fft_shifted_real - target_fft_shifted_real))
        imag_loss = torch.mean(torch.abs(output_fft_shifted_imag - target_fft_shifted_imag))
        loss = real_loss + self.alpha * imag_loss  # 可以根据需要设置不同的权重
        
        return loss
    
