import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import conv, fft2d, ifft2d, fftshift2d, Upsampler
from .complexLayers import ComplexConv2d, ComplexMaxPool2d,ComplexAvgPool2d, ComplexConv1d, ComplexReLU, ComplexSigmoid
from torch.fft import ifft2,fftshift
from .decay_matrix import decay_matrix
import torchvision.transforms

class CALayer(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SCALayer(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act=nn.ReLU()):
        super().__init__()
        self.amplitude_conv = conv(n_feat, n_feat, kernel_size)
        self.act = act
        self.global_average_pooling2d = nn.AdaptiveAvgPool2d(1)
        self.global_max_pooling2d = nn.AdaptiveMaxPool2d(1)
        self.space_attention = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.act(self.amplitude_conv(x))
        y_avg = self.global_average_pooling2d(y).view(b, c, 1, 1)
        y_max = self.global_max_pooling2d(y).view(b, c, 1, 1)
        out = torch.cat([y_avg, y_max], dim=1)
        out = self.space_attention(out.squeeze(-1).transpose(-1, -2))
        sa = out.transpose(-1, -2).unsqueeze(-1)
        output = x * sa
        return output


class FCALayer(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act=ComplexReLU()):
        super().__init__()
        self.frequency_conv = ComplexConv2d(n_feat, n_feat, kernel_size)
        self.act = act
        self.global_average_pooling2d_train = ComplexAvgPool2d(kernel_size=128, stride=1, padding=0) #train
        self.global_average_pooling2d_test = ComplexAvgPool2d(kernel_size=512, stride=1, padding=0) #test

        self.global_max_pooling2d_train = ComplexMaxPool2d(kernel_size=128, stride=1, padding=0) #train
        self.global_max_pooling2d_test = ComplexMaxPool2d(kernel_size=512, stride=1, padding=0) #test

        self.frequency_attention = nn.Sequential(
            ComplexConv1d(1, 1, kernel_size=3, padding=1, stride=2, bias=False),
            ComplexSigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_fft = fftshift2d(fft2d(x))
        frequency_x = self.act(self.frequency_conv(x_fft))
        if self.training:
            y_avg = self.global_average_pooling2d_train(frequency_x).view(b, c, 1, 1)
        else:
            y_avg = self.global_average_pooling2d_test(frequency_x).view(b, c, 1, 1)
        if self.training:
            y_max = self.global_max_pooling2d_train(frequency_x).view(b, c, 1, 1)
        else:
            y_max = self.global_max_pooling2d_test(frequency_x).view(b, c, 1, 1)
        y = torch.cat([y_avg, y_max], dim=1)
        y = self.frequency_attention(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = torch.abs(ifft2d(y))
        output = x * y
        return output


class SFCALayer(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act=nn.ReLU()):
        super().__init__()
        self.K = 2
        self.T = n_feat
        self.act = act
        self.conv = conv(n_feat, n_feat, 1)
        self.space_attention = SCALayer(conv, n_feat, kernel_size, reduction)
        self.frequency_attention = FCALayer(conv, n_feat, kernel_size, reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight = nn.Sequential(
            conv(n_feat, n_feat // reduction, kernel_size=1),
            nn.ReLU(),
            conv(n_feat // reduction, self.K, kernel_size=1))

    def forward(self, x):
        b, c, h, w = x.shape
        space_map = self.space_attention(x)
        frequency_map = self.frequency_attention(x)
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.weight(y)
        ax = F.softmax(y / self.T, dim=1)
        alpha, _ = torch.min(ax, dim=1)
        beta, _ = torch.max(ax, dim=1)
        output = space_map * alpha.view(b, 1, 1, 1) + frequency_map * beta.view(b, 1, 1, 1)
        output = self.act(self.conv(output))
        output += x
        return output


class SFCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU()):
        super(SFCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0:
                modules_body.append(act)
        modules_body.append(SFCALayer(conv, n_feat, kernel_size, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            SFCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act) for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class OIDN(nn.Module):
    def __init__(self, input_shape=9, out_ch=9, scale=2):
        super(OIDN, self).__init__()
        n_resgroups = 4
        n_resblocks = 4
        n_feats = 64
        kernel_size = 3
        reduction = 16
        act = nn.ReLU()
        modules_head = [conv(9, n_feats, kernel_size)]
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, n_resblocks=n_resblocks) for _ in range(n_resgroups)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        modules_tail = [Upsampler(conv, 2, n_feats, act=act),
                        conv(n_feats, 1, kernel_size)]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=scale)

        self.Conv_x = nn.Conv2d(64, 9, kernel_size=3, stride=1, padding=1)
        self.Conv_x1 = nn.Conv2d(9, 36, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=out_ch, kernel_size=3, padding=1)
        self.Conv_2_1 = nn.Conv2d(64, out_ch, kernel_size=4, stride=2, padding=1)
        self.Conv_2_2 = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.Conv_2_3 = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.Conv_2_4 = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.Conv_2_5 = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.Conv_2_6 = nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)

        self.activation = nn.Sigmoid()

        OTF = decay_matrix(8, 1)
        OTF = torch.Tensor(OTF)
        OTF = torch.stack([OTF] * 9, 0)
        OTF = torch.unsqueeze(OTF, 0)
        PSF = fftshift(ifft2(fftshift(OTF)))
        PSFo = torch.abs(PSF)
        self.learnable_psf = torch.nn.Parameter(PSFo)

    def forward(self, x, batch_size):
        if not self.training:
            x = nn.ReflectionPad2d((5, 5, 5, 5))(x)  # test
            # x = nn.ReflectionPad2d((0, 0, 128, 128))(x) #test


        x = self.head(x[:, 0:9, :, :])
        res = self.body(x)
        res += x
        y = self.tail(res)

        psf1 = self.Conv_2_1(res)
        psf2 = self.Conv_2_2(psf1)
        psf3 = self.Conv_2_3(psf2)
        psf4 = self.Conv_2_4(psf3)
        bias = self.Conv_2_5(psf4)
        # bias = self.Conv_2_6(psf5)

        if not self.training:
            bias = torch.nn.functional.interpolate(bias, scale_factor=0.25, mode='bicubic')  # test


        bias = torch.cat([torch.flip(bias, (2,)), bias], 2)
        bias = torch.cat([torch.flip(bias, (3,)), bias], 3)

        loss1 = bias[:, :, 1:, :] - bias[:, :, :-1, :]
        loss2 = bias[:, :, :, 1:] - bias[:, :, :, :-1]
        loss_psf = loss1.clip(min=0).sum() + loss2.clip(min=0).sum()

        psf = self.learnable_psf + (1e-5) * bias
        k = psf.size(dim=2)

        # psf.sum =1
        psf_before = torch.flatten(psf, start_dim=2, end_dim=3)
        psf_after = F.softmax(psf_before, dim=2)
        psf = torch.reshape(psf_after, [batch_size, 9, k, k])

        out1 = self.conv3(y)

        ## upsample
        x_out = self.Conv_x(x)
        x_out = self.Conv_x1(x_out)
        x = self.pixel_shuffle(x_out)

        H = x.size(dim=2)
        psf = psf.view(batch_size * 9, 1, k, k)
        out2 = (x - out1).view(1, batch_size * 9, H, H)
        out = nn.functional.conv2d(out2, psf, stride=1, padding=int(k / 2), groups=batch_size * 9)

        # crop
        if self.training:
            Centercrop = torchvision.transforms.CenterCrop((H, H))  # train 256
        else:
            Centercrop = torchvision.transforms.CenterCrop((1004, 1004))  # test 1004
        out = Centercrop(out)

        # out = out[..., 0:512, 0:1024]

        if self.training:
            out = out.view(batch_size, 9, H, H).sum(dim=1, keepdim=True)  # train
        else:
            out = out.view(batch_size, 9, 1004, 1004).sum(dim=1, keepdim=True)  # test
            # out = out.view(batch_size, 9, 512, 1024).sum(dim=1, keepdim=True)  # test

        # out_y = self.activation(out)

        return out, psf, loss_psf
