import torch
import torch.nn as nn
import torch.nn.functional as F

# from tensorboardX import SummaryWriter
from ptflops import get_model_complexity_info
from timm.models.layers import DropPath
from modeling.deform_conv_v2 import *
class DSC(nn.Module):
    """
    先是depthwiseConv，本质上就是分组卷积，在深度可分离卷积中，分组卷积的组数=输入通道数=输出通道数，该部分通道数不变
    再是pointwisejConv，就是点卷积，该部分负责扩展通道数，所以其kernel_size=1，不用padding
    """
    def __init__(self, in_channel, out_channel, ksize ,padding="same",bais=True):
        super(DSC, self).__init__()
        self.depthwiseConv = nn.Conv2d(in_channels=in_channel,
                                       out_channels=in_channel,
                                       groups=in_channel,
                                       kernel_size=ksize,
                                       padding=padding,
                                       bias=bais)
        self.pointwiseConv = nn.Conv2d(in_channels=in_channel,
                                       out_channels=out_channel,
                                       kernel_size=1,
                                       padding=0,
                                       bias=bais)

    def forward(self, x):
        out = self.depthwiseConv(x)
        out = self.pointwiseConv(out)
        return out

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels,kernel_size):
        super().__init__()
        self.double_conv = nn.Sequential(
            DSC(in_channels, out_channels, ksize=kernel_size),
            DeformConv2d(out_channels, out_channels, 3, padding=1, bias=False, modulation=True),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding="same",groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,kernel_size):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,kernel_size),
        )

    def forward(self, x):
        x1 = self.maxpool_conv(x)
        return x1


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class Up1(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,kernel_size, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels,kernel_size)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels,dep, bilinear=True,line=True):
        super().__init__()
        self.line = line
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels,3)
        self.norm = LayerNorm(in_channels//2, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels//2, dep * in_channels)
        self.act = nn.Hardswish()
        self.pwconv2 = nn.Linear(dep * in_channels, in_channels//2)
    def forward(self, x1, x2):
        if(self.line):
            x2 = x2.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
            x2 = self.norm(x2)
            x2 = self.pwconv1(x2)
            x2 = self.act(x2)
            x2 = self.pwconv2(x2)
            x2 = x2.permute(0, 3, 1, 2)
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        kk = 16
        self.inc = DoubleConv(n_channels, 64//kk,1)
        self.down1 = Down(64//kk, 128//kk,3)
        self.down2 = Down(128//kk, 256//kk,5)
        self.down3 = Down(256//kk, 512//kk,7)
        self.down4 = Down(512//kk, 512//kk,9)
        self.up1 = Up1(1024//kk, 256//kk,3, bilinear)
        self.up2 = Up1(512//kk, 128//kk,3, bilinear=False)
        self.up3 = Up1(256//kk, 64//kk,3,bilinear=False)
        self.up4 = Up(128//kk, 64//kk,4,bilinear,line=True)
        self.outc = OutConv(64//kk, n_classes)
        self.outc1 = OutConv(64 // kk, n_classes)
        self.outc2 = OutConv(128 // kk, n_classes)
        self.outc3 = OutConv(256 // kk, n_classes)
        self.outc4 = OutConv(64 // kk, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        logits3 = self.outc3(x)
        x = self.up2(x, x3)
        logits2 = self.outc2(x)
        x = self.up3(x, x2)
        logits1= self.outc1(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits4 = self.outc4(x)

        #return torch.sigmoid(logits),torch.sigmoid(logits1),torch.sigmoid(logits2),torch.sigmoid(logits3)
        return logits,logits1,logits2,logits3,torch.sigmoid(logits4)

if __name__ == '__main__':
    x = torch.rand([1, 3, 512, 512])
    sanet = Unet(n_channels=3, n_classes=6)
    flops, params = get_model_complexity_info(sanet, (3, 512, 512), as_strings=True, print_per_layer_stat=True,
                                              verbose=True)
    print('Flops:  ', flops)
    print('Params: ', params)

    y = sanet(x)
