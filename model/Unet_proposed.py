
import torch
import torch.nn as nn
import torch.nn.functional as F

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class f_module(nn.Module):
    def __init__(self, c):
        super(f_module, self).__init__()
        k = 512
        dim1 = int(k/c)

        self.f_vl1 = nn.Sequential(
            nn.Conv2d(c, dim1, kernel_size=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True)
        )
        self.f_vl2 = nn.Sequential(
            nn.Conv2d(c, dim1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True)
        )
        self.f_vl3 = nn.Sequential(
            nn.Conv2d(c, dim1, kernel_size=5, padding=2),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True)
        )
        self.f_ir1 = nn.Sequential(
            nn.Conv2d(c, dim1, kernel_size=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True)
        )
        self.f_ir2 = nn.Sequential(
            nn.Conv2d(c, dim1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True)
        )
        self.f_ir3 = nn.Sequential(
            nn.Conv2d(c, dim1, kernel_size=5, padding=2),
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True)
        )

        self.fc_vl_atten = nn.Sequential(
            nn.Conv2d(dim1+dim1+dim1+dim1+dim1+dim1, c, kernel_size=1),
            nn.BatchNorm2d(c)
        )
        self.fc_vl_atten2 = nn.Sequential(
            nn.Conv2d(dim1+dim1+dim1+dim1+dim1+dim1, c, kernel_size=1),
            nn.BatchNorm2d(c)
        )
        self.fc_ir_atten = nn.Sequential(
            nn.Conv2d(dim1+dim1+dim1+dim1+dim1+dim1, c, kernel_size=1),
            nn.BatchNorm2d(c)
        )
        self.fc_ir_atten2 = nn.Sequential(
            nn.Conv2d(dim1+dim1+dim1+dim1+dim1+dim1, c, kernel_size=1),
            nn.BatchNorm2d(c)
        )
        self.fc_deconv_atten = nn.Sequential(
            nn.Conv2d(dim1+dim1+dim1+dim1+dim1+dim1, c, kernel_size=1),
            nn.BatchNorm2d(c)
        )


        self.sigmoid = nn.Sigmoid()

    def forward(self, inputx, inputir):

        vl_feat1 = self.f_vl1(inputx)  # dim1
        vl_feat2 = self.f_vl2(inputx)
        vl_feat3 = self.f_vl3(inputx)

        ir_feat1 = self.f_ir1(inputir)  # dim1
        ir_feat2 = self.f_ir2(inputir)
        ir_feat3 = self.f_ir3(inputir)
        cat_feat = torch.cat([vl_feat1, vl_feat2, vl_feat3, ir_feat1, ir_feat2, ir_feat3], dim=1)

        vl_gap = self.fc_vl_atten(cat_feat)# [batch, dim, h, w]
        tview = vl_gap.view(vl_gap.shape[:2] + (-1,))
        vl_weight = torch.mean(tview, dim=-1)

        vl_gap = self.fc_vl_atten2(cat_feat)  # [batch, dim, h, w]
        tview = vl_gap.view(vl_gap.shape[:2] + (-1,))
        vl_weight2 = torch.mean(tview, dim=-1)

        ir_gap = self.fc_ir_atten(cat_feat)
        tview = ir_gap.view(ir_gap.shape[:2] + (-1,))
        ir_weight = torch.mean(tview, dim=-1)

        ir_gap = self.fc_ir_atten2(cat_feat)
        tview = ir_gap.view(ir_gap.shape[:2] + (-1,))
        ir_weight2 = torch.mean(tview, dim=-1)

        dim_diff = len(vl_gap.shape) - len(vl_weight.shape)
        vl_weight = vl_weight.view(vl_weight.shape + (1,)*dim_diff)
        ir_weight = ir_weight.view(ir_weight.shape + (1,)*dim_diff)

        vl_out = 2*self.sigmoid(vl_weight) * inputx
        ir_out = 2*self.sigmoid(ir_weight) * inputir

        vl_weight2 = vl_weight2.view(vl_weight2.shape + (1,)*dim_diff)
        ir_weight2 = ir_weight2.view(ir_weight2.shape + (1,)*dim_diff)

        vl_out2 = 2*self.sigmoid(vl_weight2) * inputx
        ir_out2 = 2*self.sigmoid(ir_weight2) * inputir

        dec_out = vl_out2 + ir_out2 + vl_out + ir_out
        return vl_out, ir_out, dec_out

class Unet(nn.Module):
    def __init__(self, n_class, bilinear=False):
        super(Unet, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(3, 32)
        self.inc_ir = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        self.downi1 = Down(32, 64)
        self.downi2 = Down(64, 128)
        self.downi3 = Down(128, 256)
        self.downi4 = Down(256, 512)

        self.sf1 = f_module(32)
        self.sf2 = f_module(64)
        self.sf3 = f_module(128)
        self.sf4 = f_module(256)
        self.sf5 = f_module(512)

        self.up1 = Up(512, 256, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up3 = Up(128, 64, bilinear)
        self.up4 = Up(64, 32, bilinear)

        self.outc = OutConv(32, n_class)

    def forward(self, x, ir):
        x1 = self.inc(x)
        ir1 = self.inc_ir(ir)
        x1_, ir1_, sf1 = self.sf1(x1, ir1)

        x2 = self.down1(x1_)
        ir2 = self.downi1(ir1_)
        x2_, ir2_, sf2 = self.sf2(x2, ir2)

        x3 = self.down2(x2_)
        ir3 = self.downi2(ir2_)
        x3_, ir3_, sf3 = self.sf3(x3, ir3)

        x4 = self.down3(x3_)
        ir4 = self.downi3(ir3_)
        x4_, ir4_, sf4 = self.sf4(x4, ir4)

        x5 = self.down4(x4_)
        ir5 = self.downi4(ir4_)
        _, _, sf5 = self.sf5(x5, ir5)

        D4 = self.up1(sf5, sf4)
        D3 = self.up2(D4, sf3)
        D2 = self.up3(D3, sf2)
        D1 = self.up4(D2, sf1)

        logits = self.outc(D1)
        return logits