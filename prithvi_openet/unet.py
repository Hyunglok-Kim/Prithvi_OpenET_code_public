"""Refinement U-Net: two poolings (224 -> 112 -> 56 -> 224), GroupNorm and GELU, one output channel per head."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, gn_groups=16):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False)
        self.gn = nn.GroupNorm(min(gn_groups, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(ConvGNAct(in_ch, out_ch), ConvGNAct(out_ch, out_ch))

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class RefineUNet(nn.Module):
    """(B, in_ch, H, W) -> (B, n_out, H, W)."""

    def __init__(self, in_ch, base_ch=32, n_out=5):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.bott = DoubleConv(base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch, base_ch)
        self.outc = nn.Conv2d(base_ch, n_out, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        xd = self.up2(self.bott(x3), x2)
        xd = self.up1(xd, x1)
        return self.outc(xd)
