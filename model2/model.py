import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ===== U-Net =====
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(2, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        # head
        self.real_head = nn.Conv2d(32, 1, 1)
        self.imag_head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        self.input_shape = x.shape

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.up3(b)
        d3, e3 = match_shape(d3, e3)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2, e2 = match_shape(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1, e1 = match_shape(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # ===== head =====
        real = self.real_head(d1)
        imag = self.imag_head(d1)

        # ===== resize =====
        real = F.interpolate(
            real, size=self.input_shape[2:], mode="bilinear", align_corners=False
        )
        imag = F.interpolate(
            imag, size=self.input_shape[2:], mode="bilinear", align_corners=False
        )

        # ===== activation =====
        real = torch.tanh(real)
        imag = torch.tanh(imag)

        mask = torch.complex(real, imag)

        return mask


def match_shape(x, ref):
    # x: decoder
    # ref: encoder
    _, _, H, W = x.shape
    _, _, H2, W2 = ref.shape

    Hm = min(H, H2)
    Wm = min(W, W2)

    return x[:, :, :Hm, :Wm], ref[:, :, :Hm, :Wm]
