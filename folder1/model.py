import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, 1)
        self.W_x = nn.Conv2d(F_l, F_int, 1)
        self.psi = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(F_int, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_ca=False):
        super().__init__()
        self.use_ca = use_ca

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        if use_ca:
            self.ca = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_ca:
            x = self.ca(x)
        return x


# ===== U-Net =====
class UNet(nn.Module):
    def __init__(self, use_ca=False, use_skip_attn=False):
        super().__init__()

        self.use_skip_attn = use_skip_attn

        # Encoder
        self.enc1 = DoubleConv(1, 32, use_ca)
        self.enc2 = DoubleConv(32, 64, use_ca)
        self.enc3 = DoubleConv(64, 128, use_ca)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256, use_ca)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128, use_ca)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64, use_ca)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32, use_ca)

        # Skip Attention
        if use_skip_attn:
            self.att3 = AttentionGate(128, 128, 64)
            self.att2 = AttentionGate(64, 64, 32)
            self.att1 = AttentionGate(32, 32, 16)

        # Output
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        input_shape = x.shape

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder
        d3 = self.up3(b)
        d3, e3 = match_shape(d3, e3)

        if self.use_skip_attn:
            e3 = self.att3(d3, e3)

        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2, e2 = match_shape(d2, e2)

        if self.use_skip_attn:
            e2 = self.att2(d2, e2)

        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1, e1 = match_shape(d1, e1)

        if self.use_skip_attn:
            e1 = self.att1(d1, e1)

        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.out_conv(d1)
        out = F.interpolate(out, size=input_shape[2:], mode="bilinear", align_corners=False)
        out = torch.sigmoid(out)

        return out


def match_shape(x, ref):
    # x: decoder
    # ref: encoder
    _, _, H, W = x.shape
    _, _, H2, W2 = ref.shape

    Hm = min(H, H2)
    Wm = min(W, W2)

    return x[:, :, :Hm, :Wm], ref[:, :, :Hm, :Wm]
