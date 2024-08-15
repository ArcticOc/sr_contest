import torch
from torch import nn


class ESPCN4x(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=6, upscale_factor=4):
        super(ESPCN4x, self).__init__()

        self.fea_conv = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([RFDBLarge(num_features) for _ in range(num_blocks)])
        self.lr_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(num_features, out_channels * (upscale_factor**2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
        )

    def forward(self, x):
        out = self.fea_conv(x)
        residual = out
        for block in self.blocks:
            out = block(out)
        out = self.lr_conv(out) + residual
        return self.upsample(out)


class RFDBLarge(nn.Module):
    def __init__(self, num_features, distillation_rate=0.25):
        super(RFDBLarge, self).__init__()
        self.dc = self.distilled_channels = int(num_features * distillation_rate)
        self.rc = self.remaining_channels = num_features - self.dc

        self.c1_d = nn.Conv2d(num_features, self.dc, kernel_size=1)
        self.c1_r = nn.Conv2d(num_features, self.rc, kernel_size=3, padding=1)
        self.c2_d = nn.Conv2d(self.rc, self.dc, kernel_size=1)
        self.c2_r = nn.Conv2d(self.rc, self.rc, kernel_size=3, padding=1)
        self.c3_d = nn.Conv2d(self.rc, self.dc, kernel_size=1)
        self.c3_r = nn.Conv2d(self.rc, self.rc, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(self.rc, self.dc, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = nn.Conv2d(num_features, num_features, kernel_size=1)
        self.cca = CCALayer(num_features)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)
        out = self.cca(out)
        return out + input


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
