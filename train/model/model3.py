# Optimized Lightweight IMDN

import torch
import torch.nn.functional as F
from torch import nn


class OptimizedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super(OptimizedConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=bias)

    def forward(self, x):
        return F.leaky_relu(self.conv(x), negative_slope=0.05, inplace=True)


class OptimizedIMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(OptimizedIMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = in_channels - self.distilled_channels

        self.c1 = OptimizedConv(in_channels, in_channels, 3)
        self.c2 = OptimizedConv(self.remaining_channels, in_channels, 3)
        self.c3 = OptimizedConv(self.remaining_channels, in_channels, 3)
        self.c4 = OptimizedConv(self.remaining_channels, self.distilled_channels, 3)
        self.c5 = nn.Conv2d(self.distilled_channels * 4, in_channels, 1)

    def forward(self, input):
        out_c1 = self.c1(input)
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.c2(remaining_c1)
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.c3(remaining_c2)
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) + input
        return out_fused


class ESPCN4x(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=64, num_modules=3):
        super(ESPCN4x, self).__init__()
        self.conv_1 = OptimizedConv(in_channels, nf, 5)
        self.imd_modules = nn.Sequential(*[OptimizedIMDModule(nf) for _ in range(num_modules)])
        self.conv_last = nn.Conv2d(nf, out_channels * 16, 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(4)
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.imd_modules(x)
        x = self.conv_last(x)
        x = self.pixel_shuffle(x)
        return torch.clamp(x, 0.0, 1.0)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
