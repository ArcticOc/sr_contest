# XLSR_Baseline

import math

import torch
from torch import nn
from torch.quantization import DeQuantStub, QuantStub


class ClippedReLU(nn.Module):
    def forward(self, x):
        return x.clamp(min=0.0, max=1.0)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class GConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=4):
        super().__init__()
        self.groups = groups
        self.conv2d_block = nn.ModuleList(
            [
                nn.Conv2d(in_channels // groups, out_channels // groups, kernel_size, padding=kernel_size // 2)
                for _ in range(groups)
            ]
        )

    def forward(self, x):
        return torch.cat(
            [conv(xi) for conv, xi in zip(self.conv2d_block, x.chunk(self.groups, 1), strict=False)], dim=1
        )


class Gblock(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        self.conv0 = GConv2d(in_channels, out_channels, groups=groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv1(self.relu(self.conv0(x)))


class ESPCN4x(nn.Module):
    def __init__(self, SR_rate=4):
        super().__init__()
        self.conv0 = nn.ModuleList([ConvRelu(3, 8, 3) for _ in range(4)])
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 1)
        self.conv3 = ConvRelu(48, 32, 1)
        self.conv4 = nn.Conv2d(32, 3 * SR_rate**2, 3, padding=1)
        self.Gblocks = nn.Sequential(*[Gblock(32, 32, 4) for _ in range(3)])
        self.depth2space = nn.PixelShuffle(SR_rate)
        self.clippedReLU = ClippedReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                _, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                std = math.sqrt(2 / fan_out * 0.1)
                nn.init.normal_(m.weight.data, mean=0, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)

    def forward(self, x):
        res = torch.cat([conv(x) for conv in self.conv0], dim=1)
        res = self.conv2(res)
        res = self.Gblocks(res)
        res = torch.cat((res, self.conv1(x)), dim=1)
        res = self.conv3(res)
        res = self.conv4(res)
        res = self.clippedReLU(res)
        return self.depth2space(res)


class XLSR_quantization(nn.Module):
    def __init__(self, SR_rate):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.conv0 = nn.ModuleList([ConvRelu(3, 8, 3) for _ in range(4)])
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 1)
        self.conv3 = ConvRelu(48, 32, 1)
        self.conv4 = nn.Conv2d(32, 3 * SR_rate**2, 3, padding=1)
        self.Gblocks = nn.Sequential(*[Gblock(32, 32, 4) for _ in range(3)])
        self.depth2space = nn.PixelShuffle(SR_rate)
        self.clippedReLU = ClippedReLU()
        self.cat1 = nn.quantized.FloatFunctional()
        self.cat2 = nn.quantized.FloatFunctional()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.01)

    def forward(self, x):
        x = self.quant(x)
        res = self.cat1.cat([conv(x) for conv in self.conv0], dim=1)
        res = self.conv2(res)
        res = self.Gblocks(res)
        res = self.cat2.cat((res, self.conv1(x)), dim=1)
        res = self.conv3(res)
        res = self.conv4(res)
        res = self.depth2space(res)
        res = self.clippedReLU(res)
        return self.dequant(res)

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ConvRelu):
                torch.quantization.fuse_modules(m, ['conv', 'relu'], inplace=True)
            if isinstance(m, Gblock):
                torch.quantization.fuse_modules(m, ['conv0', 'relu'], inplace=True)
