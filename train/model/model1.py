# ResNet_baseline

import torch
from torch import nn


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = DepthwiseSeparableConv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class ESPCN4x(nn.Module):
    def __init__(self, num_residual_blocks=2):
        super(ESPCN4x, self).__init__()
        self.scale = 4

        # Initial convolution layer
        self.initial = nn.Sequential(nn.Conv2d(1, 64, kernel_size=9, padding=4), nn.ReLU(inplace=True))

        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Final convolution before pixel shuffle
        self.final = nn.Conv2d(64, 1 * self.scale * self.scale, kernel_size=3, padding=1)

        # Pixel shuffle for upscaling
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

        # Quantization stubs for static quantization
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        x = self.quant(x)  # Quantize the input
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.final(x)
        x = self.pixel_shuffle(x)
        x = self.dequant(x)  # Dequantize the output
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clamp(x, 0.0, 1.0)
        return x
