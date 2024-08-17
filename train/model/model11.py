import torch
from torch import nn


class ECB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ECB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.act2 = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out += residual
        out = self.act2(out)
        return out


class ESPCN4x(nn.Module):
    def __init__(self, scale=4, num_features=64, num_blocks=4):
        super(ESPCN4x, self).__init__()
        self.scale = scale

        self.first_conv = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ECB(num_features, num_features) for _ in range(num_blocks)])
        self.last_conv = nn.Conv2d(num_features, 3 * (scale**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.last_conv(x)
        x = self.pixel_shuffle(x)
        return torch.clamp(x, 0.0, 1.0)
