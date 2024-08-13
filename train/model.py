import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
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
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.final(x)
        x = self.pixel_shuffle(x)
        x = x.reshape(-1, 3, x.shape[-2], x.shape[-1])
        x = torch.clamp(x, 0.0, 1.0)
        return x
