# Lightweight XLSR


from torch import nn


class ClippedReLU(nn.Module):
    def __init__(self):
        super(ClippedReLU, self).__init__()

    def forward(self, x):
        return x.clamp(min=0.0, max=1.0)


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(ConvRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ESPCN4x(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=64):
        super(ESPCN4x, self).__init__()
        self.first_part = nn.Sequential(
            ConvRelu(num_channels, d, kernel_size=5),
            ConvRelu(d, d // 2, kernel_size=3),
            ConvRelu(d // 2, d // 2, kernel_size=3),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(d // 2, num_channels * (scale_factor**2), kernel_size=3, padding=1), nn.PixelShuffle(scale_factor)
        )
        self.clippedReLU = ClippedReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return self.clippedReLU(x)
