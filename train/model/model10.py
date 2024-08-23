import torch
from torch import nn
from torch.quantization import DeQuantStub, QuantStub


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


class ESPCN4x_quantization(nn.Module):
    def __init__(self, scale=4, num_features=64, num_blocks=4):
        super(ESPCN4x_quantization, self).__init__()
        self.scale = scale

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.first_conv = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[ECB(num_features, num_features) for _ in range(num_blocks)])
        self.last_conv = nn.Conv2d(num_features, 3 * (scale**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)

        self.add = nn.quantized.FloatFunctional()
        self.clamp = nn.quantized.FloatFunctional()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.quant(x)
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.last_conv(x)
        x = self.pixel_shuffle(x)
        x = self.clamp.clamp(x, 0.0, 1.0)
        return self.dequant(x)

    def fuse_model(self):
        for m in self.modules():
            if isinstance(m, ECB):
                torch.quantization.fuse_modules(m, ['conv1', 'act1'], inplace=True)
                torch.quantization.fuse_modules(m, ['conv2', 'act2'], inplace=True)
