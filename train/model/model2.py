import functools

from torch import nn


class ResidualBlock_noBN(nn.Module):
    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return identity + out


class ESPCN4x(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=2, upscale=4):
        super(ESPCN4x, self).__init__()
        self.scale = upscale

        # Initial convolution layer
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Residual blocks
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk = self.make_layer(basic_block, nb)

        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # Initialize weights
        self._initialize_weights()

    def make_layer(self, block, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(block())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        fea = self.relu(self.conv_first(x))
        out = self.recon_trunk(fea)

        out = self.relu(self.pixel_shuffle(self.upconv1(out)))
        out = self.relu(self.pixel_shuffle(self.upconv2(out)))

        out = self.conv_last(self.relu(self.HRconv(out)))
        base = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out += base
        return out
