import torch
from torch import nn


def conv_layer(in_channels, out_channels, kernel_size):
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)


def pixelshuffle_block(in_channels, out_channels, upscale_factor=2):
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size=3)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(conv, pixel_shuffle)


class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = in_channels - self.distilled_channels
        self.c1 = conv_layer(in_channels, in_channels, kernel_size=3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, kernel_size=3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, kernel_size=3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, kernel_size=3)
        self.act = nn.ReLU(inplace=True)
        self.c5 = conv_layer(self.distilled_channels * 4, in_channels, kernel_size=1)

    def forward(self, input):
        out_c1 = self.act(self.c1(input))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        out_c4 = self.c4(remaining_c3)
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out_fused = self.c5(out) + input
        return out_fused


class ESPCN4x(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=3, out_nc=3, upscale=4):
        super(ESPCN4x, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=5)  # Changed to kernel_size=5

        # IMDBs
        self.IMDB1 = IMDModule(in_channels=nf)
        self.IMDB2 = IMDModule(in_channels=nf)
        self.IMDB3 = IMDModule(in_channels=nf)

        self.c = conv_layer(nf * num_modules, nf, kernel_size=1)

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        self.upsampler = pixelshuffle_block(nf, out_nc, upscale_factor=upscale)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea
        output = self.upsampler(out_lr)
        return torch.clamp(output, 0.0, 1.0)
