# Lightweight IMDN

import torch
from torch import nn


def conv_layer(in_channels, out_channels, kernel_size, stride=1, bias=True):
    padding = (kernel_size - 1) // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)


def activation(act_type='relu', neg_slope=0.05, inplace=True):
    if act_type == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_type == 'lrelu':
        return nn.LeakyReLU(neg_slope, inplace=inplace)
    else:
        raise NotImplementedError(f'activation layer [{act_type}] is not found')


class IMDModule_speed(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDModule_speed, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = in_channels - self.distilled_channels
        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c2 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c3 = conv_layer(self.remaining_channels, in_channels, 3)
        self.c4 = conv_layer(self.remaining_channels, self.distilled_channels, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.distilled_channels * 4, in_channels, 1)

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
    def __init__(self, in_channels=3, out_channels=3, nf=64, num_modules=3):
        super(ESPCN4x, self).__init__()
        self.scale = 4
        self.conv_1 = conv_layer(in_channels, nf, 5)
        self.act = activation('lrelu', neg_slope=0.05)

        self.imd_modules = nn.Sequential(*[IMDModule_speed(nf) for _ in range(num_modules)])

        self.conv_last = conv_layer(nf, out_channels * (self.scale**2), 3)
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.act(self.conv_1(x))
        x = self.imd_modules(x)
        x = self.conv_last(x)
        x = self.pixel_shuffle(x)
        return torch.clamp(x, 0.0, 1.0)
