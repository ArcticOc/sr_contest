import torch
from torch import nn


class IMDB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDB, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = in_channels - self.distilled_channels
        self.c1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.c2 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, 1)
        self.c3 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, 1)
        self.c4 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, 1, 1)
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.c5 = nn.Conv2d(in_channels, in_channels, 1)

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
    def __init__(self):
        super(ESPCN4x, self).__init__()
        self.scale = 4
        self.entry = nn.Conv2d(1, 64, 3, 1, 1)

        self.IMDBs = nn.ModuleList([IMDB(64) for _ in range(6)])

        self.conv_mid = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_out = nn.Conv2d(64, 16, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[-2], x.shape[-1])

        fea = self.entry(x)
        out_B1 = self.IMDBs[0](fea)
        out_B2 = self.IMDBs[1](out_B1)
        out_B3 = self.IMDBs[2](out_B2)
        out_B4 = self.IMDBs[3](out_B3)
        out_B5 = self.IMDBs[4](out_B4)
        out_B6 = self.IMDBs[5](out_B5)

        out_B = self.conv_mid(out_B6)
        out_B = out_B + fea

        out = self.conv_out(out_B)
        out = self.pixel_shuffle(out)

        out = out.reshape(-1, 3, out.shape[-2], out.shape[-1])
        out = torch.clamp(out, 0.0, 1.0)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
