import torch
from torch import nn


class CC(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CC, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_mean = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )
        self.conv_std = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ca_mean = self.conv_mean(self.avg_pool(x))
        m_batchsize, C, _, _ = x.size()
        ca_std = self.conv_std(torch.std(x.view(m_batchsize, C, -1), dim=2, keepdim=True).view(m_batchsize, C, 1, 1))
        cc = (ca_mean + ca_std) / 2.0
        return cc


class LatticeBlock(nn.Module):
    def __init__(self, nFeat, nDiff):
        super(LatticeBlock, self).__init__()

        self.conv_block0 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05),
            nn.Conv2d(nFeat - nDiff, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05),
            nn.Conv2d(nFeat - nDiff, nFeat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05),
        )

        self.fea_ca1 = CC(nFeat)
        self.x_ca1 = CC(nFeat)

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05),
            nn.Conv2d(nFeat - nDiff, nFeat - nDiff, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05),
            nn.Conv2d(nFeat - nDiff, nFeat, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.05),
        )

        self.fea_ca2 = CC(nFeat)
        self.x_ca2 = CC(nFeat)

        self.compress = nn.Conv2d(2 * nFeat, nFeat, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x_feature_shot = self.conv_block0(x)
        fea_ca1 = self.fea_ca1(x_feature_shot)
        x_ca1 = self.x_ca1(x)

        p1z = x + fea_ca1 * x_feature_shot
        q1z = x_feature_shot + x_ca1 * x

        x_feat_long = self.conv_block1(p1z)
        fea_ca2 = self.fea_ca2(q1z)
        p3z = x_feat_long + fea_ca2 * q1z
        x_ca2 = self.x_ca2(x_feat_long)
        q3z = q1z + x_ca2 * x_feat_long

        out = self.compress(torch.cat((p3z, q3z), 1))
        return out


class ESPCN4x(nn.Module):
    def __init__(self, scale=4, n_feats=64, n_blocks=4):
        super(ESPCN4x, self).__init__()

        self.scale = scale
        nDiff = 16

        self.head = nn.Sequential(
            nn.Conv2d(3, n_feats, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True),
        )

        body = [LatticeBlock(n_feats, nDiff) for _ in range(n_blocks)]
        self.body = nn.Sequential(*body)

        tdm_layers = []
        for _ in range(n_blocks - 1):
            tdm_layers.extend(
                [
                    nn.Sequential(nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True), nn.ReLU()),
                    nn.Sequential(nn.Conv2d(n_feats, n_feats // 2, kernel_size=1, padding=0, bias=True), nn.ReLU()),
                ]
            )
        self.tdm_layers = nn.ModuleList(tdm_layers)

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True),
            nn.Conv2d(n_feats, 3 * (scale**2), kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(scale),
        )

    def forward(self, x):
        x = self.head(x)

        res = x
        tdm_outs = []
        for i, block in enumerate(self.body):
            res = block(res)
            if i < len(self.body) - 1:
                tdm_outs.append(res)

        for i in range(len(tdm_outs) - 1, -1, -1):
            T_tdm = self.tdm_layers[i * 2](res)
            L_tdm = self.tdm_layers[i * 2 + 1](tdm_outs[i])
            res = torch.cat((T_tdm, L_tdm), 1)

        res += x
        out = self.tail(res)

        return out
