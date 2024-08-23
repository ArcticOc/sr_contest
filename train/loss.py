import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg16


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pred, target):
        return self.l1(pred, target)


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse(pred, target)


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(pretrained=True).features.eval()
        self.slice1 = torch.nn.Sequential(*list(vgg.children())[:4])
        self.slice2 = torch.nn.Sequential(*list(vgg.children())[4:9])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = [self.slice1(input), self.slice2(input)]
        target_features = [self.slice1(target), self.slice2(target)]
        loss = 0
        for i_f, t_f in zip(input_features, target_features, strict=False):
            loss += F.mse_loss(i_f, t_f)
        return loss


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred_real, pred_fake):
        real_label = torch.ones_like(pred_real)
        fake_label = torch.zeros_like(pred_fake)
        loss_real = self.criterion(pred_real, real_label)
        loss_fake = self.criterion(pred_fake, fake_label)
        return (loss_real + loss_fake) / 2


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt(diff * diff + self.eps))
        return loss


# class SSIMLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x, y):
#         return 1 - ssim(x, y, data_range=1, size_average=True)


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        sobel_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.sobel.weight.data[0, 0] = sobel_kernel
        self.sobel.weight.data[1, 0] = sobel_kernel.t()
        for param in self.sobel.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        x_gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        y_gray = 0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]
        x_edge = self.sobel(x_gray.unsqueeze(1))
        y_edge = self.sobel(y_gray.unsqueeze(1))
        return F.l1_loss(x_edge, y_edge)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = L1Loss()
        self.perceptual = VGGPerceptualLoss()
        self.edge = EdgeLoss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        perceptual_loss = self.perceptual(pred, target)
        edge_loss = self.edge(pred, target)
        return 0.5 * l1_loss + 0.3 * perceptual_loss + 0.2 * edge_loss


class LossProxy:
    def __init__(self):
        pass

    def get_loss(self, loss_name):
        loss_class = globals().get(loss_name)
        if loss_class is None:
            raise ValueError(f"Loss function '{loss_name}' is not defined.")
        return loss_class()
