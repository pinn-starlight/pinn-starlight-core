import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F

from pinn_starlight_core.data.PhotoLoader import RAWLoader


# Fixed A,B (A,B = 1)
# 当前版本使用以可学习 (x, y) 为中心的单源解析 I_city。
# TODO: 将点源式 I_city 扩展为可学习范围/边界的版本，用源区尺度或轮廓来表达光污染覆盖范围。
def _point_source_term(r : torch.Tensor, alpha : torch.Tensor):
    sqrt_alpha = torch.sqrt(alpha)

    cos_term = 2 * alpha * torch.cos(r * sqrt_alpha)
    sin_term = (sqrt_alpha / r) * torch.sin(r * sqrt_alpha)
    quad_term = -16 * (r ** 2) * (alpha ** 2)
    quartic_term = (r ** 4) * (alpha ** 3)

    return cos_term + sin_term + quad_term + quartic_term


def _bright_mask(gray_img: np.ndarray, kernel_size: int):
    img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0)
    sigma = kernel_size / 3.0
    blurred = F.gaussian_blur(
        img_tensor,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma],
    ).squeeze()

    blurred_small = blurred[::4, ::4]
    threshold = np.quantile(blurred_small.cpu().numpy(), 0.95)
    return blurred > threshold


def _load_gray_image(loader:RAWLoader):
    rgb = loader.rgb_data
    gray = (
            0.2126 * rgb[:, :, 0] +
            0.7152 * rgb[:, :, 1] +
            0.0722 * rgb[:, :, 2]
    )
    return gray


class Icity(nn.Module):
    def __init__(self, device, kernel_size=31, loader: RAWLoader | None=None):
        super().__init__()
        self.device = device
        self.sigma_min = 0.05
        self.sigma_scale = 0.75
        self.theta_scale = 0.5 * torch.pi

        if loader is None:
            raise ValueError("loader is None")

        gray_img = _load_gray_image(loader)
        bright_mask = _bright_mask(gray_img, kernel_size)
        self.H, self.W = gray_img.shape
        self.x, self.y = self._init_center(bright_mask)
        self.raw_sigma_x = nn.Parameter(torch.tensor([0.0], device=self.device))
        self.raw_sigma_y = nn.Parameter(torch.tensor([0.0], device=self.device))
        self.raw_theta = nn.Parameter(torch.tensor([0.0], device=self.device))

    def _init_center(self, bright_mask):
        ys, xs = torch.where(bright_mask)
        x_axis = torch.linspace(-1, 1, self.W)
        y_axis = torch.linspace(-1, 1, self.H)
        x_init = x_axis[xs].mean()
        y_init = y_axis[ys].mean()

        x = nn.Parameter(x_init.to(self.device).unsqueeze(0))
        y = nn.Parameter(y_init.to(self.device).unsqueeze(0))
        return x, y

    def get_sigma(self):
        sigma_x = self.sigma_min + self.sigma_scale * torch.sigmoid(self.raw_sigma_x)
        sigma_y = self.sigma_min + self.sigma_scale * torch.sigmoid(self.raw_sigma_y)
        return sigma_x, sigma_y

    def get_theta(self):
        return self.theta_scale * torch.tanh(self.raw_theta)

    def forward(self, coords, alpha):
        x = coords[:, 0]
        y = coords[:, 1]
        dx = x - self.x
        dy = y - self.y
        r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)

        sigma_x, sigma_y = self.get_sigma()
        theta = self.get_theta()
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        dx_rot = cos_theta * dx + sin_theta * dy
        dy_rot = -sin_theta * dx + cos_theta * dy

        point_source = _point_source_term(r, alpha)
        envelope = torch.exp(-0.5 * ((dx_rot / sigma_x) ** 2 + (dy_rot / sigma_y) ** 2))

        return point_source * envelope
