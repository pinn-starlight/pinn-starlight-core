import numpy as np
import torch
import torchvision.transforms.functional as F
from torch import nn

from pinn_starlight_core.data.image_loader import ImageLoader


class Alpha(nn.Module):
    def __init__(self, init: float = 1.0, alpha_min: float = 0.5, alpha_max: float = 1.5):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.init = init

        ratio = (init - alpha_min) / (alpha_max - alpha_min)
        ratio = torch.tensor(ratio, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)
        self.raw_alpha = nn.Parameter(torch.logit(ratio))

    def forward(self):
        length = self.alpha_max - self.alpha_min
        ratio = torch.sigmoid(self.raw_alpha)
        return self.alpha_min + length * ratio

    def get_str(self):
        return f"[min:{self.alpha_min},max:{self.alpha_max},init:{self.init}]"


# Fixed A,B (A,B = 1)
# 当前版本使用以可学习 (x, y) 为中心的单源解析 I_city。
# 将点源式 I_city 扩展为可学习范围/边界的版本，用源区尺度或轮廓来表达光污染覆盖范围。
def _point_source_term(r : torch.Tensor, alpha : torch.Tensor):
    sqrt_alpha = torch.sqrt(alpha)

    cos_term = 2 * alpha * torch.cos(r * sqrt_alpha)
    sin_term = (sqrt_alpha / r) * torch.sin(r * sqrt_alpha)
    quad_term = -16 * (r ** 2) * (alpha ** 2)
    quartic_term = (r ** 4) * (alpha ** 3)

    return cos_term + sin_term + quad_term + quartic_term


def _estimate_bright_center(gray_img: np.ndarray, kernel_size: int):
    img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0)
    sigma = kernel_size / 3.0
    blurred = F.gaussian_blur(
        img_tensor,
        kernel_size=[kernel_size, kernel_size],
        sigma=[sigma, sigma],
    ).squeeze()

    blurred_small = blurred[::4, ::4]
    threshold = np.quantile(blurred_small.cpu().numpy(), 0.95)
    ys, xs = torch.where(blurred >= threshold)
    if xs.numel() == 0:
        flat_index = int(torch.argmax(blurred).item())
        ys = torch.tensor([flat_index // blurred.shape[1]])
        xs = torch.tensor([flat_index % blurred.shape[1]])

    height, width = gray_img.shape
    x_axis = torch.linspace(-1, 1, width)
    y_axis = torch.linspace(-1, 1, height)
    return x_axis[xs].mean(), y_axis[ys].mean()


def _load_gray_image(loader:ImageLoader):
    rgb = loader.rgb_data
    gray = (
            0.2126 * rgb[:, :, 0] +
            0.7152 * rgb[:, :, 1] +
            0.0722 * rgb[:, :, 2]
    )
    return gray


class Icity(nn.Module):
    def __init__(
        self,
        device,
        kernel_size=31,
        loader: ImageLoader | None = None,
        gray_img: np.ndarray | None = None,
        center_mode="bright_init_learnable",
    ):
        super().__init__()
        self.device = device
        self.sigma_min = 0.05
        self.sigma_scale = 0.75
        self.theta_scale = 0.5 * torch.pi

        if gray_img is None:
            if loader is None:
                raise ValueError("loader 和 gray_img 不能同时为空")
            gray_img = _load_gray_image(loader)
        gray_array = np.asarray(gray_img, dtype=np.float32)
        self.H, self.W = gray_array.shape
        if center_mode == "origin_fixed":
            x_init = torch.tensor(0.0)
            y_init = torch.tensor(0.0)
            center_learnable = False
        elif center_mode in {"bright_init_fixed", "bright_init_learnable"}:
            x_init, y_init = _estimate_bright_center(gray_array, kernel_size)
            center_learnable = center_mode == "bright_init_learnable"
        else:
            raise ValueError(f"未知 center_mode：{center_mode}")

        self.x = nn.Parameter(
            x_init.to(self.device).reshape(1),
            requires_grad=center_learnable,
        )
        self.y = nn.Parameter(
            y_init.to(self.device).reshape(1),
            requires_grad=center_learnable,
        )
        self.raw_sigma_x = nn.Parameter(torch.tensor([0.0], device=self.device))
        self.raw_sigma_y = nn.Parameter(torch.tensor([0.0], device=self.device))
        self.raw_theta = nn.Parameter(torch.tensor([0.0], device=self.device))

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
