import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F

import pinn_starlight_core.data.PhotoLoader as Loader


# Fixed A,B (A,B = 1)
# 当前版本使用以可学习 (x, y) 为中心的单源解析 I_city。
# TODO: 将点源式 I_city 扩展为可学习范围/边界的版本，用源区尺度或轮廓来表达光污染覆盖范围。
class Icity(nn.Module):
    def __init__(self, path, device, kernel_size=31):
        super().__init__()
        self.device = device

        loader = Loader.RAWLoader()
        loader.load(path)
        gray_img = np.mean(loader.rgb_data, axis=2).astype(np.float32)
        H, W = gray_img.shape

        img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0)
        sigma = kernel_size / 3.0
        blurred = F.gaussian_blur(img_tensor,
                                   kernel_size=[kernel_size, kernel_size],
                                   sigma=[sigma, sigma]).squeeze()

        bright_mask = blurred > blurred.quantile(0.95)
        ys, xs = torch.where(bright_mask)
        x_axis = torch.linspace(-1, 1, W)
        y_axis = torch.linspace(-1, 1, H)
        x_init = x_axis[xs].mean()
        y_init = y_axis[ys].mean()

        self.x = nn.Parameter(x_init.to(device).unsqueeze(0))
        self.y = nn.Parameter(y_init.to(device).unsqueeze(0))
        self.raw_sigma = nn.Parameter(torch.tensor([0.0], device=self.device))

    def forward(self, coords, alpha):
        x = coords[:, 0]
        y = coords[:, 1]
        r = torch.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + 1e-8)
        f = 2 * alpha * torch.cos(r * torch.sqrt(alpha)) + (torch.sqrt(alpha) / r) * torch.sin(r * torch.sqrt(alpha)) - 16 * (r**2) * (alpha**2) + (r**4) * (alpha ** 3)
        sigma = 0.05 + 0.75 * torch.sigmoid(self.raw_sigma)
        return f + sigma
