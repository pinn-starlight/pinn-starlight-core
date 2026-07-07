# I_city — 粗卷积给初值 + 可学习源点
#
# 思路:
#   1. 粗卷积模糊图 → 找最亮区域质心 → 初始化源点 (cx, cy)
#   2. (cx, cy) 是 nn.Parameter, 训练时可学习
#   3. I_city = A * exp(-r²/(2σ²)) 由源点算出, A 和 σ 也可学习
#   4. PINN 物理损失通过 I_city 反传到 cx, cy, 让源点滑到正确位置
#
# Screened Poisson: ∇²I - αI + I_city = 0
# I_city 容纳所有无法被拉普拉斯+αI吸收的贡献
import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F

import pinn_starlight_core.data.PhotoLoader as Loader


class Icity(nn.Module):
    def __init__(self, path, device, kernel_size=71):
        super().__init__()
        self.device = device

        # ===== 1. 加载图片, 粗卷积求模糊图 =====
        loader = Loader.RAWLoader()
        loader.load(path)
        gray_img = np.mean(loader.rgb_data, axis=2).astype(np.float32)
        H, W = gray_img.shape

        img_tensor = torch.from_numpy(gray_img).unsqueeze(0).unsqueeze(0)
        sigma = kernel_size / 3.0
        blurred = F.gaussian_blur(img_tensor,
                                   kernel_size=[kernel_size, kernel_size],
                                   sigma=[sigma, sigma]).squeeze()

        # ===== 2. 从最亮区域质心估计初始源点 =====
        # 光污染最亮的地方 ≈ 源点附近
        bright_mask = blurred > blurred.quantile(0.95)
        ys, xs = torch.where(bright_mask)
        cx_init = xs.float().mean().item() / W   # 归一化到 [0, 1]
        cy_init = ys.float().mean().item() / H

        # ===== 3. 可学习参数 =====
        # 源点坐标 (cx, cy) — 训练时滑动到正确位置
        self.cx = nn.Parameter(torch.tensor([cx_init], device=device))
        self.cy = nn.Parameter(torch.tensor([cy_init], device=device))
        # 振幅 A — I_city 的强度
        self.amplitude = nn.Parameter(torch.tensor([0.5], device=device))
        # 衰减尺度 σ — 源点影响范围 (大=扩散光污染, 小=点光源)
        self.sigma = nn.Parameter(torch.tensor([0.2], device=device))

    def forward(self, coords):
        """coords: (N, 2) 在 [0, 1] 范围, 返回 (N,) 的 I_city 值"""
        x = coords[:, 0]
        y = coords[:, 1]
        r2 = (x - self.cx) ** 2 + (y - self.cy) ** 2
        # 高斯径向衰减: 离源点越远, I_city 越小
        return self.amplitude * torch.exp(-r2 / (2 * self.sigma ** 2))
