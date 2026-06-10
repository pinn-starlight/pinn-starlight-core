"""可学习 I_city — 全重写 (2026-06)

支持三种模式:
  A. 固定解析公式: I_city = (α - 18)*bg (cos背景) 或 α*bg (线性背景)
  B. 单源可学习偏移: I_city = A*cos(r/D) + B*(r/D)^4, cx,cy 可学
  C. 低分辨率网格: 32×32 参数 I_city 场, 双线性插值到全分辨率

用法:
  ic = IcityFixed(alpha=2.0, bg_func=bg)        # 模式A
  ic = IcityLearnable(A=0.3, D=1.0, lr_c=0.01) # 模式B
  ic = IcityGrid(size=32, device='cpu')         # 模式C
  I_city = ic(coords)                           # (N,)
  params += ic.parameters()                     # 可学参数加入 Adam
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 模式 A: 固定 I_city
# ============================================================
class IcityFixed:
    """已知背景下的固定 I_city。不产生任何可学习参数。"""
    def __init__(self, I_city_tensor):
        self.I_city = I_city_tensor

    def __call__(self, coords):
        return self.I_city

    def parameters(self):
        return []


# ============================================================
# 模式 B: 单源可学习偏移
# ============================================================
class IcityLearnable(nn.Module):
    """Garstang 形式 r = sqrt((x-cx)²+(y-cy)²), cx,cy 可学。

    cx,cy 初始化为图像中心 (0.5, 0.5)，用单独的学习率驱动。
    """
    def __init__(self, A=0.3, B=0.02, D=1.0):
        super().__init__()
        self.A = A
        self.B = B
        self.D = D
        # 从图像中心出发（不是 0），梯度信号更强
        self.cx = nn.Parameter(torch.tensor([0.5]))
        self.cy = nn.Parameter(torch.tensor([0.5]))

    def forward(self, coords):
        dx = coords[:, 0] - self.cx
        dy = coords[:, 1] - self.cy
        r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        r_D = r / self.D
        return self.A * torch.cos(r_D) + self.B * (r_D ** 4)

    def extra_params(self):
        """返回 (cx, cy) 给独立优化器（学习率可以调得比主网络大）"""
        return [self.cx, self.cy]


# ============================================================
# 模式 C: 低分辨率可学习网格
# ============================================================
class IcityGrid(nn.Module):
    """32×32 可学习 I_city 场，双线性插值到任意坐标。

    没有光源形状假设——完全让网络从数据里学。
    """
    def __init__(self, grid_size=32):
        super().__init__()
        self.grid_size = grid_size
        self.grid = nn.Parameter(torch.zeros(1, 1, grid_size, grid_size))

    def forward(self, coords):
        # coords: (N, 2), 归一化到 [0,1] 映射到 [-1,1]
        xy = coords[:, :2].unsqueeze(0).unsqueeze(0) * 2.0 - 1.0   # (1, N, 1, 2)
        sampled = F.grid_sample(self.grid, xy, align_corners=True,
                                padding_mode='border')               # (1, 1, 1, N)
        return sampled.squeeze()                                     # (N,)

    def extra_params(self):
        return [self.grid]
