# TODO: 自己从头写可学习 I_city
#
# Screened Poisson: ∇²I - αI + I_city = 0
# I_city 容纳所有无法被拉普拉斯+αI吸收的贡献
#
# 三种模式:
#   A. 标量: nn.Parameter(tensor([0.0])) — 全局常数，适均匀发光
#   B. 偏移: cx,cy → r=√((x-cx)²+(y-cy)²) → Garstang I_city
#       注意: cx,cy 梯度弱，需独立 lr
#   C. 网格: nn.Parameter(1,1,32,32) → F.grid_sample
#       无光源形状假设，需平滑正则
#
# 论文要点: PDE 不依赖小角度近似，只有 I_city 解析式依赖。
#   大 FOV 下放弃解析式 → 用可学习表示。
import torch
from torch import nn
import torch.nn.functional as F


def tv_loss(grid):
    diff_h = torch.abs(grid[:, :, 1:, :] - grid[:, :, :-1, :])
    diff_w = torch.abs(grid[:, :, :, 1:] - grid[:, :, :, :-1])

    return diff_h.mean() + diff_w.mean()


class LearnableIcity(nn.Module):
    def __init__(self, grid_size = 16, device = "cpu", values = None):
        super().__init__()
        self.device = device

        self.values = values.mean().item()
        self.grid = nn.Parameter(torch.full((1, 1, grid_size, grid_size), self.values)).to(self.device)


    def forward(self, coords):
        grid_coords = coords.unsqueeze(0).unsqueeze(0)

        grid_coords = 2 * grid_coords - 1

        values = F.grid_sample(
            self.grid,
            grid_coords,
            mode='bilinear',
            padding_mode='border',
            align_corners=True,
        )
        return values.squeeze()


class GarstangIcity(nn.Module):
    def __init__(self, alpha, A, B):
        super().__init__()
        self.alpha = alpha
        self.A = A
        self.B = B

    def forward(self, coords):
        r = torch.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
        f = ((self.A / torch.sqrt(self.alpha)) * ((2 / self.D) * torch.cos(r * torch.sqrt(self.alpha))  + (1 / r) * torch.sin(r * torch.sqrt(self.alpha))) - self.B * (r ** 2) * (self.alpha ** 2)) * (16 + (r ** 2) * self.alpha ** 2)
        return f
