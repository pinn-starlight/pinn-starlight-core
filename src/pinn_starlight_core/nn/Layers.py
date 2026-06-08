import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import tanh

# TODO: 最终方程 ∇²I - αI + I_city = 0 中，拉普拉斯项是否需要乘以物理系数？
#   autograd 对归一化坐标 [0,1] 求两次导，得到的 ∇² 是"单位域"的拉普拉斯。
#   真实照片的坐标是像素尺度（如 4000×6000）或物理尺度（如 km），
#   此时 ∂²/∂x² 的量纲不再是 1/normalized_unit²，而需要乘以 (实际尺度)⁻²。
#   结论：归一化坐标下 autograd ∇² 可直接用；物理坐标下需要缩放因子。
#   （夏，2026）

class SkyglowLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr = 0.001):
        super(SkyglowLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.a_l = None
        self.W_l = nn.Parameter(torch.empty(in_dim, out_dim))
        init.xavier_uniform_(self.W_l)
        self.b_l = nn.Parameter(torch.zeros(out_dim))

    def forward(self, a_l):
        self.a_l = a_l
        return a_l @ self.W_l + self.b_l

class SkyglowActivation:
    def __init__(self) -> None :
        self.Z_l = None

    def forward(self, Z_l):
        self.Z_l = Z_l
        return tanh(Z_l)