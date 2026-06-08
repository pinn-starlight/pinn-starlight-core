"""可学习光污染源点偏移 — 夏 & Claude 合写"""
import torch
import torch.nn as nn


class LearnableSource(nn.Module):
    """Garstang 形式的 I_city, 源点 (cx, cy) 可学习

    I_city = A*cos(r/D) + B*(r/D)^4
    其中 r = sqrt((x - cx)^2 + (y - cy)^2)

    用法:
        source = LearnableSource(A=0.3, D=1.0)
        params = source.parameters()  # 包含 cx, cy
        I_city = source(coords)       # (N,)
    """

    def __init__(self, A=0.3, B=0.02, D=1.0, cx_init=0.0, cy_init=0.0, device='cpu'):
        super().__init__()
        self.A = A
        self.B = B
        self.D = D
        self.cx = nn.Parameter(torch.tensor([cx_init], device=device))   # 横偏移
        self.cy = nn.Parameter(torch.tensor([cy_init], device=device))   # 纵偏移

    def forward(self, coords):
        """coords: (N, 2) 归一化坐标 [0,1]"""
        dx = coords[:, 0] - self.cx
        dy = coords[:, 1] - self.cy
        r = torch.sqrt(dx ** 2 + dy ** 2)                     # 到源点的距离
        r_D = r / self.D
        I_city = self.A * torch.cos(r_D) + self.B * (r_D ** 4)
        return I_city

    def clamp_center(self):
        """约束源点坐标在 [-1, 1] 内（训练每步后用）"""
        with torch.no_grad():
            self.cx.clamp_(-1.0, 1.0)
            self.cy.clamp_(-1.0, 1.0)
