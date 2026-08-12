import torch
from torch import autograd


# 若改为 RGB 训练，优先在 data loss 中比较三通道观测与预测。
# 物理损失可先只约束亮度/灰度分量，再评估是否需要扩展到逐通道 PDE。
# 避开建筑物

# I_bg_pred 表示网络估计的光污染背景，而非包含星光的总观测亮度。
# 数据项与物理项均采用均方误差，并通过联合优化平衡数据拟合与背景平滑约束。


def mse_data(I_obs, I_bg_pred):
    return ((I_bg_pred - I_obs) ** 2).mean()


def mse_physics(I_bg_pred, I_city, alpha, coords):
    pde_residual = laplacian(I_bg_pred, coords) - alpha * I_bg_pred + I_city
    return (pde_residual ** 2).mean()


def laplacian(I_bg_pred, point):
    grad = autograd.grad(I_bg_pred, point, grad_outputs=torch.ones_like(I_bg_pred), create_graph=True)[0]
    d_dx = grad[:, 0]
    d_dy = grad[:, 1]

    d2_dx = autograd.grad(d_dx, point, grad_outputs=torch.ones_like(d_dx), create_graph=True)[0][:, 0]
    d2_dy = autograd.grad(d_dy, point, grad_outputs=torch.ones_like(d_dy), create_graph=True)[0][:, 1]

    return d2_dx + d2_dy
