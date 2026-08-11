import torch
from torch import autograd


# 若改为 RGB 训练，优先在 data loss 中比较三通道观测与预测。
# 物理损失可先只约束亮度/灰度分量，再评估是否需要扩展到逐通道 PDE。
# 避开建筑物

# TODO: 当前数据损失直接令 I_pred 拟合 I_obs，
# 但 I_obs = I_star + I_up，而 I_pred 仅表示 I_up。
# 后续需引入星光残差模型、稀疏先验或鲁棒数据损失。

def mse_data(I_obs, I_pred):
    return ((I_pred - I_obs) ** 2).mean()


def mse_physics(I_obs, I_pred, I_city, alpha, coords):
    pde_residual = laplacian(I_pred, coords) - alpha * I_pred + I_city
    return (pde_residual ** 2).mean()


def laplacian(I_pred, point):
    grad = autograd.grad(I_pred, point, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    d_dx = grad[:, 0]
    d_dy = grad[:, 1]

    d2_dx = autograd.grad(d_dx, point, grad_outputs=torch.ones_like(d_dx), create_graph=True)[0][:, 0]
    d2_dy = autograd.grad(d_dy, point, grad_outputs=torch.ones_like(d_dy), create_graph=True)[0][:, 1]

    return d2_dx + d2_dy