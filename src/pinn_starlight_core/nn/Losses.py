from pinn_starlight_core.utils.PINLaplacian import laplacian

# TODO: 若改为 RGB 训练，优先在 data loss 中比较三通道观测与预测。
# TODO: 物理损失可先只约束亮度/灰度分量，再评估是否需要扩展到逐通道 PDE。
# TODO：避开建筑物

def mse_data(I_obs, I_pred):
    return ((I_pred - I_obs) ** 2).mean()


def mse_physics(I_obs, I_pred, I_city, alpha, coords):
    pde_residual = laplacian(I_pred, coords) - alpha * I_pred + I_city
    return (pde_residual ** 2).mean()
