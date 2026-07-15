from pinn_starlight_core.utils.PINLaplacian import laplacian

# TODO: 若改为 RGB 训练，优先在 data loss 中比较三通道观测与预测。
# TODO: 物理损失可先只约束亮度/灰度分量，再评估是否需要扩展到逐通道 PDE。
# TODO：避开建筑物

class MSEData:
    def __init__(self):
        self.I_obs = None
        self.I_pred = None

    def forward(self, I_obs, I_pred):
        self.I_obs = I_obs
        self.I_pred = I_pred
        return ((I_pred - I_obs) ** 2).mean()


class MSEPhysics:
    """Screened Poisson: ∇²I - αI + I_city = 0

    齐次解为 K₀ 族（指数衰减），与光污染远离光源后单调衰减的物理直觉一致。
    实验中优于 Helmholtz（+α, Bessel 振荡解），选为最终 PDE 形式。
    """

    def __init__(self):
        self.I_obs = None
        self.I_pred = None
        self.I_city = None
        self.coords = None

    def forward(self, I_obs, I_pred, I_city, alpha, coords):
        self.I_obs = I_obs
        self.I_pred = I_pred
        self.I_city = I_city
        self.coords = coords

        f = laplacian(I_pred, coords) - alpha * I_pred + I_city
        return (f ** 2).mean()

