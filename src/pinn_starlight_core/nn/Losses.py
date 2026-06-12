"""损失函数 — Screened Poisson (实验优选, 2026-06)"""
import torch
from pinn_starlight_core.utils.Laplacian import laplacian


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
        self.alpha = None
        self.coords = None

    def forward(self, I_obs, I_pred, I_city, alpha,coords):
        self.I_obs = I_obs
        self.I_pred = I_pred
        self.I_city = I_city
        self.alpha = alpha
        self.coords = coords

        f = laplacian(I_pred, coords) - alpha * I_pred + I_city
        return (f ** 2).mean()

