import torch
from pinn_starlight_core.utils.laplacian import laplacian

class MSEData:
    def __init__(self) -> None:
        self.I_obs = None
        self.I_pred = None

    def forward(self, I_obs, I_pred) -> torch.Tensor:
        self.I_obs = I_obs
        self.I_pred = I_pred
        return ((I_pred - I_obs) ** 2).mean()

    def backward(self) -> torch.Tensor:
        return 2 * (self.I_pred - self.I_obs) / self.I_obs.size(0)


class MSEPhysics:
    def __init__(self) -> None:
        self.I_obs = None
        self.I_pred = None
        self.I_city = None
        self.alpha = None
        self.weight = None
        self.point = None

    def forward(self, I_obs, I_pred, I_city, alpha, weight, point) -> torch.Tensor:
        self.I_obs = I_obs
        self.I_pred = I_pred
        self.I_city = I_city
        self.alpha = alpha
        self.weight = weight
        self.point = point

        return weight * ((laplacian(I_pred, point) - alpha * I_pred + I_city) ** 2).mean()

    def backward(self) -> torch.Tensor:
        f = laplacian(self.I_pred, self.point) - self.alpha * self.I_pred + self.I_city
        return 2 * self.weight * (laplacian(f, self.point) - self.alpha * f) / self.I_pred.size(0)