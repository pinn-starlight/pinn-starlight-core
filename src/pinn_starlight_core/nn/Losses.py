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

    def backward(self):
        return 2 * (self.I_pred - self.I_obs) / self.I_obs.size(0)


class MSEPhysics:
    def __init__(self) -> None:
        self.I_obs = None
        self.I_pred = None
        self.I_city = None
        self.alpha = None
        self.weight = None
        self.coords = None

    def forward(self, I_obs, I_pred, I_city, alpha, weight, coords):
        self.I_obs = I_obs
        self.I_pred = I_pred
        self.I_city = I_city
        self.alpha = alpha
        self.weight = weight
        self.coords = coords

        return weight * ((laplacian(I_pred, coords) - alpha * I_pred + I_city) ** 2).mean()

    def backward(self):
        f = laplacian(self.I_pred, self.coords) - self.alpha * self.I_pred + self.I_city

        return 2 * self.weight * (laplacian(f, self.coords) - self.alpha * f) / self.I_pred.size(0)