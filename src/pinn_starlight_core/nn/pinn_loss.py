"""Loss terms for the coordinate PINN.

Stage 10 interprets the source term as signed effective forcing ``S_city``.
It is not a pointwise non-negative physical urban-radiance map.
"""

import torch
from torch import autograd


def mse_data(I_obs, I_bg_pred):
    return ((I_bg_pred - I_obs) ** 2).mean()


def mse_physics(I_bg_pred, S_city, alpha, coords):
    pde_residual = laplacian(I_bg_pred, coords) - alpha * I_bg_pred + S_city
    return (pde_residual ** 2).mean()



def mse_physics_from_laplacian(I_bg_pred, laplacian_pred, S_city, alpha):
    """Physics MSE when an equivalent Laplacian is supplied explicitly."""
    pde_residual = laplacian_pred - alpha * I_bg_pred + S_city
    return (pde_residual ** 2).mean()


def sparse_effective_forcing(weights):
    """Mean non-negative component amplitude used by the Stage-10 L1 penalty."""
    values = torch.as_tensor(weights)
    return values.mean()


def laplacian(I_bg_pred, point):
    grad = autograd.grad(
        I_bg_pred,
        point,
        grad_outputs=torch.ones_like(I_bg_pred),
        create_graph=True,
    )[0]
    d_dx = grad[:, 0]
    d_dy = grad[:, 1]

    d2_dx = autograd.grad(
        d_dx,
        point,
        grad_outputs=torch.ones_like(d_dx),
        create_graph=True,
    )[0][:, 0]
    d2_dy = autograd.grad(
        d_dy,
        point,
        grad_outputs=torch.ones_like(d_dy),
        create_graph=True,
    )[0][:, 1]
    return d2_dx + d2_dy
