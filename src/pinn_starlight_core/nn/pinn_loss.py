"""Loss terms for the coordinate PINN.

Stage 10 interprets the source term as signed effective forcing ``S_city``.
It is not a pointwise non-negative physical urban-radiance map.  The physics
term penalizes the residual of the screened Poisson equation

    ΔB − α B + S_city = 0

where ``B`` is the predicted background, ``α`` the screening coefficient and
``S_city`` the effective forcing.
"""

import torch
from torch import autograd


def mse_data(observed, background_pred):
    """Data term: mean squared mismatch between prediction and observation."""
    return ((background_pred - observed) ** 2).mean()


def _pde_residual(background_pred, laplacian_pred, S_city, alpha):
    """Screened-Poisson residual ``ΔB − αB + S_city`` for a known Laplacian."""
    return laplacian_pred - alpha * background_pred + S_city


def mse_physics(background_pred, S_city, alpha, coords):
    """Physics term with the Laplacian differentiated from ``coords`` by autograd."""
    laplacian_pred = laplacian(background_pred, coords)
    return (_pde_residual(background_pred, laplacian_pred, S_city, alpha) ** 2).mean()


def mse_physics_from_laplacian(background_pred, laplacian_pred, S_city, alpha):
    """Physics MSE when an equivalent Laplacian is supplied explicitly."""
    return (_pde_residual(background_pred, laplacian_pred, S_city, alpha) ** 2).mean()


def laplacian(background_pred, coords):
    """Trace of the Hessian of ``background_pred`` with respect to ``coords``.

    ``create_graph=True`` keeps the second derivatives differentiable so the
    physics term can be back-propagated into the network weights.
    """
    grad = autograd.grad(
        background_pred,
        coords,
        grad_outputs=torch.ones_like(background_pred),
        create_graph=True,
    )[0]
    d_dx = grad[:, 0]
    d_dy = grad[:, 1]

    d2_dx = autograd.grad(
        d_dx,
        coords,
        grad_outputs=torch.ones_like(d_dx),
        create_graph=True,
    )[0][:, 0]
    d2_dy = autograd.grad(
        d_dy,
        coords,
        grad_outputs=torch.ones_like(d_dy),
        create_graph=True,
    )[0][:, 1]
    return d2_dx + d2_dy
