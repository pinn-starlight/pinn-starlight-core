import torch
from torch import autograd


def laplacian(I_pred, point):
    grad = autograd.grad(I_pred, point, grad_outputs=torch.ones_like(I_pred), create_graph=True)[0]
    d_dx = grad[:, 0]
    d_dy = grad[:, 1]

    d2_dx = autograd.grad(d_dx, point, grad_outputs=torch.ones_like(d_dx), create_graph=True)[0][:, 0]
    d2_dy = autograd.grad(d_dy, point, grad_outputs=torch.ones_like(d_dy), create_graph=True)[0][:, 1]

    return d2_dx + d2_dy