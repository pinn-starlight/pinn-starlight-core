"""E0-E4 shared Stage-10 coordinate PINN background estimator.

The network predicts the light-pollution background B(x,y).  Its screened-
Poisson auxiliary loss uses a sparse mixture of up to K signed *effective
forcing* templates S_city.  The forcing geometry is initialized from the
observed image only; synthetic truth is never passed across the Method boundary.
"""

from __future__ import annotations

import copy
import math
import os
import time
from typing import TypedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import pinn_starlight_core.nn.pinn_loss as loss
from experiments.common.utils import experiment_utils as utils
from pinn_starlight_core.data.image_loader import coordinate_grid
from pinn_starlight_core.nn.physics_model import EffectiveForcingMixture
from pinn_starlight_core.nn.pinn_layers import SkyglowMLP
from pinn_starlight_core.physics.forcing import candidate_centers_from_observed


class PINNConfig(TypedDict, total=False):
    hidden_dims: list[int]
    physics_weight: float
    sparse_weight: float
    steps: int
    batch_size: int
    model_lr: float
    forcing_lr: float
    alpha: float
    center_warmup_steps: int
    center_refinement_enabled: bool
    forcing_max_components: int
    forcing_sigma_bounds: list[float]
    forcing_theta_bounds: list[float]
    forcing_winsor_percentile: float
    forcing_low_pass_sigma_fraction: float
    forcing_min_center_distance: float
    forcing_center_refinement: float
    forcing_initial_weights: list[float]
    forcing_initial_sigma: float
    forcing_rms_grid_size: int
    forcing_rms_epsilon: float
    inactive_relative_threshold: float
    prediction_batch_size: int
    log_every: int


DEFAULT_CONFIG: PINNConfig = {
    "hidden_dims": [128, 128],
    "physics_weight": 0.5,
    "sparse_weight": 1e-3,
    "steps": 10000,
    "batch_size": 8192,
    "model_lr": 1e-3,
    "forcing_lr": 1e-3,
    "alpha": 0.5,
    "center_warmup_steps": 1000,
    "center_refinement_enabled": True,
    "forcing_max_components": 3,
    "forcing_sigma_bounds": [0.05, 0.55],
    "forcing_theta_bounds": [-0.5 * math.pi, 0.5 * math.pi],
    "forcing_winsor_percentile": 99.5,
    "forcing_low_pass_sigma_fraction": 0.04,
    "forcing_min_center_distance": 0.30,
    "forcing_center_refinement": 0.08,
    "forcing_initial_weights": [1.0, 0.05, 0.05],
    "forcing_initial_sigma": 0.425,
    "forcing_rms_grid_size": 64,
    "forcing_rms_epsilon": 1e-8,
    "inactive_relative_threshold": 0.05,
    "prediction_batch_size": 65536,
    "log_every": 100,
}


def train_background(
    observed,
    config=None,
    device=None,
    seed=20260728,
    resume_state=None,
    show_progress=True,
    return_state=False,
    laplacian_backend=None,
    initial_centers=None,
):
    """Optimize one image and return B_hat plus diagnostics/resume state."""
    config = normalized_config(config)
    if config["alpha"] != 0.5:
        raise ValueError("Stage-10 formal PINN requires alpha fixed at 0.5")

    utils.set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    laplacian_backend = resolve_laplacian_backend(laplacian_backend)
    utils.reset_peak_vram(device)
    observed = utils.load_observed(observed)
    height, width = observed.shape
    coords = coordinate_grid(height, width, device)
    values = torch.from_numpy(observed.reshape(-1)).to(device)

    model = SkyglowMLP(config["hidden_dims"]).to(device)
    forcing = _build_forcing(
        observed,
        config,
        device,
        initial_centers=initial_centers,
    ).to(device)
    optimizer = _optimizer(model, forcing, config)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    start_step, history, elapsed_before = _restore_resume_state(
        model, forcing, optimizer, generator, resume_state
    )

    alpha = torch.tensor(float(config["alpha"]), dtype=torch.float32, device=device)
    started = time.perf_counter()
    history, final_total, final_data, final_physics, final_sparse = _train_steps(
        model,
        forcing,
        coords,
        values,
        optimizer,
        generator,
        alpha,
        config,
        start_step,
        history,
        show_progress,
        laplacian_backend,
    )
    runtime_s = elapsed_before + time.perf_counter() - started
    background_pred = _predict_background(
        model,
        coords,
        height,
        width,
        int(config["prediction_batch_size"]),
    )
    final_step = start_step + int(config["steps"])
    components = forcing.diagnostics(
        alpha,
        inactive_relative_threshold=float(config["inactive_relative_threshold"]),
    )
    strongest = max(components, key=lambda item: item["weight"])

    result = {
        # Formal Method contract: return B_hat and diagnostics only.  Residual
        # reconstruction remains in the common experiment layer.
        "background_pred": background_pred,
        "history": history,
        "step": final_step,
        "runtime_s": float(runtime_s),
        "peak_vram_mb": utils.peak_vram_mb(device),
        "total_loss": final_total,
        "data_loss": final_data,
        "physics_loss": final_physics,
        "sparse_loss": final_sparse,
        "alpha": float(config["alpha"]),
        "components": components,
        "effective_component_count": int(sum(item["active"] for item in components)),
        # Compatibility scalar aliases: strongest effective component only.
        "center_x": float(strongest["center_x"]),
        "center_y": float(strongest["center_y"]),
        "sigma_x": float(strongest["sigma_x"]),
        "sigma_y": float(strongest["sigma_y"]),
        "theta": float(strongest["theta"]),
        "forcing_weight": float(strongest["weight"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters())
        + sum(parameter.numel() for parameter in forcing.parameters()),
        "config": config,
        "seed": int(seed),
        "laplacian_backend": laplacian_backend,
    }
    if return_state:
        result["state"] = _capture_resume_state(
            model, forcing, optimizer, generator, final_step, history, runtime_s
        )
    return result


def _restore_resume_state(model, forcing, optimizer, generator, resume_state):
    """Reload a checkpoint so a run continues exactly where it stopped."""
    if resume_state is None:
        return 0, [], 0.0

    model.load_state_dict(resume_state["model_state"])
    forcing.load_state_dict(resume_state["forcing_state"])
    # torch's Optimizer.load_state_dict keeps references to the incoming
    # momentum buffers and updates them in place, so a plain load would mutate
    # the caller's checkpoint.  Copy first so a snapshot stays reusable.
    optimizer.load_state_dict(copy.deepcopy(resume_state["optimizer_state"]))
    generator.set_state(resume_state["generator_state"])
    return (
        int(resume_state["step"]),
        list(resume_state.get("history", [])),
        float(resume_state.get("runtime_s", 0.0)),
    )


def _capture_resume_state(model, forcing, optimizer, generator, step, history, runtime_s):
    """Snapshot everything a later run needs in order to resume."""
    return _to_cpu(
        {
            "model_state": model.state_dict(),
            "forcing_state": forcing.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "step": step,
            "history": history,
            "runtime_s": runtime_s,
        }
    )


def initial_centers_from_observed(observed, config=None):
    """Compute the formal observed-only forcing-center initialization once."""
    config = normalized_config(config)
    return candidate_centers_from_observed(
        utils.load_observed(observed),
        max_components=int(config["forcing_max_components"]),
        winsor_percentile=float(config["forcing_winsor_percentile"]),
        low_pass_sigma_fraction=float(config["forcing_low_pass_sigma_fraction"]),
        min_center_distance=float(config["forcing_min_center_distance"]),
    )


def _build_forcing(observed, config: PINNConfig, device, *, initial_centers=None):
    sigma_min, sigma_max = (float(value) for value in config["forcing_sigma_bounds"])
    theta_min, theta_max = (float(value) for value in config["forcing_theta_bounds"])
    if not np.isclose(theta_min, -theta_max, rtol=0.0, atol=1e-12):
        raise ValueError("PINN forcing theta bounds must be symmetric around zero")
    return EffectiveForcingMixture(
        device=device,
        gray_img=observed,
        max_components=int(config["forcing_max_components"]),
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        theta_bound=theta_max,
        winsor_percentile=float(config["forcing_winsor_percentile"]),
        low_pass_sigma_fraction=float(config["forcing_low_pass_sigma_fraction"]),
        min_center_distance=float(config["forcing_min_center_distance"]),
        center_refinement=float(config["forcing_center_refinement"]),
        center_refinement_enabled=bool(config["center_refinement_enabled"]),
        initial_weights=tuple(float(value) for value in config["forcing_initial_weights"]),
        initial_sigma=float(config["forcing_initial_sigma"]),
        initial_centers=initial_centers,
        rms_grid_size=int(config["forcing_rms_grid_size"]),
        rms_epsilon=float(config["forcing_rms_epsilon"]),
    )


def _train_steps(
    model,
    forcing,
    coords,
    values,
    optimizer,
    generator,
    alpha,
    config: PINNConfig,
    start_step,
    history,
    show_progress,
    laplacian_backend,
):
    steps = int(config["steps"])
    batch_size = min(int(config["batch_size"]), coords.shape[0])
    log_every = max(1, int(config["log_every"]))
    progress_bar = tqdm(range(steps), desc="PINN", leave=False) if show_progress else None
    progress = progress_bar if progress_bar is not None else range(steps)

    final_total = final_data = final_physics = final_sparse = float("nan")
    model.train()
    forcing.train()
    for local_step in progress:
        global_step = start_step + local_step + 1
        forcing.set_center_gradient_enabled(
            bool(config["center_refinement_enabled"])
            and global_step > int(config["center_warmup_steps"])
        )
        indices = torch.randint(
            0,
            coords.shape[0],
            (batch_size,),
            device=coords.device,
            generator=generator,
        )
        batch_xy = coords[indices]
        batch_observed = values[indices]
        if float(config["physics_weight"]) > 0.0 and laplacian_backend == "analytic":
            background_pred, laplacian_pred = _background_and_laplacian(model, batch_xy)
        else:
            if float(config["physics_weight"]) > 0.0:
                batch_xy = batch_xy.clone().requires_grad_(True)
            background_pred = torch.sigmoid(model(batch_xy).squeeze(-1))
            laplacian_pred = None
        data_loss = loss.mse_data(batch_observed, background_pred)

        if float(config["physics_weight"]) > 0.0:
            forcing_pred = forcing(batch_xy, alpha)
            if laplacian_backend == "analytic":
                physics_loss = loss.mse_physics_from_laplacian(
                    background_pred,
                    laplacian_pred,
                    forcing_pred,
                    alpha,
                )
            else:
                physics_loss = loss.mse_physics(
                    background_pred,
                    forcing_pred,
                    alpha,
                    batch_xy,
                )
            sparse_loss = forcing.sparse_penalty()
        else:
            physics_loss = torch.zeros((), device=coords.device)
            sparse_loss = torch.zeros((), device=coords.device)

        total_loss = (
            data_loss
            + float(config["physics_weight"]) * physics_loss
            + float(config["sparse_weight"]) * sparse_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        should_log = global_step == 1 or global_step % log_every == 0 or local_step + 1 == steps
        if should_log:
            # Converting a CUDA scalar to Python synchronizes the device.  Keep
            # those synchronization points at the already-frozen history cadence
            # instead of forcing four synchronizations on every optimizer step.
            final_total = float(total_loss.detach().item())
            final_data = float(data_loss.detach().item())
            final_physics = float(physics_loss.detach().item())
            final_sparse = float(sparse_loss.detach().item())
            history.append(
                _history_row(
                    global_step,
                    final_total,
                    final_data,
                    final_physics,
                    final_sparse,
                    forcing,
                    alpha,
                    config,
                )
            )
            if progress_bar is not None:
                progress_bar.set_postfix(loss=f"{final_total:.6f}")

    return history, final_total, final_data, final_physics, final_sparse


def resolve_laplacian_backend(value=None):
    requested = (
        str(
            value
            if value is not None
            else os.environ.get("PINN_STARLIGHT_PINN_LAPLACIAN", "autograd")
        )
        .strip()
        .lower()
    )
    aliases = {
        "chain_rule": "analytic",
        "analytic_chain_rule": "analytic",
        "autodiff": "autograd",
        "reference": "autograd",
    }
    requested = aliases.get(requested, requested)
    if requested not in {"analytic", "autograd"}:
        raise ValueError("PINN laplacian backend must be one of: analytic, autograd")
    return requested


def _background_and_laplacian(model, coords):
    """Evaluate sigmoid(MLP(x,y)) and its exact 2-D Laplacian by chain rule.

    This is mathematically equivalent to nested coordinate autograd for the
    formal Linear/Tanh coordinate MLP, but avoids constructing second-order
    autograd graphs with respect to the sampled coordinates.  Parameter
    gradients remain fully differentiable because every chain-rule operation
    is expressed with ordinary PyTorch tensor operations.
    """
    value = coords
    d_dx = torch.zeros_like(coords)
    d_dy = torch.zeros_like(coords)
    lap = torch.zeros_like(coords)
    d_dx[:, 0] = 1.0
    d_dy[:, 1] = 1.0

    for layer in model:
        if isinstance(layer, nn.Linear):
            value = F.linear(value, layer.weight, layer.bias)
            d_dx = F.linear(d_dx, layer.weight, None)
            d_dy = F.linear(d_dy, layer.weight, None)
            lap = F.linear(lap, layer.weight, None)
            continue
        if isinstance(layer, nn.Tanh):
            activated = torch.tanh(value)
            first = 1.0 - activated.square()
            second = -2.0 * activated * first
            lap = second * (d_dx.square() + d_dy.square()) + first * lap
            d_dx = first * d_dx
            d_dy = first * d_dy
            value = activated
            continue
        raise TypeError(
            "analytic PINN Laplacian supports only nn.Linear and nn.Tanh layers; "
            f"got {type(layer).__name__}"
        )

    logits = value.squeeze(-1)
    dlogits_dx = d_dx.squeeze(-1)
    dlogits_dy = d_dy.squeeze(-1)
    logits_lap = lap.squeeze(-1)
    background = torch.sigmoid(logits)
    first = background * (1.0 - background)
    second = first * (1.0 - 2.0 * background)
    background_lap = second * (dlogits_dx.square() + dlogits_dy.square()) + first * logits_lap
    return background, background_lap


def normalized_config(config) -> PINNConfig:
    result: PINNConfig = dict(DEFAULT_CONFIG)
    if config is not None:
        result.update(config)

    result["hidden_dims"] = [int(value) for value in result["hidden_dims"]]
    for key in (
        "steps",
        "batch_size",
        "center_warmup_steps",
        "forcing_max_components",
        "forcing_rms_grid_size",
        "prediction_batch_size",
        "log_every",
    ):
        result[key] = int(result[key])
    for key in (
        "physics_weight",
        "sparse_weight",
        "model_lr",
        "forcing_lr",
        "alpha",
        "forcing_winsor_percentile",
        "forcing_low_pass_sigma_fraction",
        "forcing_min_center_distance",
        "forcing_center_refinement",
        "forcing_initial_sigma",
        "forcing_rms_epsilon",
        "inactive_relative_threshold",
    ):
        result[key] = float(result[key])
    result["forcing_sigma_bounds"] = [float(value) for value in result["forcing_sigma_bounds"]]
    result["forcing_theta_bounds"] = [float(value) for value in result["forcing_theta_bounds"]]
    result["forcing_initial_weights"] = [
        float(value) for value in result["forcing_initial_weights"]
    ]
    result["center_refinement_enabled"] = bool(result["center_refinement_enabled"])

    if result["steps"] <= 0 or result["batch_size"] <= 0:
        raise ValueError("steps and batch_size must be positive")
    if result["physics_weight"] < 0 or result["sparse_weight"] < 0:
        raise ValueError("physics_weight and sparse_weight must be non-negative")
    if result["center_warmup_steps"] < 0:
        raise ValueError("center_warmup_steps must be non-negative")
    if result["forcing_max_components"] <= 0:
        raise ValueError("forcing_max_components must be positive")
    if len(result["forcing_initial_weights"]) < result["forcing_max_components"]:
        raise ValueError("forcing_initial_weights must cover forcing_max_components")
    sigma_bounds = result["forcing_sigma_bounds"]
    if len(sigma_bounds) != 2 or not 0.0 < sigma_bounds[0] < sigma_bounds[1]:
        raise ValueError("invalid forcing_sigma_bounds")
    if not sigma_bounds[0] < result["forcing_initial_sigma"] < sigma_bounds[1]:
        raise ValueError("forcing_initial_sigma must lie strictly inside sigma bounds")
    theta_bounds = result["forcing_theta_bounds"]
    if len(theta_bounds) != 2 or not theta_bounds[0] < 0.0 < theta_bounds[1]:
        raise ValueError("invalid forcing_theta_bounds")
    if not np.isclose(theta_bounds[0], -theta_bounds[1], rtol=0.0, atol=1e-12):
        raise ValueError("forcing_theta_bounds must be symmetric")
    if not 0.0 < result["inactive_relative_threshold"] <= 1.0:
        raise ValueError("inactive_relative_threshold must lie in (0,1]")
    return result


def _to_cpu(value):
    """Deep-copy a nested state structure onto the CPU.

    ``Tensor.cpu()`` is a no-op for tensors that already live on the CPU, so
    the clone is what makes the snapshot independent: without it a returned
    ``state`` aliases the live optimizer/model tensors and reusing it for a
    second run silently produces different results.
    """
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return value


def _optimizer(model, forcing, config: PINNConfig):
    groups = [{"params": model.parameters(), "lr": float(config["model_lr"])}]
    forcing_parameters = [
        parameter for parameter in forcing.parameters() if parameter.requires_grad
    ]
    if float(config["physics_weight"]) > 0.0 and forcing_parameters:
        groups.append({"params": forcing_parameters, "lr": float(config["forcing_lr"])})
    return torch.optim.Adam(groups)


def _predict_background(model, coords, height, width, batch_size):
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            batch = coords[start : start + batch_size]
            predictions.append(torch.sigmoid(model(batch).squeeze(-1)).cpu())
    return torch.cat(predictions).reshape(height, width).numpy().astype(np.float32)


def _history_row(step, total, data, physics, sparse, forcing, alpha, config):
    components = forcing.diagnostics(
        alpha,
        inactive_relative_threshold=float(config["inactive_relative_threshold"]),
    )
    strongest = max(components, key=lambda item: item["weight"])
    row = {
        "step": int(step),
        "total_loss": float(total),
        "data_loss": float(data),
        "physics_loss": float(physics),
        "sparse_loss": float(sparse),
        "center_refinement_enabled_at_step": bool(
            config["center_refinement_enabled"] and int(step) > int(config["center_warmup_steps"])
        ),
        "effective_component_count": int(sum(item["active"] for item in components)),
        "center_x": float(strongest["center_x"]),
        "center_y": float(strongest["center_y"]),
        "sigma_x": float(strongest["sigma_x"]),
        "sigma_y": float(strongest["sigma_y"]),
        "theta": float(strongest["theta"]),
        "forcing_weight": float(strongest["weight"]),
    }
    for item in components:
        prefix = f"component_{item['index']}"
        for key in (
            "center_x",
            "center_y",
            "sigma_x",
            "sigma_y",
            "theta",
            "weight",
            "relative_weight",
            "negative_energy_fraction",
        ):
            row[f"{prefix}_{key}"] = item[key]
        row[f"{prefix}_active"] = item["active"]
    return row
