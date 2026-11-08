"""Stage11 final decoupled gradient-coordinated PINN.

Formal follow-up method
-----------------------
This module contains only the final frozen method used by Stage11 confirmation
runs.  It does not import or depend on research_diagnostics.

Pipeline:
1) MSE data-only coordinate MLP -> B0.
2) Freeze B0; fit fixed-geometry effective forcing amplitudes to
       Q0 = alpha * B0 - Laplacian(B0)
   with non-negative least squares.
3) Freeze forcing; fine-tune the background network.  The regularizer gradient
   is projected away from directions that conflict with the data gradient, then
   scaled so its actual norm contribution tracks rho * ||g_data||.

A matched-compute data-only fine-tune control is produced from the same Stage-1
state and the same Stage-3 random batch sequence.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from scipy.optimize import nnls
from tqdm import tqdm

import pinn_starlight_core.nn.pinn_loss as pinn_loss
from experiments.common.baselines import coordinate_pinn as base_pinn
from experiments.common.utils import experiment_utils as utils
from pinn_starlight_core.data.image_loader import coordinate_grid
from pinn_starlight_core.nn.pinn_layers import SkyglowMLP

CONTROL_NAME = "data_finetune_control"
METHOD_NAME = "pinn_decoupled_gradient"


def normalized_followup_config(config: dict) -> dict:
    cfg = dict(config)
    for key in ("stage1", "stage2", "stage3", "test_seeds"):
        if key not in cfg:
            raise ValueError(f"final PINN config missing {key}")
    stage1 = dict(cfg["stage1"])
    stage2 = dict(cfg["stage2"])
    stage3 = dict(cfg["stage3"])

    if float(stage1["alpha"]) != 0.5:
        raise ValueError("Stage11 final PINN requires alpha=0.5")
    if float(stage1.get("physics_weight", 0.0)) != 0.0:
        raise ValueError("Stage11 final PINN Stage 1 must be data-only")
    if float(stage1.get("sparse_weight", 0.0)) != 0.0:
        raise ValueError("Stage11 final PINN Stage 1 sparse weight must be zero")
    if bool(stage2.get("geometry_refinement", False)):
        raise ValueError("Stage11 final PINN freezes forcing geometry")
    if bool(stage3.get("neumann_boundary_loss", False)):
        raise ValueError("Stage11 final PINN has no PINN Neumann boundary loss")
    if not bool(stage3.get("conflict_projection", False)):
        raise ValueError("Stage11 final PINN requires conflict projection")
    if float(stage3["lambda_min"]) != 0.0:
        raise ValueError("Stage11 exact gradient balance requires lambda_min=0")
    if not (0.0 < float(stage3["rho"]) <= 1.0):
        raise ValueError("rho must be in (0,1]")
    if int(stage1["steps"]) <= 0 or int(stage3["steps"]) <= 0:
        raise ValueError("training steps must be positive")
    if not (0.0 < float(stage2["fit_fraction"]) < 1.0):
        raise ValueError("Stage-2 fit_fraction must be in (0,1)")

    cfg["stage1"] = stage1
    cfg["stage2"] = stage2
    cfg["stage3"] = stage3
    cfg["test_seeds"] = [int(x) for x in cfg["test_seeds"]]
    return cfg


def run_paired(
    observed: np.ndarray,
    *,
    config: dict,
    device: torch.device,
    seed: int,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run the matched data-only control and final improved PINN on one image."""
    cfg = normalized_followup_config(config)
    observed = utils.load_observed(observed)
    stage1_cfg = dict(base_pinn.normalized_config(cfg["stage1"]))
    initial_centers = base_pinn.initial_centers_from_observed(observed, stage1_cfg)

    # Stage 1: one shared data-only state.
    stage1 = base_pinn.train_background(
        observed,
        config=stage1_cfg,
        device=device,
        seed=int(seed),
        show_progress=show_progress,
        return_state=True,
        laplacian_backend="analytic",
        initial_centers=initial_centers,
    )
    model_state = stage1["state"]["model_state"]

    # Stage 2: identify fixed forcing from the shared B0.
    forcing_started = time.perf_counter()
    forcing_state, forcing_diag = fit_fixed_forcing(
        observed=observed,
        model_state=model_state,
        stage1_config=stage1_cfg,
        stage2_config=cfg["stage2"],
        initial_centers=initial_centers,
        device=device,
        seed=int(seed) + 1009,
    )
    forcing_runtime_s = time.perf_counter() - forcing_started

    # Stage 3: same starting network and same generator seed.
    stage3_seed = int(seed) + 2003
    control = finetune(
        observed=observed,
        model_state=model_state,
        forcing_state=None,
        stage1_config=stage1_cfg,
        stage3_config=cfg["stage3"],
        initial_centers=initial_centers,
        device=device,
        seed=stage3_seed,
        use_physics=False,
        show_progress=show_progress,
    )
    improved = finetune(
        observed=observed,
        model_state=model_state,
        forcing_state=forcing_state,
        stage1_config=stage1_cfg,
        stage3_config=cfg["stage3"],
        initial_centers=initial_centers,
        device=device,
        seed=stage3_seed,
        use_physics=True,
        show_progress=show_progress,
    )

    model_params = int(
        sum(v.numel() for k, v in model_state.items() if torch.is_tensor(v))
    )
    forcing_params = int(
        sum(v.numel() for k, v in forcing_state.items() if torch.is_tensor(v))
    )

    shared = {
        "stage1_background_pred": np.asarray(stage1["background_pred"], dtype=np.float32),
        "stage1_history": list(stage1["history"]),
        "stage1_runtime_s": float(stage1["runtime_s"]),
        "stage1_model_state": model_state,
        "forcing_state": forcing_state,
        "forcing_diagnostics": forcing_diag,
        "forcing_runtime_s": float(forcing_runtime_s),
        "model_parameter_count": model_params,
        "forcing_parameter_count": forcing_params,
        "config": cfg,
        "seed": int(seed),
    }

    control.update(
        {
            "method": CONTROL_NAME,
            "runtime_s_total": float(stage1["runtime_s"] + control["stage3_runtime_s"]),
            "parameter_count": model_params,
        }
    )
    improved.update(
        {
            "method": METHOD_NAME,
            "runtime_s_total": float(
                stage1["runtime_s"] + forcing_runtime_s + improved["stage3_runtime_s"]
            ),
            "parameter_count": model_params + forcing_params,
        }
    )
    return {
        "shared": shared,
        CONTROL_NAME: control,
        METHOD_NAME: improved,
    }


def fit_fixed_forcing(
    *,
    observed,
    model_state,
    stage1_config,
    stage2_config,
    initial_centers,
    device,
    seed,
):
    height, width = observed.shape
    all_coords = coordinate_grid(height, width, device)
    n_total = int(all_coords.shape[0])
    n_pick = min(int(stage2_config["prefit_points"]), n_total)

    rng = np.random.default_rng(int(seed))
    selected_np = rng.choice(n_total, size=n_pick, replace=False)
    rng.shuffle(selected_np)
    n_fit = int(round(float(stage2_config["fit_fraction"]) * n_pick))
    n_fit = min(max(n_fit, 3), n_pick - 1)

    selected = torch.as_tensor(selected_np, dtype=torch.long, device=device)
    coords = all_coords[selected]

    model = SkyglowMLP(stage1_config["hidden_dims"]).to(device)
    model.load_state_dict(model_state)
    model.eval()
    alpha = torch.tensor(
        float(stage1_config["alpha"]), dtype=torch.float32, device=device
    )

    q_parts = []
    chunk = 8192
    with torch.no_grad():
        for start in range(0, coords.shape[0], chunk):
            xy = coords[start : start + chunk]
            background, laplacian = base_pinn._background_and_laplacian(model, xy)
            q_parts.append((alpha * background - laplacian).detach())
    q_all = torch.cat(q_parts, dim=0)

    forcing = base_pinn._build_forcing(
        observed,
        stage1_config,
        device,
        initial_centers=initial_centers,
    ).to(device)
    forcing.set_center_gradient_enabled(False)

    with torch.no_grad():
        components = forcing._unnormalized_components(coords, alpha)
        basis = components / forcing.component_rms(alpha).unsqueeze(0)

    a_fit = basis[:n_fit].detach().cpu().double().numpy()
    y_fit = q_all[:n_fit].detach().cpu().double().numpy()
    a_check = basis[n_fit:].detach().cpu().double().numpy()
    y_check = q_all[n_fit:].detach().cpu().double().numpy()

    weights, residual_norm = nnls(a_fit, y_fit)
    _set_forcing_weights(forcing, weights)

    fit_pred = a_fit @ weights
    check_pred = a_check @ weights

    diagnostics = {
        "mode": "fixed_geometry_nonnegative_least_squares",
        "prefit_points": int(n_pick),
        "fit_points": int(n_fit),
        "diagnostic_check_points": int(n_pick - n_fit),
        "weights": [float(x) for x in weights],
        "nnls_residual_norm": float(residual_norm),
        "fit_rmse": _rmse(y_fit, fit_pred),
        "diagnostic_check_rmse": _rmse(y_check, check_pred),
        "fit_relative_rmse": _relative_rmse(y_fit, fit_pred),
        "diagnostic_check_relative_rmse": _relative_rmse(y_check, check_pred),
        "diagnostic_check_correlation": _safe_corr(y_check, check_pred),
        "uses_true_background": False,
        "geometry_refined": False,
    }

    forcing.set_center_gradient_enabled(False)
    for parameter in forcing.parameters():
        parameter.requires_grad_(False)

    component_diag = forcing.diagnostics(
        alpha,
        inactive_relative_threshold=float(
            stage1_config["inactive_relative_threshold"]
        ),
    )
    diagnostics["components"] = component_diag
    state = _cpu_state(forcing.state_dict())
    return state, diagnostics


def finetune(
    *,
    observed,
    model_state,
    forcing_state,
    stage1_config,
    stage3_config,
    initial_centers,
    device,
    seed,
    use_physics,
    show_progress,
):
    utils.set_seed(int(seed))
    utils.reset_peak_vram(device)

    height, width = observed.shape
    coords = coordinate_grid(height, width, device)
    values = torch.from_numpy(observed.reshape(-1)).to(device)

    model = SkyglowMLP(stage1_config["hidden_dims"]).to(device)
    model.load_state_dict(model_state)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    alpha = torch.tensor(
        float(stage1_config["alpha"]), dtype=torch.float32, device=device
    )

    forcing = None
    if use_physics:
        forcing = base_pinn._build_forcing(
            observed,
            stage1_config,
            device,
            initial_centers=initial_centers,
        ).to(device)
        forcing.load_state_dict(forcing_state)
        forcing.set_center_gradient_enabled(False)
        forcing.eval()
        for parameter in forcing.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(params, lr=float(stage3_config["model_lr"]))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    steps = int(stage3_config["steps"])
    batch_size = min(int(stage1_config["batch_size"]), int(coords.shape[0]))
    log_every = max(1, int(stage3_config["log_every"]))
    rho = float(stage3_config["rho"])
    lambda_min = float(stage3_config["lambda_min"])
    lambda_max = float(stage3_config["lambda_max"])
    ramp_steps = max(1, int(stage3_config["ramp_steps"]))

    history = []
    conflicts = 0
    logged_physics = 0
    lambda_values = []
    scaled_ratio_values = []
    target_ratio_values = []
    balance_error_values = []

    started = time.perf_counter()
    iterator = range(1, steps + 1)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc=(METHOD_NAME if use_physics else CONTROL_NAME),
            leave=False,
        )

    final_data = final_physics = float("nan")
    final_lambda = 0.0

    for step in iterator:
        indices = torch.randint(
            0,
            coords.shape[0],
            (batch_size,),
            device=device,
            generator=generator,
        )
        batch_xy = coords[indices]
        batch_target = values[indices]

        if not use_physics:
            background = torch.sigmoid(model(batch_xy).squeeze(-1))
            data_loss = pinn_loss.mse_data(batch_target, background)
            optimizer.zero_grad(set_to_none=True)
            data_loss.backward()
            optimizer.step()

            if step == 1 or step % log_every == 0 or step == steps:
                final_data = float(data_loss.detach().item())
                history.append(
                    {
                        "step": step,
                        "data_loss": final_data,
                        "physics_loss": 0.0,
                        "data_grad_norm": float("nan"),
                        "raw_physics_grad_norm": 0.0,
                        "used_physics_grad_norm": 0.0,
                        "raw_grad_cosine": float("nan"),
                        "lambda_use": 0.0,
                        "target_scaled_physics_to_data_ratio": 0.0,
                        "scaled_physics_to_data_ratio": 0.0,
                        "projection_applied": False,
                    }
                )
            continue

        background, laplacian = base_pinn._background_and_laplacian(model, batch_xy)
        data_loss = pinn_loss.mse_data(batch_target, background)
        forcing_pred = forcing(batch_xy, alpha)
        physics_loss = pinn_loss.mse_physics_from_laplacian(
            background, laplacian, forcing_pred, alpha
        )

        data_grads = torch.autograd.grad(
            data_loss, params, retain_graph=True, create_graph=False
        )
        physics_grads = torch.autograd.grad(
            physics_loss, params, retain_graph=False, create_graph=False
        )

        data_norm_sq = _sum_sq(data_grads)
        raw_phys_norm_sq = _sum_sq(physics_grads)
        dot = _dot(data_grads, physics_grads)
        eps = torch.tensor(1e-24, dtype=torch.float64, device=device)

        data_norm = torch.sqrt(data_norm_sq + eps)
        raw_phys_norm = torch.sqrt(raw_phys_norm_sq + eps)
        raw_cos = dot / (data_norm * raw_phys_norm + eps)

        coefficient = torch.minimum(
            dot / (data_norm_sq + eps),
            torch.zeros((), dtype=torch.float64, device=device),
        )
        used_physics_grads = [
            gp - coefficient.to(gp.dtype) * gd
            for gd, gp in zip(data_grads, physics_grads)
        ]
        used_phys_norm = torch.sqrt(_sum_sq(used_physics_grads) + eps)

        lambda_eff = rho * data_norm / (used_phys_norm + eps)
        lambda_eff = torch.clamp(
            lambda_eff, min=lambda_min, max=lambda_max
        )
        ramp = min(1.0, float(step) / ramp_steps)
        lambda_use = lambda_eff * ramp

        optimizer.zero_grad(set_to_none=True)
        scale = lambda_use.to(params[0].dtype)
        for parameter, gd, gp in zip(params, data_grads, used_physics_grads):
            parameter.grad = (gd + scale * gp).detach()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            gd_norm = float(data_norm.detach().item())
            raw_norm = float(raw_phys_norm.detach().item())
            used_norm = float(used_phys_norm.detach().item())
            cosine = float(raw_cos.detach().item())
            lam = float(lambda_use.detach().item())
            actual_ratio = lam * used_norm / max(gd_norm, 1e-30)
            target_ratio = rho * ramp
            projected = bool((dot < 0.0).detach().item())

            final_data = float(data_loss.detach().item())
            final_physics = float(physics_loss.detach().item())
            final_lambda = lam

            history.append(
                {
                    "step": step,
                    "data_loss": final_data,
                    "physics_loss": final_physics,
                    "data_grad_norm": gd_norm,
                    "raw_physics_grad_norm": raw_norm,
                    "used_physics_grad_norm": used_norm,
                    "raw_physics_to_data_ratio": raw_norm / max(gd_norm, 1e-30),
                    "used_physics_to_data_ratio": used_norm / max(gd_norm, 1e-30),
                    "raw_grad_cosine": cosine,
                    "lambda_use": lam,
                    "target_scaled_physics_to_data_ratio": target_ratio,
                    "scaled_physics_to_data_ratio": actual_ratio,
                    "balance_error": actual_ratio - target_ratio,
                    "projection_applied": projected,
                }
            )
            logged_physics += 1
            conflicts += int(cosine < 0.0)
            lambda_values.append(lam)
            scaled_ratio_values.append(actual_ratio)
            target_ratio_values.append(target_ratio)
            balance_error_values.append(abs(actual_ratio - target_ratio))

            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    data=f"{final_data:.3e}",
                    phys=f"{final_physics:.3e}",
                    lam=f"{lam:.2e}",
                    cos=f"{cosine:.2f}",
                )

    stage3_runtime_s = time.perf_counter() - started
    background_pred = base_pinn._predict_background(
        model,
        coords,
        height,
        width,
        int(stage1_config["prediction_batch_size"]),
    )

    return {
        "background_pred": np.asarray(background_pred, dtype=np.float32),
        "history": history,
        "stage3_runtime_s": float(stage3_runtime_s),
        "peak_vram_mb": utils.peak_vram_mb(device),
        "final_data_loss": float(final_data),
        "final_physics_loss": float(final_physics if use_physics else 0.0),
        "final_lambda_use": float(final_lambda if use_physics else 0.0),
        "gradient_conflict_fraction": (
            float(conflicts / logged_physics) if logged_physics else float("nan")
        ),
        "mean_lambda_use": (
            float(np.mean(lambda_values)) if lambda_values else 0.0
        ),
        "median_lambda_use": (
            float(np.median(lambda_values)) if lambda_values else 0.0
        ),
        "mean_scaled_physics_to_data_ratio": (
            float(np.mean(scaled_ratio_values)) if scaled_ratio_values else 0.0
        ),
        "median_scaled_physics_to_data_ratio": (
            float(np.median(scaled_ratio_values)) if scaled_ratio_values else 0.0
        ),
        "mean_target_scaled_physics_to_data_ratio": (
            float(np.mean(target_ratio_values)) if target_ratio_values else 0.0
        ),
        "mean_absolute_balance_error": (
            float(np.mean(balance_error_values)) if balance_error_values else 0.0
        ),
        "projection_enabled": bool(use_physics),
        # Reconstruction backup only.  Saving the final network state does not
        # alter the optimization; it lets a disposable server be retired after
        # the run while preserving a local re-inference path.
        "model_state": _cpu_state(model.state_dict()),
    }


def _set_forcing_weights(forcing, weights):
    values = np.maximum(np.asarray(weights, dtype=np.float64), 1e-8)
    raw = np.where(values > 20.0, values, np.log(np.expm1(values)))
    with torch.no_grad():
        forcing.raw_weights.copy_(
            torch.as_tensor(
                raw,
                dtype=forcing.raw_weights.dtype,
                device=forcing.raw_weights.device,
            )
        )


def _sum_sq(grads):
    total = torch.zeros((), dtype=torch.float64, device=grads[0].device)
    for grad in grads:
        total = total + torch.sum(grad.detach().double().square())
    return total


def _dot(grads_a, grads_b):
    total = torch.zeros((), dtype=torch.float64, device=grads_a[0].device)
    for a, b in zip(grads_a, grads_b):
        total = total + torch.sum(a.detach().double() * b.detach().double())
    return total


def _cpu_state(value):
    if isinstance(value, dict):
        return {k: _cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(v) for v in value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _relative_rmse(y, p):
    denom = float(np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2)))
    return _rmse(y, p) / max(denom, 1e-12)


def _safe_corr(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if np.std(y) <= 1e-12 or np.std(p) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])
