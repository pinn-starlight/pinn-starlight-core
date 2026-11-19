"""Anchored variant of the Stage-11 coordinated finetune (diagnostic only).

The formal ``sgc_pinn.finetune`` is frozen and must not be edited: its sha256 is
recorded in the Stage-B Global Lock.  This module therefore mirrors that loop in
a new file and adds one optional term

    L_anchor = mean((B(x) - B_init(x)) ** 2)

to the data side *before* the physics gradient is projected away.  With
``anchor_weight=0`` (or no ``anchor_background``) the call is delegated to the
formal function, so the frozen behaviour is never re-implemented.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm

import pinn_starlight_core.nn.pinn_loss as pinn_loss
from experiments.common.baselines import coordinate_pinn as base_pinn
from experiments.common.baselines import sgc_pinn
from experiments.common.utils import experiment_utils as utils
from pinn_starlight_core.data.image_loader import coordinate_grid


def finetune_anchored(
    *,
    observed,
    model_state,
    forcing_state,
    stage1_config,
    stage3_config,
    initial_centers,
    device,
    seed,
    anchor_background=None,
    anchor_weight: float = 0.0,
    show_progress: bool = False,
):
    """Run the coordinated finetune with an optional initialization anchor.

    ``anchor_background`` is the *initialization prediction* (not a parameter
    snapshot) on the same pixel grid as ``observed``.  It is indexed with the
    same random batch as the data term, so no extra forward pass is required.
    """
    weight = float(anchor_weight)
    if weight < 0.0:
        raise ValueError("anchor_weight must be non-negative")
    if anchor_background is None or weight == 0.0:
        return sgc_pinn.finetune(
            observed=observed,
            model_state=model_state,
            forcing_state=forcing_state,
            stage1_config=stage1_config,
            stage3_config=stage3_config,
            initial_centers=initial_centers,
            device=device,
            seed=seed,
            use_physics=True,
            show_progress=show_progress,
        )

    anchor = np.ascontiguousarray(np.asarray(anchor_background, dtype=np.float32))
    if anchor.shape != tuple(observed.shape):
        raise ValueError(f"anchor shape {anchor.shape} != observed {tuple(observed.shape)}")
    if not np.isfinite(anchor).all():
        raise ValueError("anchor_background contains NaN/Inf")

    utils.set_seed(int(seed))
    utils.reset_peak_vram(device)

    height, width = observed.shape
    coords = coordinate_grid(height, width, device)
    values = torch.from_numpy(np.asarray(observed, dtype=np.float32).reshape(-1)).to(device)
    anchor_values = torch.from_numpy(anchor.reshape(-1)).to(device)

    preparation = sgc_pinn._prepare_finetune(  # noqa: SLF001 - frozen loop is reused
        observed=observed,
        model_state=model_state,
        forcing_state=forcing_state,
        stage1_config=stage1_config,
        stage3_config=stage3_config,
        initial_centers=initial_centers,
        device=device,
        seed=seed,
        use_physics=True,
    )
    model = preparation.model
    forcing = preparation.forcing
    optimizer = preparation.optimizer
    generator = preparation.generator
    alpha = preparation.alpha
    params = preparation.params

    steps = int(stage3_config["steps"])
    batch_size = min(int(stage1_config["batch_size"]), int(coords.shape[0]))
    log_every = max(1, int(stage3_config["log_every"]))
    rho = float(stage3_config["rho"])
    lambda_min = float(stage3_config["lambda_min"])
    lambda_max = float(stage3_config["lambda_max"])
    ramp_steps = max(1, int(stage3_config["ramp_steps"]))

    history = []
    stats = sgc_pinn._FinetuneStats()  # noqa: SLF001 - shared diagnostic accumulator

    started = time.perf_counter()
    iterator = range(1, steps + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="balanced_anchor", leave=False)

    final_data = final_physics = final_anchor = float("nan")
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
        anchor_target = anchor_values[indices]

        background, laplacian = base_pinn._background_and_laplacian(model, batch_xy)
        data_loss = pinn_loss.mse_data(batch_target, background)
        anchor_loss = ((background - anchor_target) ** 2).mean()
        data_side_loss = data_loss + weight * anchor_loss

        forcing_pred = forcing(batch_xy, alpha)
        physics_loss = pinn_loss.mse_physics_from_laplacian(
            background, laplacian, forcing_pred, alpha
        )

        data_grads = torch.autograd.grad(
            data_side_loss, params, retain_graph=True, create_graph=False
        )
        physics_grads = torch.autograd.grad(
            physics_loss, params, retain_graph=False, create_graph=False
        )

        data_norm_sq = sgc_pinn._sum_sq(data_grads)  # noqa: SLF001
        raw_phys_norm_sq = sgc_pinn._sum_sq(physics_grads)  # noqa: SLF001
        dot = sgc_pinn._dot(data_grads, physics_grads)  # noqa: SLF001
        eps = torch.tensor(1e-24, dtype=torch.float64, device=device)

        data_norm = torch.sqrt(data_norm_sq + eps)
        raw_phys_norm = torch.sqrt(raw_phys_norm_sq + eps)
        raw_cos = dot / (data_norm * raw_phys_norm + eps)

        coefficient = torch.minimum(
            dot / (data_norm_sq + eps),
            torch.zeros((), dtype=torch.float64, device=device),
        )
        used_physics_grads = [
            gp - coefficient.to(gp.dtype) * gd for gd, gp in zip(data_grads, physics_grads)
        ]
        used_phys_norm = torch.sqrt(sgc_pinn._sum_sq(used_physics_grads) + eps)  # noqa: SLF001

        lambda_eff = rho * data_norm / (used_phys_norm + eps)
        lambda_eff = torch.clamp(lambda_eff, min=lambda_min, max=lambda_max)
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

            final_data = float(data_loss.detach().item())
            final_physics = float(physics_loss.detach().item())
            final_anchor = float(anchor_loss.detach().item())
            final_lambda = lam

            history.append(
                {
                    "step": step,
                    "data_loss": final_data,
                    "physics_loss": final_physics,
                    "anchor_loss": final_anchor,
                    "anchor_weight": weight,
                    "data_side_loss": float(data_side_loss.detach().item()),
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
                    "projection_applied": bool((dot < 0.0).detach().item()),
                }
            )
            stats.record(
                cosine=cosine,
                lambda_use=lam,
                scaled_ratio=actual_ratio,
                target_ratio=target_ratio,
            )

    stage3_runtime_s = time.perf_counter() - started
    background_pred = base_pinn._predict_background(  # noqa: SLF001 - frozen helper
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
        "final_physics_loss": float(final_physics),
        "final_anchor_loss": float(final_anchor),
        "anchor_weight": weight,
        "final_lambda_use": float(final_lambda),
        "gradient_conflict_fraction": stats.conflict_fraction(),
        "mean_lambda_use": stats.mean(stats.lambda_use),
        "median_lambda_use": stats.median(stats.lambda_use),
        "mean_scaled_physics_to_data_ratio": stats.mean(stats.scaled_ratio),
        "median_scaled_physics_to_data_ratio": stats.median(stats.scaled_ratio),
        "mean_target_scaled_physics_to_data_ratio": stats.mean(stats.target_ratio),
        "mean_absolute_balance_error": stats.mean(stats.balance_error),
        "projection_enabled": True,
        "model_state": sgc_pinn._cpu_state(model.state_dict()),  # noqa: SLF001
    }
