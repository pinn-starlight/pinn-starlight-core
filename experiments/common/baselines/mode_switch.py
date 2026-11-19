"""Star / nebula / balanced operating points (experimental).

The three modes bundle already frozen building blocks instead of inventing a new
formal method:

``star``
    data-only stage 1 -> fixed forcing fit -> coordinated finetune.  This is the
    frozen SGC-PINN pipeline and its best-background / point-source operating
    point.
``nebula``
    joint data + physics coordinate PINN.  This is the frozen original PINN and
    its best diffuse-structure operating point.
``balanced``
    physics-first stage 1 -> fixed forcing fit -> anchored coordinated finetune.
    The anchor ``weight * mean((B - B_init)^2)`` is added to the data side before
    the physics gradient is projected away; the strength is a diagnostic knob.

Nothing here changes a formal configuration.  Callers pass an explicit config
(see :func:`build_config`) and get a background prediction plus a small
diagnostic record.
"""

from __future__ import annotations

import time

import numpy as np

from experiments.common.baselines import coordinate_pinn as base_pinn
from experiments.common.baselines import sgc_pinn
from experiments.common.baselines.anchored_finetune import finetune_anchored
from experiments.common.utils import experiment_utils as utils
from experiments.configs import diagnostics as diag

MODES = diag.MODE_NAMES


def build_config(
    *,
    suite: str = "full",
    init_steps: int | None = None,
    finetune_steps: int | None = None,
    pinn_steps: int | None = None,
    anchor_weight: float | None = None,
) -> dict:
    """Assemble the mode config for one suite; no formal config is touched."""
    weight = (
        float(diag.MODE_DEFAULTS["balanced"]["anchor_weight"])
        if anchor_weight is None
        else float(anchor_weight)
    )
    return {
        "suite": suite,
        "stage1_data": diag.stage1_config(suite=suite, init="data_first", steps=init_steps),
        "stage1_physics": diag.stage1_config(suite=suite, init="physics_first", steps=init_steps),
        "stage2": diag.stage2_config(suite=suite),
        "stage3": diag.stage3_config(suite=suite, steps=finetune_steps),
        "pinn": diag.pinn_config(suite=suite, steps=pinn_steps),
        "anchor_weight": weight,
    }


def _validate_config(config: dict) -> dict:
    required = {"stage1_data", "stage1_physics", "stage2", "stage3", "pinn"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"mode config is missing {missing}; use build_config()")
    weight = float(config.get("anchor_weight", 0.0))
    if not 0.0 <= weight <= 1.0:
        raise ValueError("anchor_weight must lie in [0, 1]")
    return {
        **config,
        "stage1_data": base_pinn.normalized_config(config["stage1_data"]),
        "stage1_physics": base_pinn.normalized_config(config["stage1_physics"]),
        "stage2": dict(config["stage2"]),
        "stage3": dict(config["stage3"]),
        "pinn": base_pinn.normalized_config(config["pinn"]),
        "anchor_weight": weight,
    }


def _decoupled(
    observed,
    *,
    stage1_config: dict,
    stage2_config: dict,
    stage3_config: dict,
    anchor_weight: float,
    device,
    seed: int,
    show_progress: bool,
) -> dict:
    """Stage 1 -> fixed forcing -> coordinated (optionally anchored) finetune."""
    centers = base_pinn.initial_centers_from_observed(observed, stage1_config)
    stage1 = base_pinn.train_background(
        observed,
        config=stage1_config,
        device=device,
        seed=seed,
        show_progress=show_progress,
        return_state=True,
        laplacian_backend="analytic",
        initial_centers=centers,
    )

    forcing_started = time.perf_counter()
    forcing_state, forcing_diag = sgc_pinn.fit_fixed_forcing(
        observed=observed,
        model_state=stage1["state"]["model_state"],
        stage1_config=stage1_config,
        stage2_config=stage2_config,
        initial_centers=centers,
        device=device,
        seed=int(seed) + 1009,
    )
    forcing_runtime_s = time.perf_counter() - forcing_started

    started = time.perf_counter()
    refined = finetune_anchored(
        observed=observed,
        model_state=stage1["state"]["model_state"],
        forcing_state=forcing_state,
        stage1_config=stage1_config,
        stage3_config=stage3_config,
        initial_centers=centers,
        device=device,
        seed=int(seed) + 2003,
        anchor_background=stage1["background_pred"],
        anchor_weight=float(anchor_weight),
        show_progress=show_progress,
    )
    stage3_runtime_s = time.perf_counter() - started

    return {
        "background_pred": np.asarray(refined["background_pred"], dtype=np.float32),
        "runtime_s": float(stage1["runtime_s"] + forcing_runtime_s + stage3_runtime_s),
        "details": {
            "stage1_runtime_s": float(stage1["runtime_s"]),
            "forcing_runtime_s": float(forcing_runtime_s),
            "stage3_runtime_s": float(stage3_runtime_s),
            "stage1_background_mean": float(np.mean(stage1["background_pred"])),
            "anchor_weight": float(anchor_weight),
            "final_data_loss": refined.get("final_data_loss"),
            "final_physics_loss": refined.get("final_physics_loss"),
            "final_anchor_loss": refined.get("final_anchor_loss"),
            "gradient_conflict_fraction": refined.get("gradient_conflict_fraction"),
            "forcing": {
                key: forcing_diag.get(key)
                for key in (
                    "fit_relative_rmse",
                    "diagnostic_check_relative_rmse",
                    "diagnostic_check_correlation",
                )
            },
        },
        "history": refined.get("history", []),
    }


def run_mode(
    observed,
    *,
    mode: str,
    config: dict,
    device,
    seed: int = diag.DIAGNOSTIC_SEED,
    show_progress: bool = False,
) -> dict:
    """Run one operating point and return its background prediction."""
    mode = str(mode).strip().lower()
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
    cfg = _validate_config(config)
    observed = utils.load_observed(observed)

    if mode == "nebula":
        result = base_pinn.train_background(
            observed,
            config=cfg["pinn"],
            device=device,
            seed=int(seed),
            show_progress=show_progress,
            laplacian_backend="analytic",
        )
        return {
            "mode": mode,
            "background_pred": np.asarray(result["background_pred"], dtype=np.float32),
            "runtime_s": float(result["runtime_s"]),
            "details": {
                "physics_weight": float(cfg["pinn"]["physics_weight"]),
                "sparse_weight": float(cfg["pinn"]["sparse_weight"]),
                "final_data_loss": result.get("data_loss"),
                "final_physics_loss": result.get("physics_loss"),
                "effective_component_count": result.get("effective_component_count"),
            },
            "history": result.get("history", []),
        }

    if mode == "star":
        run = _decoupled(
            observed,
            stage1_config=cfg["stage1_data"],
            stage2_config=cfg["stage2"],
            stage3_config=cfg["stage3"],
            anchor_weight=0.0,
            device=device,
            seed=int(seed),
            show_progress=show_progress,
        )
        return {"mode": mode, **run}

    run = _decoupled(
        observed,
        stage1_config=cfg["stage1_physics"],
        stage2_config=cfg["stage2"],
        stage3_config=cfg["stage3"],
        anchor_weight=cfg["anchor_weight"],
        device=device,
        seed=int(seed),
        show_progress=show_progress,
    )
    return {"mode": mode, **run}


def mode_config_for_steps(*, steps: int, suite: str = "smoke", anchor_weight: float | None = None):
    """Convenience for the CLI: all stages get the same tiny/demo budget."""
    return build_config(
        suite=suite,
        init_steps=steps,
        finetune_steps=steps,
        pinn_steps=steps,
        anchor_weight=anchor_weight,
    )
