"""Shared single-image method chain for the CLI and the CPU demo.

The formal experiments keep their own frozen runners.  This module is the one
place the *demo / release* command line uses to call the four paper methods on
one small image, so ``pinn-starlight estimate`` and ``tools/code_check_demo``
cannot drift apart.

Budgets here are deliberately tiny (CPU demo), not paper budgets.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.baselines import cosine_ls as fourier_ls
from experiments.common.baselines import fft_gaussian as fft
from experiments.common.baselines import mode_switch
from experiments.common.baselines import sgc_pinn as final_pinn
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.configs.demo_methods import METHOD_NAMES, PAPER_METHODS
from experiments.configs.formal_experiment import FORMAL_SEEDS

DEFAULT_SEED = FORMAL_SEEDS[0]


# --------------------------------------------------------------------------
# demo budgets
# --------------------------------------------------------------------------
def stage1_config(steps: int) -> dict:
    return pinn.normalized_config(
        {
            "hidden_dims": [128, 64],
            "steps": int(steps),
            "batch_size": 8192,
            "model_lr": 1e-2,
            "alpha": 0.5,
            "physics_weight": 0.4,
            "sparse_weight": 0.0,
            "center_warmup_steps": min(200, max(1, steps // 3)),
            "center_refinement_enabled": False,
            "forcing_max_components": 3,
            "forcing_rms_grid_size": 48,
            "prediction_batch_size": 65536,
            "log_every": max(1, steps // 5),
        }
    )


def stage3_config(steps: int) -> dict:
    return {
        "steps": int(steps),
        "model_lr": 2e-4,
        "rho": 0.1,
        "lambda_min": 0.0,
        "lambda_max": 0.02,
        "ramp_steps": max(1, steps // 2),
        "conflict_projection": True,
        "sparse_weight": 0.0,
        "neumann_boundary_loss": False,
        "gradient_balance_reference": "projected_regularizer_gradient",
        "laplacian_backend": "analytic",
        "log_every": max(1, steps // 5),
    }


def _stage2_config(steps: int) -> dict:
    return {
        "mode": "fixed_geometry_nonnegative_least_squares",
        "prefit_points": max(4096, min(32768, 8192)),
        "fit_fraction": 0.8,
        "geometry_refinement": False,
        "uses_true_background": False,
    }


# --------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------
def _run_joint_pinn(observed: np.ndarray, *, steps: int, device, seed: int, show_progress: bool):
    config = stage1_config(steps)
    config["physics_weight"] = 0.4
    config["sparse_weight"] = 1e-3
    centers = pinn.initial_centers_from_observed(observed, config)
    started = time.perf_counter()
    result = pinn.train_background(
        observed,
        config=config,
        device=device,
        seed=seed,
        show_progress=show_progress,
        return_state=True,
        laplacian_backend="analytic",
        initial_centers=centers,
    )
    return result, time.perf_counter() - started


def _run_sgc(observed: np.ndarray, *, steps: int, device, seed: int, show_progress: bool):
    config = stage1_config(steps)
    config["physics_weight"] = 0.0
    config["sparse_weight"] = 0.0
    centers = pinn.initial_centers_from_observed(observed, config)
    started = time.perf_counter()
    stage1 = pinn.train_background(
        observed,
        config=config,
        device=device,
        seed=seed,
        show_progress=show_progress,
        return_state=True,
        laplacian_backend="analytic",
        initial_centers=centers,
    )
    forcing_state, forcing_diag = final_pinn.fit_fixed_forcing(
        observed=observed,
        model_state=stage1["state"]["model_state"],
        stage1_config=config,
        stage2_config=_stage2_config(steps),
        initial_centers=centers,
        device=device,
        seed=seed + 1009,
    )
    refined = final_pinn.finetune(
        observed=observed,
        model_state=stage1["state"]["model_state"],
        forcing_state=forcing_state,
        stage1_config=config,
        stage3_config=stage3_config(steps),
        initial_centers=centers,
        device=device,
        seed=seed + 2003,
        use_physics=True,
        show_progress=show_progress,
    )
    elapsed = time.perf_counter() - started
    refined["forcing_diagnostics"] = forcing_diag
    return refined, elapsed


def _run_balanced(observed: np.ndarray, *, steps: int, device, seed: int, show_progress: bool):
    started = time.perf_counter()
    run = mode_switch.run_mode(
        observed,
        mode="balanced",
        config=mode_switch.mode_config_for_steps(steps=steps),
        device=device,
        seed=seed,
        show_progress=show_progress,
    )
    return run, time.perf_counter() - started


def run_methods(
    observed,
    *,
    steps: int = 500,
    device,
    methods=None,
    seed: int = DEFAULT_SEED,
    show_progress: bool = False,
) -> dict:
    """Run the requested demo methods on one already-loaded image."""
    observed = utils.load_observed(observed)
    selected = tuple(PAPER_METHODS if methods is None else methods)
    unknown = sorted(set(selected) - set(METHOD_NAMES))
    if unknown:
        raise ValueError(f"unknown method(s): {unknown}; expected a subset of {METHOD_NAMES}")

    backgrounds: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    raw: dict[str, dict] = {}

    if "fft_gaussian" in selected:
        started = time.perf_counter()
        backgrounds["fft_gaussian"] = fft.estimate_background(observed, 0.02)
        timings["fft_gaussian"] = time.perf_counter() - started

    if "fourier_ls" in selected:
        started = time.perf_counter()
        backgrounds["fourier_ls"] = fourier_ls.estimate_background(
            observed, 1e-3, 1e-4, backend="cpu", show_progress=False
        )
        timings["fourier_ls"] = time.perf_counter() - started

    if "pinn" in selected:
        result, elapsed = _run_joint_pinn(
            observed, steps=steps, device=device, seed=seed, show_progress=show_progress
        )
        backgrounds["pinn"] = result["background_pred"]
        timings["pinn"] = elapsed
        raw["pinn"] = result

    if "final_pinn" in selected:
        result, elapsed = _run_sgc(
            observed, steps=steps, device=device, seed=seed, show_progress=show_progress
        )
        backgrounds["final_pinn"] = result["background_pred"]
        timings["final_pinn"] = elapsed
        raw["final_pinn"] = result

    if "balanced" in selected:
        result, elapsed = _run_balanced(
            observed, steps=steps, device=device, seed=seed, show_progress=show_progress
        )
        backgrounds["balanced"] = result["background_pred"]
        timings["balanced"] = elapsed
        raw["balanced"] = result

    return {"backgrounds": backgrounds, "timings": timings, "raw": raw}


# --------------------------------------------------------------------------
# input / descriptive metrics
# --------------------------------------------------------------------------
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nef", ".cr2", ".dng", ".arw"}


def load_image_array(path: Path | str, *, max_side: int = 384) -> dict:
    """Load any supported image and cap its long side (demo only)."""
    from pinn_starlight_core.data.image_loader import ImageLoader

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(
            f"unsupported extension {path.suffix!r}; expected one of {sorted(IMAGE_SUFFIXES)}"
        )
    probe = ImageLoader(path, device="cpu", downsample=1)
    stride = max(1, int(np.ceil(max(probe.H, probe.W) / float(max_side))))
    loader = ImageLoader(path, device="cpu", downsample=stride) if stride > 1 else probe
    _, gray, width, height = loader.get_gray_data()
    observed = np.asarray(gray, dtype=np.float32).reshape(height, width)
    return {
        "sample_id": path.stem,
        "source": str(path),
        "observed": np.ascontiguousarray(observed, dtype=np.float32),
        "downsample": stride,
        "has_truth": False,
    }


def load_dataset_sample(sample_id: str, *, max_side: int = 384, dataset: str = "stage_a") -> dict:
    """Load one validation sample from an extracted dataset (no archive access)."""
    if dataset == "stage_a":
        manifest = Path(utils.SYNTHETIC_MANIFEST)
    elif dataset == "stage_b":
        from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST

        manifest = Path(STAGE_B_MANIFEST)
    else:
        raise ValueError(f"unknown dataset: {dataset!r}")
    rows = utils.load_manifest(manifest, split="validation")
    row = next((item for item in rows if item["sample_id"] == sample_id), None)
    if row is None:
        available = ", ".join(sorted(item["sample_id"] for item in rows))
        raise KeyError(
            f"{sample_id!r} is not in the {dataset} validation split; available: {available}"
        )

    observed_full = utils.load_float_tiff(row["observed"])
    stride = max(1, int(np.ceil(max(observed_full.shape) / float(max_side))))
    observed = np.ascontiguousarray(observed_full[::stride, ::stride], dtype=np.float32)
    sample = {
        "sample_id": sample_id,
        "source": utils.project_relative(utils.to_absolute_path(row["observed"])),
        "observed": observed,
        "downsample": stride,
        "has_truth": True,
        "background_true": np.ascontiguousarray(
            utils.load_float_tiff(row["background_true"])[::stride, ::stride], dtype=np.float32
        ),
        "clean_true": np.ascontiguousarray(
            utils.load_float_tiff(row["clean_true"])[::stride, ::stride], dtype=np.float32
        ),
        "metadata": utils.read_json(utils.to_absolute_path(row["metadata"])),
        "dataset": dataset,
        "scene_type": row.get("scene_type", row.get("background_type")),
    }
    if dataset == "stage_b":
        sample["base_clean"] = np.ascontiguousarray(
            utils.load_float_tiff(row["base_clean"])[::stride, ::stride], dtype=np.float32
        )
        sample["protected_true"] = np.ascontiguousarray(
            utils.load_float_tiff(row["protected_true"])[::stride, ::stride], dtype=np.float32
        )
    return sample


def describe_real(observed, background_pred) -> dict:
    """Descriptive residual metrics for a photograph without ground truth."""
    observed = utils.load_observed(observed)
    residual = utils.reconstruct_residual(observed, background_pred)
    return metrics.evaluate_real(observed, background_pred, residual)


def evaluate_synthetic(sample: dict, backgrounds: dict) -> dict:
    """Truth-based metrics per method, with per-method error isolation."""
    full = {
        "observed": sample["observed"],
        "background_true": sample["background_true"],
        "clean_true": sample["clean_true"],
        "metadata": sample["metadata"],
    }
    if sample.get("dataset") == "stage_b":
        from experiments.common.utils import stage_b_metrics

        full = {**full, **{key: sample[key] for key in ("base_clean", "protected_true")}}

    table = {}
    for name, background in backgrounds.items():
        prediction = {
            "background_pred": background,
            "residual_pred": utils.reconstruct_residual(sample["observed"], background),
        }
        try:
            if sample.get("dataset") == "stage_b":
                result = stage_b_metrics.evaluate_stage_b(full, prediction)
            else:
                result = metrics.evaluate_synthetic(full, prediction)
        except Exception as exc:  # noqa: BLE001 - one method must not kill the run
            table[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        table[name] = {
            key: result.get(key)
            for key in (
                "bg_mae",
                "bg_rmse",
                "bg_psnr",
                "bg_ssim",
                "residual_ssim",
                "star_f1",
                "matched_star_mean_relative_flux_error",
                "faint_star_recall",
                "protected_projection_ratio",
                "protected_region_mae",
            )
        }
    return table
