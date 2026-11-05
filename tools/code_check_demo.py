"""Live, CPU-only demonstration for the competition code inspection.

What this shows
---------------
The formal paper pipeline needs a GPU and ~11 GB of data.  This script instead
runs the *same scientific code* on one small validation sample and produces a
real figure plus a real metrics table in well under a minute on a laptop CPU:

    Observed -> FFT-Gaussian -> Cosine-LS (DCT-II) -> Original PINN -> SGC-PINN

Everything is computed live; nothing is loaded from a cached result.  The
figure is deliberately laid out like the paper's qualitative figures
(backgrounds on top, background-subtracted residuals below) so the audience can
connect the demo to Fig. 3 / Fig. 8.

The demo runs the same four methods as the paper.  Show
``artifacts/review/e4_review_for_paper.zip`` for the archived real-image
results (see docs/CODE_CHECK_PLAYBOOK.md).

Usage
-----
    python -m tools.code_check_demo
    python -m tools.code_check_demo --downsample 6 --steps 20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.baselines import sgc_pinn as final_pinn
from experiments.common.baselines import fft_gaussian as fft
from experiments.common.baselines import cosine_ls as fourier_ls
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE = "clean_single_002_multi_source_high"
DEFAULT_OUTPUT = utils.BASE_OUTPUT_ROOT / "code_check_demo"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_sample(sample_id: str, downsample: int) -> dict:
    manifest = utils.SYNTHETIC_DATA_ROOT / "manifest.csv"
    if not manifest.is_file():
        raise SystemExit(
            "Stage-A dataset not found.\n"
            "Run:  bash tools/fetch_smoke_data.sh\n"
            f"(expected manifest at {manifest})"
        )
    rows = utils.load_manifest(manifest, split="validation")
    row = next((r for r in rows if r["sample_id"] == sample_id), None)
    if row is None:
        raise SystemExit(
            f"sample {sample_id!r} is not in the validation manifest. "
            f"Run: bash tools/fetch_smoke_data.sh"
        )

    def load(key):
        array = utils.load_float_tiff(row[key])
        if downsample > 1:
            array = array[::downsample, ::downsample]
        return np.ascontiguousarray(array, dtype=np.float32)

    return {
        "sample_id": sample_id,
        "observed": load("observed"),
        "background_true": load("background_true"),
        "clean_true": load("clean_true"),
        "metadata": utils.read_json(utils.to_absolute_path(row["metadata"])),
        "downsample": downsample,
    }


# --------------------------------------------------------------------------
# methods (same code the formal runners use, only tiny budgets)
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


def run_methods(sample: dict, steps: int, device) -> dict:
    """Run the four paper methods with tiny CPU budgets.

    Faithfulness notes:
    * ``Original PINN`` uses the paper's joint data+physics objective
      (physics_weight 0.4, sparse_weight 1e-3).
    * ``SGC-PINN`` follows the paper's three stages: a **data-only** stage 1,
      a fixed-geometry forcing fit, then a coordinated finetune with conflict
      projection.  It must not inherit the joint objective from above.
    """
    observed = sample["observed"]
    predictions, timings = {}, {}

    started = time.perf_counter()
    predictions["fft_gaussian"] = fft.estimate_background(observed, 0.02)
    timings["fft_gaussian"] = time.perf_counter() - started

    started = time.perf_counter()
    predictions["fourier_ls"] = fourier_ls.estimate_background(
        observed, 1e-3, 1e-4, backend="cpu", show_progress=False
    )
    timings["fourier_ls"] = time.perf_counter() - started

    # --- Original PINN: joint data + physics -------------------------------
    joint_cfg = stage1_config(steps)
    joint_cfg["physics_weight"] = 0.4
    joint_cfg["sparse_weight"] = 1e-3
    centers = pinn.initial_centers_from_observed(observed, joint_cfg)
    started = time.perf_counter()
    joint = pinn.train_background(
        observed,
        config=joint_cfg,
        device=device,
        seed=20260728,
        show_progress=False,
        return_state=True,
        laplacian_backend="analytic",
        initial_centers=centers,
    )
    predictions["pinn"] = joint["background_pred"]
    timings["pinn"] = time.perf_counter() - started

    # --- SGC-PINN: data-only stage 1 -> forcing fit -> coordinated stage 3 --
    sgc_cfg = stage1_config(steps)
    sgc_cfg["physics_weight"] = 0.0
    sgc_cfg["sparse_weight"] = 0.0
    sgc_centers = pinn.initial_centers_from_observed(observed, sgc_cfg)
    started = time.perf_counter()
    stage1 = pinn.train_background(
        observed,
        config=sgc_cfg,
        device=device,
        seed=20260728,
        show_progress=False,
        return_state=True,
        laplacian_backend="analytic",
        initial_centers=sgc_centers,
    )
    forcing_state, _ = final_pinn.fit_fixed_forcing(
        observed=observed,
        model_state=stage1["state"]["model_state"],
        stage1_config=sgc_cfg,
        stage2_config={
            "mode": "fixed_geometry_nonnegative_least_squares",
            "prefit_points": 32768,
            "fit_fraction": 0.8,
            "geometry_refinement": False,
            "uses_true_background": False,
        },
        initial_centers=sgc_centers,
        device=device,
        seed=20260728 + 1009,
    )
    refined = final_pinn.finetune(
        observed=observed,
        model_state=stage1["state"]["model_state"],
        forcing_state=forcing_state,
        stage1_config=sgc_cfg,
        stage3_config=stage3_config(steps),
        initial_centers=sgc_centers,
        device=device,
        seed=20260728 + 2003,
        use_physics=True,
        show_progress=False,
    )
    predictions["final_pinn"] = refined["background_pred"]
    timings["final_pinn"] = time.perf_counter() - started

    return {
        "backgrounds": predictions,
        "timings": timings,
        "raw": {"pinn": joint, "final_pinn": refined},
    }


def evaluate(sample: dict, backgrounds: dict) -> dict:
    observed = sample["observed"]
    table = {}
    full_sample = {
        "observed": observed,
        "background_true": sample["background_true"],
        "clean_true": sample["clean_true"],
        "metadata": sample["metadata"],
    }
    for name, bg in backgrounds.items():
        prediction = {
            "background_pred": bg,
            "residual_pred": utils.reconstruct_residual(observed, bg),
        }
        result = metrics.evaluate_synthetic(full_sample, prediction)
        table[name] = {
            k: result.get(k)
            for k in (
                "bg_mae", "bg_rmse", "bg_psnr", "bg_ssim",
                "residual_ssim", "star_f1",
                "matched_star_mean_relative_flux_error",
                "forced_reference_mean_relative_flux_error",
            )
        }
    return table


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def draw_figure(path: Path, sample: dict, backgrounds: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed = sample["observed"]
    truth_bg = sample["background_true"]
    truth_clean = sample["clean_true"]

    columns = [
        ("Observed", observed, observed),
        ("Truth", truth_bg, truth_clean),
        ("FFT-Gaussian", backgrounds["fft_gaussian"], None),
        ("Cosine-LS", backgrounds["fourier_ls"], None),
        ("Original PINN", backgrounds["pinn"], None),
        ("SGC-PINN", backgrounds["final_pinn"], None),
    ]

    fig, axes = plt.subplots(2, len(columns), figsize=(3.0 * len(columns), 6.4))
    bg_vmin, bg_vmax = float(np.percentile(truth_bg, 1)), float(np.percentile(truth_bg, 99))
    hi_vmin, hi_vmax = float(np.percentile(observed, 1)), float(np.percentile(observed, 99))

    for col, (title, bg, given) in enumerate(columns):
        top = axes[0, col]
        top.imshow(np.asarray(bg), cmap="gray", vmin=bg_vmin, vmax=bg_vmax)
        top.set_title(f"{title}\nbackground", fontsize=10)

        bottom = axes[1, col]
        residual = given if given is not None else utils.reconstruct_residual(observed, bg)
        bottom.imshow(np.asarray(residual), cmap="gray", vmin=hi_vmin, vmax=hi_vmax)
        bottom.set_title("observed" if col == 0 else "residual (BG subtracted)", fontsize=10)

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"CPU code-inspection demo - {sample['sample_id']} "
        f"(downsample x{sample['downsample']}) - computed live, not a paper result",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default=DEFAULT_SAMPLE)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    import torch

    device = torch.device("cpu")
    print("=" * 78)
    print("PINN-Starlight - live CPU code-inspection demo")
    print("=" * 78)
    print(f"torch          : {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
    print(f"device         : {device}")
    print(f"sample         : {args.sample}")
    print(f"downsample     : x{args.downsample}")
    print(f"optimiser steps: {args.steps} per PINN stage")
    print("=" * 78)

    started = time.perf_counter()
    sample = load_sample(args.sample, args.downsample)
    print(f"loaded observed {sample['observed'].shape} in {time.perf_counter()-started:.1f}s")

    started = time.perf_counter()
    run = run_methods(sample, args.steps, device)
    print(f"ran 4 methods in {time.perf_counter()-started:.1f}s\n")

    print(f"{'method':<16}{'bg MAE':>12}{'bg PSNR':>10}{'bg SSIM':>10}"
          f"{'res SSIM':>10}{'star F1':>9}   {'time':>7}")
    print("-" * 78)
    table = evaluate(sample, run["backgrounds"])
    labels = {
        "fft_gaussian": "FFT-Gaussian",
        "fourier_ls": "Cosine-LS",
        "pinn": "Original PINN",
        "final_pinn": "SGC-PINN",
    }

    def fmt(value, spec):
        if value is None or not np.isfinite(value):
            return "n/a".rjust(9)
        return format(value, spec)

    for key, label in labels.items():
        row = table[key]
        print(
            f"{label:<16}{fmt(row['bg_mae'], '.6f'):>12}{fmt(row['bg_psnr'], '.2f'):>10}"
            f"{fmt(row['bg_ssim'], '.4f'):>10}{fmt(row['residual_ssim'], '.4f'):>10}"
            f"{fmt(row['star_f1'], '.3f'):>9}{run['timings'][key]:>8.2f}s"
        )
    print("-" * 78)
    print("CPU demo budgets are far below the paper protocol; values are NOT paper numbers.")

    args.out.mkdir(parents=True, exist_ok=True)
    figure_path = args.out / "demo_comparison.png"
    draw_figure(figure_path, sample, run["backgrounds"])

    summary = {
        "sample_id": sample["sample_id"],
        "downsample": args.downsample,
        "steps": args.steps,
        "device": "cpu",
        "note": "Live CPU demo with reduced budgets; not a paper result.",
        "timings_s": run["timings"],
        "metrics": table,
    }
    (args.out / "demo_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nfigure  : {figure_path}")
    print(f"metrics : {args.out / 'demo_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
