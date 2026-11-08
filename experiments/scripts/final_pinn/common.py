"""Shared Stage11 final-PINN evaluation helpers."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

from experiments.common.utils import metrics


PRIMARY_METRICS = (
    "bg_mae",
    "bg_rmse",
    "bg_psnr",
    "bg_ssim",
    "residual_ssim",
    "star_f1",
    "matched_star_mean_relative_flux_error",
    "forced_reference_mean_relative_flux_error",
    "forced_reference_p95_relative_flux_error",
)


def forced_reference_metrics(sample, residual):
    reference_info = sample.get("metadata", {}).get("star_reference", {})
    stars = list(reference_info.get("stars") or [])
    phot = dict(reference_info.get("photometry") or metrics.default_photometry_config())

    values = []
    for star in stars:
        true_flux = float(star.get("raw_aperture_flux", float("nan")))
        if not np.isfinite(true_flux) or abs(true_flux) <= 1e-8:
            continue
        pred_flux = metrics.measure_raw_aperture_flux(
            residual,
            star["x"],
            star["y"],
            aperture_radius=float(phot["aperture_radius"]),
            annulus_inner_radius=float(phot["annulus_inner_radius"]),
            annulus_outer_radius=float(phot["annulus_outer_radius"]),
        )
        if not np.isfinite(pred_flux):
            continue
        values.append(abs(pred_flux - true_flux) / abs(true_flux))

    if not values:
        return {
            "forced_reference_mean_relative_flux_error": float("nan"),
            "forced_reference_median_relative_flux_error": float("nan"),
            "forced_reference_p95_relative_flux_error": float("nan"),
            "forced_reference_flux_count": 0,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "forced_reference_mean_relative_flux_error": float(np.mean(arr)),
        "forced_reference_median_relative_flux_error": float(np.median(arr)),
        "forced_reference_p95_relative_flux_error": float(np.percentile(arr, 95.0)),
        "forced_reference_flux_count": int(arr.size),
    }


def paired_comparisons(rows, candidate="pinn_decoupled_gradient", baseline="data_finetune_control"):
    by_key = {}
    for row in rows:
        by_key[(row["method"], str(row["seed"]), row["sample_id"])] = row
    output = []
    seeds = sorted({str(r["seed"]) for r in rows})
    sample_ids = sorted({r["sample_id"] for r in rows})
    lower_better = {
        "bg_mae": True,
        "bg_rmse": True,
        "bg_psnr": False,
        "bg_ssim": False,
        "residual_ssim": False,
        "star_f1": False,
        "matched_star_mean_relative_flux_error": True,
        "forced_reference_mean_relative_flux_error": True,
        "forced_reference_p95_relative_flux_error": True,
    }

    for metric in PRIMARY_METRICS:
        deltas = []
        wins = losses = ties = 0
        for seed in seeds:
            for sample_id in sample_ids:
                a = by_key.get((candidate, seed, sample_id))
                b = by_key.get((baseline, seed, sample_id))
                if not a or not b:
                    continue
                try:
                    av = float(a[metric])
                    bv = float(b[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if not np.isfinite(av) or not np.isfinite(bv):
                    continue
                delta = av - bv
                deltas.append(delta)
                tol = max(1e-12, 1e-8 * max(abs(av), abs(bv), 1.0))
                if abs(delta) <= tol:
                    ties += 1
                elif (delta < 0 if lower_better[metric] else delta > 0):
                    wins += 1
                else:
                    losses += 1
        output.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "metric": metric,
                "better_direction": "lower" if lower_better[metric] else "higher",
                "valid_pairs": len(deltas),
                "candidate_wins": wins,
                "candidate_losses": losses,
                "ties": ties,
                "mean_delta_candidate_minus_baseline": (
                    float(np.mean(deltas)) if deltas else float("nan")
                ),
                "median_delta_candidate_minus_baseline": (
                    float(np.median(deltas)) if deltas else float("nan")
                ),
            }
        )
    return output


def write_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
