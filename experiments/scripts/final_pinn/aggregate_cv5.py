"""Aggregate Stage11-A final-PINN five-fold confirmation by camera-reference fold."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from experiments.common.utils import experiment_utils as utils
from experiments.scripts.final_pinn.common import write_csv

METRICS = (
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


def _read(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _finite(rows, key):
    out = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            out.append(value)
    return out


def main(*, allow_incomplete=False):
    fold_rows = {}
    for fold in range(5):
        path = (
            utils.BASE_OUTPUT_ROOT
            / "cv5"
            / f"fold_{fold}"
            / "final_pinn_validation"
            / "metrics.csv"
        )
        if path.is_file():
            fold_rows[fold] = _read(path)
        elif not allow_incomplete:
            raise FileNotFoundError(f"missing Stage11 final-PINN fold output: {path}")

    output_root = utils.BASE_OUTPUT_ROOT / "cv5_summary" / "stage_a_final_pinn"
    output_root.mkdir(parents=True, exist_ok=True)

    fold_means = []
    methods = sorted({row["method"] for rows in fold_rows.values() for row in rows})
    for fold, rows in sorted(fold_rows.items()):
        for method in methods:
            subset = [r for r in rows if r["method"] == method]
            if not subset:
                continue
            rec = {"fold": fold, "method": method, "row_count": len(subset)}
            for metric in METRICS:
                vals = _finite(subset, metric)
                rec[metric] = float(np.mean(vals)) if vals else float("nan")
            fold_means.append(rec)
    write_csv(output_root / "fold_means.csv", fold_means)

    summary = []
    for method in methods:
        subset = [r for r in fold_means if r["method"] == method]
        rec = {"method": method, "fold_count": len(subset)}
        for metric in METRICS:
            vals = _finite(subset, metric)
            rec[f"{metric}_mean_across_folds"] = float(np.mean(vals)) if vals else float("nan")
            rec[f"{metric}_std_across_folds"] = float(np.std(vals)) if vals else float("nan")
        summary.append(rec)
    write_csv(output_root / "method_summary_across_folds.csv", summary)

    # Paired fold-level delta: improved - control.
    by = {(r["fold"], r["method"]): r for r in fold_means}
    deltas = []
    for fold in sorted(fold_rows):
        a = by.get((fold, "pinn_decoupled_gradient"))
        b = by.get((fold, "data_finetune_control"))
        if not a or not b:
            continue
        rec = {"fold": fold}
        for metric in METRICS:
            try:
                rec[f"{metric}_delta_improved_minus_control"] = float(a[metric]) - float(b[metric])
            except (TypeError, ValueError):
                rec[f"{metric}_delta_improved_minus_control"] = float("nan")
        deltas.append(rec)
    write_csv(output_root / "paired_fold_deltas.csv", deltas)

    print("Stage11-A final-PINN CV5 summary ->", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    main(allow_incomplete=args.allow_incomplete)
