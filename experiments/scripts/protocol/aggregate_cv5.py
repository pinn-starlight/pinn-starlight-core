"""Aggregate grouped-five-fold Stage11 results at the fold level.

The camera reference base is the independent grouping unit.  To avoid treating
multiple synthetic variants or stochastic seeds as independent camera samples,
this script first averages metrics *within each fold and method*, then reports
mean/std across the five held-out folds.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CV_ROOT = PROJECT_ROOT / "experiments" / "outputs" / "cv5"

A_METRICS = (
    "bg_mae",
    "bg_ssim",
    "residual_ssim",
    "star_f1",
    "matched_star_mean_relative_flux_error",
    "runtime_s",
)
B_METRICS = A_METRICS + (
    "faint_star_recall",
    "protected_region_mae",
    "protected_projection_abs_error",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _finite(rows, key):
    values = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _mean(rows, key):
    values = _finite(rows, key)
    return float(np.mean(values)) if values else float("nan")


def _std(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1))


def aggregate(stage: str, *, require_all_folds: bool = True) -> Path:
    stage = stage.lower()
    if stage == "a":
        relative = Path("e2_benchmark/metrics.csv")
        metrics = A_METRICS
        grouping_key = "background_type"
    elif stage == "b":
        relative = Path("stage_b/benchmark/metrics.csv")
        metrics = B_METRICS
        grouping_key = "scene_type"
    else:
        raise ValueError("stage must be 'a' or 'b'")

    fold_rows = []
    raw_by_fold = {}
    missing = []
    for fold in range(5):
        path = CV_ROOT / f"fold_{fold}" / relative
        if not path.is_file():
            missing.append(fold)
            continue
        rows = _read_csv(path)
        raw_by_fold[fold] = rows
        methods = sorted({row["method"] for row in rows})
        test_bases = sorted({row.get("base_id", "") for row in rows})
        if len([v for v in test_bases if v]) != 1:
            raise ValueError(
                f"Stage11-{stage.upper()} fold {fold} should contain one held-out base, got {test_bases}"
            )
        for method in methods:
            subset = [row for row in rows if row["method"] == method]
            record = {
                "fold": fold,
                "test_base_id": test_bases[0],
                "method": method,
                "row_count": len(subset),
            }
            for metric in metrics:
                record[metric] = _mean(subset, metric)
            fold_rows.append(record)

    if require_all_folds and missing:
        raise FileNotFoundError(
            f"Stage11-{stage.upper()} CV5 results are incomplete; missing folds: {missing}"
        )
    if not fold_rows:
        raise FileNotFoundError(
            f"No Stage11-{stage.upper()} CV5 metric files found under {CV_ROOT}"
        )

    output_root = PROJECT_ROOT / "experiments" / "outputs" / "cv5_summary" / f"stage_{stage}"
    _write_csv(output_root / "fold_method_summary.csv", fold_rows)

    methods = sorted({row["method"] for row in fold_rows})
    overall = []
    for method in methods:
        subset = [row for row in fold_rows if row["method"] == method]
        record = {"method": method, "fold_count": len(subset)}
        for metric in metrics:
            values = [float(row[metric]) for row in subset if math.isfinite(float(row[metric]))]
            record[f"{metric}_mean_across_folds"] = (
                float(np.mean(values)) if values else float("nan")
            )
            record[f"{metric}_std_across_folds"] = _std(values)
        overall.append(record)
    _write_csv(output_root / "method_summary_across_folds.csv", overall)

    grouped = []
    levels = set()
    for rows in raw_by_fold.values():
        for row in rows:
            levels.add(
                (
                    row["method"],
                    row.get(grouping_key, row.get("background_type", "")),
                    row.get("intensity_level", ""),
                )
            )
    for method, group_name, intensity in sorted(levels):
        per_fold = []
        for fold, rows in raw_by_fold.items():
            subset = [
                row
                for row in rows
                if row["method"] == method
                and row.get(grouping_key, row.get("background_type", "")) == group_name
                and row.get("intensity_level", "") == intensity
            ]
            if not subset:
                continue
            rec = {"fold": fold}
            for metric in metrics:
                rec[metric] = _mean(subset, metric)
            per_fold.append(rec)
        out = {
            "method": method,
            grouping_key: group_name,
            "intensity_level": intensity,
            "fold_count": len(per_fold),
        }
        for metric in metrics:
            vals = [float(row[metric]) for row in per_fold if math.isfinite(float(row[metric]))]
            out[f"{metric}_mean_across_folds"] = float(np.mean(vals)) if vals else float("nan")
            out[f"{metric}_std_across_folds"] = _std(vals)
        grouped.append(out)
    _write_csv(output_root / "grouped_summary_across_folds.csv", grouped)
    return output_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "b", "all"), default="all")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    stages = ("a", "b") if args.stage == "all" else (args.stage,)
    for stage in stages:
        output = aggregate(stage, require_all_folds=not args.allow_incomplete)
        print(f"Stage11-{stage.upper()} CV5 summary -> {output}")


if __name__ == "__main__":
    main()
