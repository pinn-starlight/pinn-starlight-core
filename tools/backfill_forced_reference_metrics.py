"""Backfill forced-reference photometry on frozen Stage11 predictions.

This tool is evaluation-only. It never retrains a model, changes a selection
artifact, modifies a Global Lock, or overwrites formal benchmark outputs.
It reads the already-frozen ``background_pred.tif`` files, reconstructs the
scientific residual from the original observed TIFF, applies the existing
frozen forced-reference aperture-photometry rule, and writes derived CSV/JSON
under ``experiments/outputs/forced_reference_backfill``.

Default coverage:
- Stage-A E2 four-method benchmark;
- Stage-A E3 Original-PINN ablation;
- Stage-B four-method benchmark;
- Stage-A CV5 four-method benchmark, fold 0..4.

Final-PINN outputs are not recomputed because their formal evaluator already
contains forced-reference mean/P95 metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.common.utils import experiment_utils as utils
from experiments.scripts.final_pinn.common import forced_reference_metrics

OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "outputs" / "forced_reference_backfill"

FORCED_KEYS = (
    "forced_reference_mean_relative_flux_error",
    "forced_reference_median_relative_flux_error",
    "forced_reference_p95_relative_flux_error",
    "forced_reference_flux_count",
)

COMMON_PAPER_METRICS = (
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

STAGE_B_EXTRA_METRICS = (
    "faint_star_recall",
    "protected_region_mae",
    "protected_projection_ratio",
    "protected_projection_abs_error",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _finite(rows: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def _mean(rows: list[dict], key: str) -> float:
    values = _finite(rows, key)
    return float(np.mean(values)) if values else float("nan")


def _std(rows: list[dict], key: str, *, ddof: int = 0) -> float:
    values = _finite(rows, key)
    if not values:
        return float("nan")
    if ddof and len(values) <= ddof:
        return float("nan")
    return float(np.std(values, ddof=ddof))


def _manifest_map(manifest: Path) -> dict[str, dict[str, str]]:
    rows = utils.load_manifest(manifest)
    return {str(row["sample_id"]): row for row in rows}


def _seed_dir(seed) -> str:
    text = str(seed).strip()
    if text == "deterministic":
        return "deterministic"
    try:
        return f"seed_{int(float(text))}"
    except ValueError as exc:
        raise ValueError(f"unsupported seed label: {seed!r}") from exc


def _prediction_dir(kind: str, benchmark_root: Path, row: dict) -> Path:
    sample_id = str(row["sample_id"])
    if kind in {"e2", "stage_b", "cv5"}:
        method = str(row["method"])
        return benchmark_root / method / _seed_dir(row["seed"]) / sample_id
    if kind == "e3":
        variant = str(row["variant"])
        return benchmark_root / variant / _seed_dir(row["seed"]) / sample_id
    raise ValueError(f"unknown target kind: {kind}")


def _augment_rows(
    *,
    kind: str,
    label: str,
    metrics_path: Path,
    benchmark_root: Path,
    manifest_path: Path,
    out_dir: Path,
    allow_incomplete: bool,
) -> dict:
    source_rows = _read_csv(metrics_path)
    sample_rows = _manifest_map(manifest_path)
    sample_cache: dict[str, dict] = {}
    augmented: list[dict] = []
    missing: list[dict] = []

    for source in source_rows:
        sample_id = str(source["sample_id"])
        manifest_row = sample_rows.get(sample_id)
        if manifest_row is None:
            missing.append({"label": label, "sample_id": sample_id, "reason": "missing_manifest_row"})
            continue

        pred_dir = _prediction_dir(kind, benchmark_root, source)
        background_path = pred_dir / "background_pred.tif"
        if not background_path.is_file():
            missing.append({
                "label": label,
                "sample_id": sample_id,
                "method_or_variant": source.get("method", source.get("variant", "")),
                "seed": source.get("seed", ""),
                "reason": "missing_background_pred",
                "expected_path": background_path.relative_to(PROJECT_ROOT).as_posix(),
            })
            continue

        sample = sample_cache.get(sample_id)
        if sample is None:
            sample = utils.load_synthetic_sample(manifest_row)
            sample_cache[sample_id] = sample

        background = utils.load_float_tiff(background_path)
        if background.shape != sample["observed"].shape:
            raise ValueError(
                f"shape mismatch for {label} {sample_id}: "
                f"background={background.shape}, observed={sample['observed'].shape}"
            )
        residual = utils.reconstruct_residual(sample["observed"], background)
        forced = forced_reference_metrics(sample, residual)
        if int(forced["forced_reference_flux_count"]) <= 0:
            raise ValueError(f"no valid forced-reference stars for {label} {sample_id}")

        item = dict(source)
        item.update(forced)
        item["frozen_background_pred"] = background_path.relative_to(PROJECT_ROOT).as_posix()
        item["frozen_background_pred_sha256"] = _sha256(background_path)
        augmented.append(item)

    if missing and not allow_incomplete:
        preview = "\n".join(json.dumps(x, ensure_ascii=False) for x in missing[:12])
        raise FileNotFoundError(
            f"{label}: {len(missing)} frozen prediction/sample rows are missing. "
            f"Re-run with --allow-incomplete only for diagnosis.\n{preview}"
        )

    _write_csv(out_dir / "augmented_metrics.csv", augmented)
    _write_csv(out_dir / "missing_rows.csv", missing)

    grouping_key = "variant" if kind == "e3" else "method"
    metric_keys = list(COMMON_PAPER_METRICS)
    if kind == "stage_b":
        metric_keys.extend(STAGE_B_EXTRA_METRICS)
    summary = []
    groups = sorted({str(row[grouping_key]) for row in augmented})
    for group in groups:
        subset = [row for row in augmented if str(row[grouping_key]) == group]
        rec = {grouping_key: group, "row_count": len(subset)}
        for key in metric_keys:
            rec[key] = _mean(subset, key)
            rec[key + "_std"] = _std(subset, key)
        counts = _finite(subset, "forced_reference_flux_count")
        rec["forced_reference_flux_count_min"] = int(min(counts)) if counts else 0
        rec["forced_reference_flux_count_max"] = int(max(counts)) if counts else 0
        summary.append(rec)
    _write_csv(out_dir / ("variant_summary.csv" if kind == "e3" else "method_summary.csv"), summary)

    return {
        "label": label,
        "kind": kind,
        "metrics_path": metrics_path.relative_to(PROJECT_ROOT).as_posix(),
        "metrics_sha256": _sha256(metrics_path),
        "manifest_path": manifest_path.relative_to(PROJECT_ROOT).as_posix(),
        "manifest_sha256": _sha256(manifest_path),
        "source_row_count": len(source_rows),
        "computed_row_count": len(augmented),
        "missing_row_count": len(missing),
        "output_dir": out_dir.relative_to(PROJECT_ROOT).as_posix(),
    }


def _cv5_summary() -> None:
    fold_means: list[dict] = []
    for fold in range(5):
        path = OUTPUT_ROOT / "cv5" / f"fold_{fold}" / "augmented_metrics.csv"
        if not path.is_file():
            continue
        rows = _read_csv(path)
        for method in sorted({row["method"] for row in rows}):
            subset = [row for row in rows if row["method"] == method]
            rec = {"fold": fold, "method": method, "row_count": len(subset)}
            for metric in COMMON_PAPER_METRICS:
                rec[metric] = _mean(subset, metric)
            fold_means.append(rec)

    _write_csv(OUTPUT_ROOT / "cv5" / "fold_method_summary.csv", fold_means)

    overall: list[dict] = []
    for method in sorted({row["method"] for row in fold_means}):
        subset = [row for row in fold_means if row["method"] == method]
        rec = {"method": method, "fold_count": len(subset)}
        for metric in COMMON_PAPER_METRICS:
            vals = _finite(subset, metric)
            rec[f"{metric}_mean_across_folds"] = float(np.mean(vals)) if vals else float("nan")
            rec[f"{metric}_std_across_folds"] = (
                float(np.std(vals, ddof=1)) if len(vals) >= 2 else float("nan")
            )
        overall.append(rec)
    _write_csv(OUTPUT_ROOT / "cv5" / "method_summary_across_folds.csv", overall)


def _check_final_pinn(path: Path, label: str) -> dict:
    if not path.is_file():
        return {
            "label": label,
            "metrics_path": path.relative_to(PROJECT_ROOT).as_posix(),
            "exists": False,
            "row_count": 0,
            "finite_forced_mean_count": 0,
            "finite_forced_p95_count": 0,
        }
    rows = _read_csv(path)
    return {
        "label": label,
        "metrics_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "metrics_sha256": _sha256(path),
        "exists": True,
        "row_count": len(rows),
        "finite_forced_mean_count": len(_finite(rows, "forced_reference_mean_relative_flux_error")),
        "finite_forced_p95_count": len(_finite(rows, "forced_reference_p95_relative_flux_error")),
    }


def _prepare_output(force: bool) -> None:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        if not force:
            raise FileExistsError(
                f"derived backfill output already exists: {OUTPUT_ROOT}. "
                "Use --force to replace only this derived namespace."
            )
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("a", "b", "cv5", "e3"),
        default=("a", "b", "cv5", "e3"),
        help="derived evaluation targets; default: a b cv5 e3",
    )
    parser.add_argument("--force", action="store_true", help="replace only forced_reference_backfill outputs")
    parser.add_argument("--allow-incomplete", action="store_true", help="diagnostic only; do not use for final paper tables")
    args = parser.parse_args()

    _prepare_output(args.force)
    targets = tuple(dict.fromkeys(args.targets))
    records: list[dict] = []

    a_manifest = PROJECT_ROOT / "collections" / "synthetic_formal" / "manifest.csv"
    b_manifest = PROJECT_ROOT / "collections" / "synthetic_stage_b" / "manifest.csv"
    outputs = PROJECT_ROOT / "experiments" / "outputs"

    if "a" in targets:
        records.append(
            _augment_rows(
                kind="e2",
                label="stage_a_e2",
                metrics_path=outputs / "e2_benchmark" / "metrics.csv",
                benchmark_root=outputs / "e2_benchmark",
                manifest_path=a_manifest,
                out_dir=OUTPUT_ROOT / "stage_a_e2",
                allow_incomplete=args.allow_incomplete,
            )
        )

    if "e3" in targets:
        records.append(
            _augment_rows(
                kind="e3",
                label="stage_a_e3",
                metrics_path=outputs / "e3_ablation" / "metrics.csv",
                benchmark_root=outputs / "e3_ablation",
                manifest_path=a_manifest,
                out_dir=OUTPUT_ROOT / "stage_a_e3",
                allow_incomplete=args.allow_incomplete,
            )
        )

    if "b" in targets:
        records.append(
            _augment_rows(
                kind="stage_b",
                label="stage_b_benchmark",
                metrics_path=outputs / "stage_b" / "benchmark" / "metrics.csv",
                benchmark_root=outputs / "stage_b" / "benchmark",
                manifest_path=b_manifest,
                out_dir=OUTPUT_ROOT / "stage_b_benchmark",
                allow_incomplete=args.allow_incomplete,
            )
        )

    if "cv5" in targets:
        for fold in range(5):
            manifest = PROJECT_ROOT / "collections" / "synthetic_formal" / "cv5" / f"fold_{fold}" / "manifest.csv"
            root = outputs / "cv5" / f"fold_{fold}" / "e2_benchmark"
            records.append(
                _augment_rows(
                    kind="cv5",
                    label=f"cv5_fold_{fold}",
                    metrics_path=root / "metrics.csv",
                    benchmark_root=root,
                    manifest_path=manifest,
                    out_dir=OUTPUT_ROOT / "cv5" / f"fold_{fold}",
                    allow_incomplete=args.allow_incomplete,
                )
            )
        _cv5_summary()

    final_checks = [
        _check_final_pinn(outputs / "final_pinn_validation" / "metrics.csv", "stage_a_final_pinn"),
        _check_final_pinn(outputs / "stage_b" / "final_pinn_validation" / "metrics.csv", "stage_b_final_pinn"),
    ]
    for fold in range(5):
        final_checks.append(
            _check_final_pinn(
                outputs / "cv5" / f"fold_{fold}" / "final_pinn_validation" / "metrics.csv",
                f"cv5_fold_{fold}_final_pinn",
            )
        )

    _write_csv(OUTPUT_ROOT / "coverage_report.csv", records)
    _write_csv(OUTPUT_ROOT / "existing_final_pinn_coverage.csv", final_checks)
    metadata = {
        "schema": 1,
        "scientific_status": "posthoc_metric_backfill_on_frozen_predictions",
        "formal_result_replacement": False,
        "selection_or_training_performed": False,
        "prediction_arrays_modified": False,
        "formal_metric_rule": {
            "mean": "forced_reference_mean_relative_flux_error",
            "p95": "forced_reference_p95_relative_flux_error",
            "reference_coordinates": "frozen sample metadata star_reference coordinates",
            "reference_flux": "frozen sample metadata raw_aperture_flux",
            "prediction_flux": "existing metrics.measure_raw_aperture_flux on reconstructed raw residual",
            "residual": "observed - frozen background_pred.tif",
            "clipping": False,
        },
        "targets": list(targets),
        "script": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
        "script_sha256": _sha256(Path(__file__)),
        "coverage": records,
        "existing_final_pinn_coverage": final_checks,
    }
    _write_json(OUTPUT_ROOT / "run_metadata.json", metadata)

    print("Forced-reference metric backfill complete.")
    for rec in records:
        print(
            f"  {rec['label']}: computed={rec['computed_row_count']} "
            f"missing={rec['missing_row_count']}"
        )
    print("Output:", OUTPUT_ROOT)
    if args.allow_incomplete:
        print("WARNING: --allow-incomplete was used; do not use incomplete summaries for final paper tables.")


if __name__ == "__main__":
    main()
