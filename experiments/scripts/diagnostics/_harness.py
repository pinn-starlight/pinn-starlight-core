"""Shared plumbing for the non-formal mechanism probes.

Every probe is validation-only, CPU-capable and writes into a fresh
``experiments/outputs/diagnostics/<probe>/<timestamp>_<suite>/`` directory.
Nothing here may be imported by a formal Stage-A/B/CV5/E4 runner.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics, scores
from experiments.configs.diagnostics import (
    DATASETS,
    DIAGNOSTIC_SCHEMA_VERSION,
    PROTECTED_MASK_THRESHOLD_FRACTION,
)

SCHEMA_VERSION = DIAGNOSTIC_SCHEMA_VERSION


# --------------------------------------------------------------------------
# guards and devices
# --------------------------------------------------------------------------
def require_standard_split() -> None:
    """Diagnostics are standard-split only; CV5 environment variables are an error."""
    if utils.SPLIT_MODE != "standard":
        raise RuntimeError(
            "mechanism probes only support the standard split; "
            f"got PINN_STARLIGHT_SPLIT_MODE={utils.SPLIT_MODE!r} (fold {utils.CV_FOLD})"
        )


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a diagnostic device; unlike the formal runners, CPU is allowed."""
    text = str(name).strip().lower()
    if text in {"auto", ""}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"device {text!r} requested but torch.cuda.is_available() is False; "
            "use --device cpu for a smoke run"
        )
    return device


# --------------------------------------------------------------------------
# data (validation split only)
# --------------------------------------------------------------------------
def manifest_for(dataset: str) -> Path:
    if dataset == "stage_a":
        return Path(utils.SYNTHETIC_MANIFEST)
    if dataset == "stage_b":
        from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST

        return Path(STAGE_B_MANIFEST)
    raise ValueError(f"unknown dataset: {dataset!r}; expected one of {DATASETS}")


def load_rows(dataset: str, samples: str | None = None) -> list[dict]:
    """Load validation rows; never returns train or test rows."""
    rows = utils.load_manifest(manifest_for(dataset), split="validation")
    if not rows:
        raise RuntimeError(f"{dataset} validation split is empty: {manifest_for(dataset)}")
    if samples:
        wanted = [item.strip() for item in str(samples).split(",") if item.strip()]
        by_id = {row["sample_id"]: row for row in rows}
        missing = [name for name in wanted if name not in by_id]
        if missing:
            available = ", ".join(sorted(by_id))
            raise ValueError(f"unknown validation sample(s): {missing}; available: {available}")
        rows = [by_id[name] for name in wanted]
    return rows


def stride_for(array: np.ndarray, max_side: int | None) -> int:
    if max_side is None:
        return 1
    longest = max(int(array.shape[0]), int(array.shape[1]))
    return max(1, int(np.ceil(longest / float(max_side))))


def _load_array(row: dict, key: str, stride: int) -> np.ndarray:
    array = utils.load_float_tiff(utils.to_absolute_path(row[key]))
    if stride > 1:
        array = array[::stride, ::stride]
    return np.ascontiguousarray(array, dtype=np.float32)


def load_sample(row: dict, dataset: str, max_side: int | None = None) -> dict:
    """Load one validation sample, optionally stride-downsampled.

    Ground truth stays next to the input; callers must pass only
    ``sample["observed"]`` to a method.
    """
    observed_full = utils.load_float_tiff(utils.to_absolute_path(row["observed"]))
    stride = stride_for(observed_full, max_side)
    sample = {
        "sample_id": row["sample_id"],
        "base_id": row.get("base_id"),
        "split": row["split"],
        "dataset": dataset,
        "background_type": row.get("background_type"),
        "scene_type": row.get("scene_type", row.get("background_type")),
        "intensity_level": row.get("intensity_level"),
        "stride": int(stride),
        "observed": _load_array(row, "observed", stride),
        "background_true": _load_array(row, "background_true", stride),
        "clean_true": _load_array(row, "clean_true", stride),
        "metadata": utils.read_json(utils.to_absolute_path(row["metadata"])),
    }
    if dataset == "stage_b":
        sample["base_clean"] = _load_array(row, "base_clean", stride)
        sample["protected_true"] = _load_array(row, "protected_true", stride)
    return sample


def check_validation(sample: dict) -> None:
    if sample.get("split") != "validation":
        raise RuntimeError(f"probe received a non-validation sample: {sample.get('sample_id')}")


def _required_keys(dataset: str) -> tuple[str, ...]:
    if dataset == "stage_b":
        return ("observed", "background_true", "clean_true", "base_clean", "protected_true")
    return ("observed", "background_true", "clean_true")


def missing_files(row: dict, dataset: str) -> list[str]:
    missing = []
    for key in _required_keys(dataset):
        path = utils.to_absolute_path(row[key])
        if not Path(path).is_file():
            missing.append(str(utils.project_relative(path)))
    return missing


def require_available(
    rows: list[dict],
    dataset: str,
    *,
    allow_subset: bool,
    limit: int | None = None,
) -> list[dict]:
    """Fail early with an actionable message when the arrays are not on disk.

    Smoke runs may use whichever validation samples happen to be extracted;
    full runs must have every requested sample, because silently dropping rows
    would make the summary incomparable.
    """
    if allow_subset:
        available = [row for row in rows if not missing_files(row, dataset)]
        if not available:
            raise RuntimeError(
                f"none of the requested {dataset} validation samples is on disk.\n"
                "Run: bash tools/fetch_smoke_data.sh\n"
                f"manifest: {utils.project_relative(manifest_for(dataset))}"
            )
        return available if limit is None else available[:limit]

    incomplete = {row["sample_id"]: missing_files(row, dataset) for row in rows}
    incomplete = {name: files for name, files in incomplete.items() if files}
    if incomplete:
        detail = "\n".join(f"  {name}: {files}" for name, files in incomplete.items())
        raise RuntimeError(
            f"missing {dataset} validation arrays for a full run:\n{detail}\n"
            "Restore the dataset (server archive or PINN_STARLIGHT_DATA_ROOT) first."
        )
    return rows if limit is None else rows[:limit]


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def projection_metrics(sample: dict, residual_pred: np.ndarray) -> dict:
    """Projection of the recovered structure onto the protected structure.

    Kept inline (instead of importing the private Stage-B helper) so the
    diagnostic can run on a stride-downsampled grid.  A smoke test asserts it
    matches ``stage_b_metrics.evaluate_stage_b`` exactly on stride 1.
    """
    protected = np.asarray(sample["protected_true"], dtype=np.float64)
    base_clean = np.asarray(sample["base_clean"], dtype=np.float64)
    recovered = np.asarray(residual_pred, dtype=np.float64) - base_clean
    peak = float(np.max(protected))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError("protected_true must have a positive finite peak")
    mask = protected >= (PROTECTED_MASK_THRESHOLD_FRACTION * peak)
    denominator = float(np.sum(protected * protected))
    ratio = float(np.sum(recovered * protected) / max(denominator, 1e-20))
    return {
        "protected_region_mae": float(np.mean(np.abs(recovered[mask] - protected[mask]))),
        "protected_projection_ratio": ratio,
        "protected_projection_abs_error": float(abs(ratio - 1.0)),
        "protected_mask_fraction": float(np.mean(mask)),
    }


def _base_metrics(sample: dict, background_pred: np.ndarray) -> tuple[dict, np.ndarray]:
    residual = utils.reconstruct_residual(sample["observed"], background_pred)
    result = {
        "bg_mae": scores.mae(background_pred, sample["background_true"]),
        "bg_rmse": scores.rmse(background_pred, sample["background_true"]),
        "bg_psnr": scores.psnr(background_pred, sample["background_true"]),
        "bg_ssim": scores.ssim(background_pred, sample["background_true"]),
        "residual_mae": scores.mae(residual, sample["clean_true"]),
        "residual_rmse": scores.rmse(residual, sample["clean_true"]),
        "residual_psnr": scores.psnr(residual, sample["clean_true"]),
        "residual_ssim": scores.ssim(residual, sample["clean_true"]),
    }
    return result, residual


def evaluate(sample: dict, background_pred: np.ndarray, *, allow_star_metrics: bool = True) -> dict:
    """Evaluate one prediction without ever inventing star metrics.

    On a stride-1 image the formal Stage-B/Stage-A evaluator is used verbatim.
    On a downsampled grid only background/residual/projection metrics are
    reported and ``star_metrics_skipped`` is set.
    """
    check_validation(sample)
    background_pred = np.asarray(background_pred, dtype=np.float32)
    if background_pred.shape != sample["observed"].shape:
        raise ValueError(
            f"background shape {background_pred.shape} != observed {sample['observed'].shape}"
        )
    if not np.isfinite(background_pred).all():
        raise ValueError("background_pred contains NaN/Inf")

    if sample["stride"] == 1 and allow_star_metrics:
        prediction = {
            "background_pred": background_pred,
            "residual_pred": utils.reconstruct_residual(sample["observed"], background_pred),
        }
        if sample["dataset"] == "stage_b":
            from experiments.common.utils import stage_b_metrics

            result = stage_b_metrics.evaluate_stage_b(sample, prediction)
        else:
            result = metrics.evaluate_synthetic(sample, prediction)
        result["star_metrics_skipped"] = False
        return result

    result, residual = _base_metrics(sample, background_pred)
    if sample["dataset"] == "stage_b":
        result.update(projection_metrics(sample, residual))
    if sample["stride"] != 1:
        result["star_metrics_skipped"] = True
        result["star_metrics_note"] = f"stride={sample['stride']}"
    else:
        result["star_metrics_skipped"] = True
        result["star_metrics_note"] = "disabled"
    return result


# --------------------------------------------------------------------------
# trajectories
# --------------------------------------------------------------------------
def trajectory_stats(history: list[dict], key: str = "physics_loss") -> dict:
    """Summarise when a logged loss falls fastest.

    Decay is measured in log-space per optimiser step so runs with different
    ``log_every`` remain comparable.  ``t50`` / ``t90`` are the first logged
    steps at or below 50% / 10% of the first finite value.
    """
    points = [
        (int(row["step"]), float(row[key]))
        for row in history
        if row.get(key) is not None and np.isfinite(float(row[key])) and float(row[key]) > 0.0
    ]
    points.sort()
    if len(points) < 2:
        return {
            "n_logged": len(points),
            "first_value": float(points[0][1]) if points else float("nan"),
            "last_value": float(points[-1][1]) if points else float("nan"),
            "max_log_decay_per_step": float("nan"),
            "step_at_max_decay": float("nan"),
            "t50_step": float("nan"),
            "t90_step": float("nan"),
            "final_relative": float("nan"),
        }

    first_step, first_value = points[0]
    last_step, last_value = points[-1]
    best_decay = -float("inf")
    best_step = float("nan")
    for (step0, value0), (step1, value1) in zip(points, points[1:]):
        span = max(1, step1 - step0)
        decay = (np.log(value0) - np.log(value1)) / span
        if decay > best_decay:
            best_decay = float(decay)
            best_step = float(step1)

    def first_at(fraction: float) -> float:
        threshold = fraction * first_value
        for step, value in points:
            if value <= threshold:
                return float(step)
        return float(last_step)

    return {
        "n_logged": len(points),
        "first_step": int(first_step),
        "first_value": float(first_value),
        "last_step": int(last_step),
        "last_value": float(last_value),
        "max_log_decay_per_step": float(best_decay),
        "step_at_max_decay": best_step,
        "t50_step": first_at(0.5),
        "t90_step": first_at(0.1),
        "final_relative": float(last_value / first_value),
    }


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------
def new_run_dir(probe: str, suite: str, out_root: str | Path | None = None) -> Path:
    """Create a fresh run directory; never deletes an existing one."""
    root = Path(out_root) if out_root else utils.BASE_OUTPUT_ROOT / "diagnostics"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / probe / f"{stamp}_{suite}"
    suffix = 1
    while path.exists():
        suffix += 1
        path = root / probe / f"{stamp}_{suite}_{suffix:02d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_rows(path: str | Path, rows: list[dict]) -> None:
    utils.save_rows_csv(Path(path), rows)


def write_json(path: str | Path, payload: dict) -> None:
    utils.write_json(Path(path), payload)


def git_state() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=utils.PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=utils.PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except Exception:  # noqa: BLE001 - provenance is best effort
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def run_metadata(
    *,
    probe: str,
    suite: str,
    dataset: str,
    device: torch.device,
    seed: int,
    budgets: dict,
    samples: list[dict],
    args: dict | None = None,
    extra: dict | None = None,
) -> dict:
    manifest = manifest_for(dataset)
    return {
        "schema_version": SCHEMA_VERSION,
        "probe": probe,
        "suite": suite,
        "dataset": dataset,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git_state(),
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "seed": int(seed),
        "budgets": dict(budgets),
        "data": {
            "manifest": utils.project_relative(manifest),
            "manifest_sha256": utils.file_sha256(manifest),
            "split": "validation",
            "samples": [sample["sample_id"] for sample in samples],
            "strides": {sample["sample_id"]: sample["stride"] for sample in samples},
        },
        "args": dict(args or {}),
        "extra": dict(extra or {}),
        "note": "non-formal diagnostic; not a paper result",
    }


def summarise(
    rows: list[dict], group_keys: tuple[str, ...], value_keys: tuple[str, ...]
) -> list[dict]:
    """Mean/std of ``value_keys`` for every combination of ``group_keys``."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        group = groups[key]
        item = {name: value for name, value in zip(group_keys, key)}
        item["count"] = len(group)
        for name in value_keys:
            values = [
                float(row[name])
                for row in group
                if row.get(name) is not None and np.isfinite(float(row[name]))
            ]
            item[f"{name}_mean"] = float(np.mean(values)) if values else float("nan")
            item[f"{name}_std"] = float(np.std(values)) if values else float("nan")
            item[f"{name}_n"] = len(values)
        output.append(item)
    return output
