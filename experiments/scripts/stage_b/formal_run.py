"""Formal Stage10-B benchmark.

This runner opens only the Stage-B test split after the Stage-B Global Lock
exists. All B-specific selection/training has already finished.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch

from experiments.common.baselines import unet_small as unet
from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import stage_b_metrics
from experiments.common.utils import stage_b_provenance as provenance
from experiments.configs.formal_experiment import FORMAL_EXECUTION_CONFIG
from experiments.configs.stage_b_experiment import STAGE_B_CONFIG, stage_b_config_snapshot
from experiments.scripts.benchmark import run_method
from experiments.scripts.stage_b.lock import LOCK_PATH, FORMAL_SOURCE_FILES

OUTPUT_ROOT = utils.OUTPUT_ROOT / "stage_b/benchmark"
METHODS = ("fft_gaussian", "fourier_ls", "unet_small", "pinn")


def _load_lock():
    lock = utils.read_json(LOCK_PATH)
    if lock.get("lock_state") != "FROZEN" or not lock.get("formal_test_ready"):
        raise RuntimeError("Stage-B Global Lock is not formal-test ready")
    if utils.file_sha256(STAGE_B_MANIFEST) != lock["data_provenance"]["manifest_sha256"]:
        raise ValueError("Stage-B manifest changed after lock")
    provenance.verify_source_snapshot(lock["source_provenance"], FORMAL_SOURCE_FILES)
    for method, info in lock["selection_artifacts"].items():
        path = utils.to_absolute_path(info["path"])
        if utils.file_sha256(path) != info["sha256"]:
            raise ValueError(f"Stage-B selection artifact changed after lock: {method}")
    return lock


def _load_sample(row):
    sample = utils.load_synthetic_sample(row)
    sample["base_clean"] = utils.load_float_tiff(row["base_clean"])
    sample["protected_true"] = utils.load_float_tiff(row["protected_true"])
    sample["scene_type"] = row.get("scene_type", row["background_type"])
    return sample


def _with_residual(observed, result):
    result = dict(result)
    result["residual_pred"] = utils.reconstruct_residual(observed, result["background_pred"])
    return result


def _save_result(output_dir, sample, result):
    utils.save_prediction_arrays(
        output_dir,
        sample["observed"],
        result["background_pred"],
        result["residual_pred"],
    )


def _metric_row(method, seed, sample, result):
    evaluated = stage_b_metrics.evaluate_stage_b(sample, result)
    background = np.asarray(result["background_pred"], dtype=np.float32)
    row = {
        "method": method,
        "seed": seed,
        "sample_id": sample["sample_id"],
        "base_id": sample["base_id"],
        "scene_type": sample["scene_type"],
        "background_type": sample["scene_type"],
        "intensity_level": sample["intensity_level"],
        "runtime_s": result["runtime_s"],
        "peak_vram_mb": result.get("peak_vram_mb"),
        "parameter_count": result.get("parameter_count", 0),
        "background_below_zero_fraction": float(np.mean(background < 0.0)),
        "background_above_one_fraction": float(np.mean(background > 1.0)),
        **evaluated,
    }
    if "fourier_info" in result:
        row.update({
            "fourier_objective": result["fourier_info"]["objective"],
            "fourier_profile_nfev": result["fourier_info"]["profile_nfev"],
            "fourier_effective_component_count": result["fourier_info"]["effective_component_count"],
        })
    return row


def _finite_values(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return values


def _mean(rows, key):
    values = _finite_values(rows, key)
    return float(np.mean(values)) if values else float("nan")


def _std(rows, key):
    values = _finite_values(rows, key)
    return float(np.std(values)) if values else float("nan")


def _main_table(rows, training_rows):
    metric_keys = (
        "bg_mae",
        "bg_ssim",
        "residual_ssim",
        "star_f1",
        "matched_star_mean_relative_flux_error",
        "faint_star_recall",
        "protected_region_mae",
        "protected_projection_abs_error",
        "runtime_s",
    )
    table = []
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        item = {"method": method, "count": len(subset)}
        for key in metric_keys:
            item[key] = _mean(subset, key)
            item[key + "_std"] = _std(subset, key)
        train_times = [
            float(r["training_time_s"])
            for r in training_rows
            if r["method"] == method and r.get("training_time_s") is not None
        ]
        item["train_time_s"] = float(np.mean(train_times)) if train_times else "N/A"
        table.append(item)
    return table


def _grouped_table(rows):
    table = []
    for method in METHODS:
        for scene in STAGE_B_CONFIG["scene_types"]:
            for level in STAGE_B_CONFIG["intensity_levels"]:
                subset = [
                    r for r in rows
                    if r["method"] == method
                    and r["scene_type"] == scene
                    and r["intensity_level"] == level
                ]
                if not subset:
                    continue
                table.append({
                    "method": method,
                    "scene_type": scene,
                    "intensity_level": level,
                    "count": len(subset),
                    "bg_mae": _mean(subset, "bg_mae"),
                    "residual_ssim": _mean(subset, "residual_ssim"),
                    "star_f1": _mean(subset, "star_f1"),
                    "faint_star_recall": _mean(subset, "faint_star_recall"),
                    "flux_error": _mean(subset, "matched_star_mean_relative_flux_error"),
                    "protected_region_mae": _mean(subset, "protected_region_mae"),
                    "protected_projection_abs_error": _mean(subset, "protected_projection_abs_error"),
                })
    return table


def _failure_cases(rows):
    output = []
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        candidates = [
            ("largest_bg_mae", max(subset, key=lambda r: r["bg_mae"])),
            ("largest_protected_error", max(subset, key=lambda r: r["protected_projection_abs_error"])),
            ("lowest_faint_star_recall", min(subset, key=lambda r: r["faint_star_recall"])),
        ]
        seen = set()
        for reason, row in candidates:
            ident = (row["seed"], row["sample_id"])
            if ident in seen:
                continue
            seen.add(ident)
            output.append({
                "method": method,
                "reason": reason,
                "seed": row["seed"],
                "sample_id": row["sample_id"],
                "scene_type": row["scene_type"],
                "intensity_level": row["intensity_level"],
                "bg_mae": row["bg_mae"],
                "star_f1": row["star_f1"],
                "faint_star_recall": row["faint_star_recall"],
                "matched_star_mean_relative_flux_error": row["matched_star_mean_relative_flux_error"],
                "protected_region_mae": row["protected_region_mae"],
                "protected_projection_abs_error": row["protected_projection_abs_error"],
            })
    return output


def main(*, force=False, smoke=False):
    lock = _load_lock()
    output_root = utils.OUTPUT_ROOT / ("stage_b/smoke_after_lock" if smoke else "stage_b/benchmark")
    output_root = utils.prepare_output_root(output_root, force=force)

    # Open B test arrays only after the verified B lock.
    test_rows = utils.load_manifest(STAGE_B_MANIFEST, split="test")
    if smoke:
        test_rows = test_rows[:1]
    if not test_rows:
        raise ValueError("Stage-B test split is empty")
    samples = [_load_sample(row) for row in test_rows]

    device = torch.device(str(FORMAL_EXECUTION_CONFIG["benchmark_device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Formal Stage-B requires the frozen CUDA execution policy")

    utils.save_run_metadata(output_root, {
        "experiment": "Stage10-B adapted formal benchmark",
        "stage_b_config": stage_b_config_snapshot(),
        "stage_b_global_lock": utils.project_relative(LOCK_PATH),
        "stage_b_global_lock_sha256": utils.file_sha256(LOCK_PATH),
        "smoke": bool(smoke),
        "test_samples": [r["sample_id"] for r in test_rows],
        "method_policy": "Stage-B train/validation selection completed before Stage-B lock",
    })
    utils.write_json(output_root / "stage_b_global_lock_snapshot.json", lock)

    rows, training_rows = [], []
    configs = lock["methods"]

    for sample in samples:
        result = _with_residual(
            sample["observed"],
            run_method("fft_gaussian", sample["observed"], configs["fft_gaussian"], device=device),
        )
        _save_result(output_root / "fft_gaussian/deterministic" / sample["sample_id"], sample, result)
        rows.append(_metric_row("fft_gaussian", "deterministic", sample, result))

    for i, sample in enumerate(samples, 1):
        result = _with_residual(
            sample["observed"],
            run_method(
                "fourier_ls",
                sample["observed"],
                configs["fourier_ls"],
                device=device,
                show_progress=True,
                progress_desc=f"Stage-B Cosine-LS [{i}/{len(samples)}] {sample['sample_id']}",
            ),
        )
        out = output_root / "fourier_ls/deterministic" / sample["sample_id"]
        _save_result(out, sample, result)
        if "fourier_info" in result:
            utils.write_json(out / "estimated_parameters.json", result["fourier_info"])
        rows.append(_metric_row("fourier_ls", "deterministic", sample, result))

    checkpoints = configs["unet_small"]["checkpoints"]
    if smoke:
        checkpoints = checkpoints[:1]
    for info in checkpoints:
        seed = int(info["seed"])
        checkpoint = utils.to_absolute_path(info["checkpoint"])
        if utils.file_sha256(checkpoint) != info["sha256"]:
            raise ValueError(f"Stage-B U-Net checkpoint changed after lock: {checkpoint}")
        model, checkpoint_data = unet.load_checkpoint_model(checkpoint, device)
        if abs(float(checkpoint_data["validation_loss"]) - float(info["validation_loss"])) > 1e-12:
            raise ValueError(f"Stage-B U-Net checkpoint metadata mismatch: {checkpoint}")
        pcount = utils.parameter_count(model)
        training_rows.append({"method": "unet_small", **info})
        for sample in samples:
            result = _with_residual(
                sample["observed"],
                run_method(
                    "unet_small",
                    sample["observed"],
                    configs["unet_small"],
                    unet_model=model,
                    unet_parameter_count=pcount,
                    device=device,
                ),
            )
            _save_result(output_root / "unet_small" / f"seed_{seed}" / sample["sample_id"], sample, result)
            rows.append(_metric_row("unet_small", seed, sample, result))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pinn_seeds = list(configs["pinn"]["test_seeds"])
    if smoke:
        pinn_seeds = pinn_seeds[:1]
    for seed in pinn_seeds:
        seed = int(seed)
        for sample in samples:
            result = _with_residual(
                sample["observed"],
                run_method(
                    "pinn",
                    sample["observed"],
                    configs["pinn"],
                    device=device,
                    seed=seed,
                    return_state=True,
                ),
            )
            out = output_root / "pinn" / f"seed_{seed}" / sample["sample_id"]
            _save_result(out, sample, result)
            if "pinn_result" in result:
                utils.save_rows_csv(out / "history.csv", result["pinn_result"]["history"])
                if "state" in result["pinn_result"]:
                    torch.save(
                        {
                            "model_state": result["pinn_result"]["state"]["model_state"],
                            "forcing_state": result["pinn_result"]["state"]["forcing_state"],
                            "step": result["pinn_result"]["state"]["step"],
                        },
                        out / "final_model_forcing_state.pt",
                    )
            rows.append(_metric_row("pinn", seed, sample, result))

    main_table = _main_table(rows, training_rows)
    grouped = _grouped_table(rows)
    failures = _failure_cases(rows)

    utils.save_rows_csv(output_root / "metrics.csv", rows)
    utils.save_rows_csv(output_root / "training.csv", training_rows)
    utils.save_rows_csv(output_root / "main_result_table.csv", main_table)
    utils.save_rows_csv(output_root / "grouped_results.csv", grouped)
    utils.save_rows_csv(output_root / "failure_cases.csv", failures)
    utils.write_json(output_root / "summary.json", utils.summarize_rows(rows, "method"))
    print("Stage10-B formal benchmark complete:", output_root / "main_result_table.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="after-lock diagnostic only")
    args = parser.parse_args()
    main(force=args.force, smoke=args.smoke)
