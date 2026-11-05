"""E2: frozen-method benchmark on the synthetic test split only.

All global selection/training must be completed by Global Configuration Lock.
This runner never loads train/validation samples and never chooses a checkpoint.
"""

import argparse
import time

import numpy as np
import torch

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.baselines import fft_gaussian as fft
from experiments.common.baselines import cosine_ls as fourier_ls
from experiments.common.baselines import unet_small as unet
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.common.utils import reporting
from experiments.configs.formal_experiment import FORMAL_EXECUTION_CONFIG, FORMAL_METHODS


METHODS = tuple(FORMAL_METHODS)
OUTPUT_ROOT = utils.OUTPUT_ROOT / "e2_benchmark"
MANIFEST = utils.SYNTHETIC_MANIFEST
GLOBAL_LOCK = utils.GLOBAL_CONFIGURATION_LOCK
MAX_TEST_SAMPLES = 0


def run_method(method: str, observed, method_config: dict, **kwargs) -> dict:
    """Run one frozen test Method from observed + its pre-test frozen state."""
    observed = utils.load_observed(observed)
    device = kwargs.get("device", torch.device("cpu"))
    utils.reset_peak_vram(device)
    started = time.perf_counter()
    if method == "fft_gaussian":
        background_pred = fft.estimate_background(
            observed, float(method_config["sigma"])
        )
        extra = {"parameter_count": 0, "peak_vram_mb": None}
    elif method == "fourier_ls":
        lambda_ls, sparse_weight = fourier_ls.validate_frozen_config(method_config)
        background_pred, fourier_info = fourier_ls.estimate_background(
            observed,
            lambda_ls,
            sparse_weight,
            return_info=True,
            backend=FORMAL_EXECUTION_CONFIG["fourier_ls_backend"],
            show_progress=bool(kwargs.get("show_progress", False)),
            progress_desc=str(kwargs.get("progress_desc", "Cosine-LS Stage-10 profile BCD")),
        )
        extra = {
            "fourier_info": fourier_info,
            "parameter_count": 0,
            "peak_vram_mb": None,
        }
    elif method == "unet_small":
        inference = method_config["inference"]
        background_pred, inference_info = unet.predict_with_model(
            kwargs["unet_model"],
            observed,
            device=device,
            inference_mode=inference["mode"],
            tile_size=int(inference["tile_size"]),
            overlap=int(inference["overlap"]),
            return_info=True,
        )
        extra = {
            "parameter_count": kwargs["unet_parameter_count"],
            "peak_vram_mb": utils.peak_vram_mb(device),
            **inference_info,
        }
    elif method == "pinn":
        pinn_result = pinn.train_background(
            observed,
            config=method_config["config"],
            device=device,
            seed=kwargs["seed"],
            show_progress=True,
            laplacian_backend=FORMAL_EXECUTION_CONFIG["pinn_laplacian_backend"],
            return_state=bool(kwargs.get("return_state", False)),
        )
        background_pred = pinn_result["background_pred"]
        extra = {
            "pinn_result": pinn_result,
            "parameter_count": pinn_result["parameter_count"],
            "peak_vram_mb": pinn_result["peak_vram_mb"],
        }
    else:
        raise ValueError(f"未知方法：{method}")

    runtime_s = time.perf_counter() - started
    return {
        "background_pred": np.asarray(background_pred, dtype=np.float32),
        "runtime_s": float(runtime_s),
        "peak_vram_mb": extra["peak_vram_mb"],
        **extra,
    }



def _formal_device():
    """Resolve the frozen single-GPU Stage10-A benchmark device."""
    device = torch.device(str(FORMAL_EXECUTION_CONFIG["benchmark_device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Formal Stage10-A requires CUDA for the frozen benchmark policy; "
            "CUDA is not available."
        )
    return device


def main(*, force: bool = False):
    print("E2 Stage-10 frozen benchmark")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    device = _formal_device()
    global_lock = utils.load_global_configuration_lock(
        GLOBAL_LOCK, verify=True, require_formal_ready=True
    )
    method_configs = global_lock["methods"]

    # E2 opens only the held-out test split after the lock exists.
    test_rows = utils.load_manifest(MANIFEST, split="test")
    if MAX_TEST_SAMPLES > 0:
        test_rows = test_rows[:MAX_TEST_SAMPLES]
    if not test_rows:
        raise ValueError("test split must be non-empty")
    samples = [utils.load_synthetic_sample(row) for row in test_rows]

    utils.save_run_metadata(
        output_root,
        {
            "experiment": "E2",
            "manifest": utils.project_relative(MANIFEST),
            "global_configuration_lock": utils.project_relative(GLOBAL_LOCK),
            "global_configuration_lock_sha256": utils.file_sha256(GLOBAL_LOCK),
            "force_overwrite_output": force,
            "max_test_samples": MAX_TEST_SAMPLES,
            "methods": METHODS,
            "compute_device": str(device),
            "formal_execution_policy": dict(FORMAL_EXECUTION_CONFIG),
            "evaluation_schema_version": metrics.EVALUATION_SCHEMA_VERSION,
            "star_detection_rule": metrics.DETECTION_IMAGE_RULE,
            "flux_metric": metrics.MATCHED_STAR_FLUX_METRIC,
            "test_samples": [row["sample_id"] for row in test_rows],
        },
    )
    utils.write_json(output_root / "global_configuration_lock_snapshot.json", global_lock)

    metric_rows = []
    training_rows = []

    gaussian_config = method_configs["fft_gaussian"]
    for sample in samples:
        result = _with_common_residual(
            sample["observed"],
            run_method("fft_gaussian", sample["observed"], gaussian_config),
        )
        _save_result(
            output_root / "fft_gaussian/deterministic" / sample["sample_id"],
            sample,
            result,
        )
        metric_rows.append(_metric_row("fft_gaussian", "deterministic", sample, result))

    fourier_config = method_configs["fourier_ls"]
    for sample_index, sample in enumerate(samples, start=1):
        result = _with_common_residual(
            sample["observed"],
            run_method(
                "fourier_ls",
                sample["observed"],
                fourier_config,
                show_progress=True,
                progress_desc=(
                    f"E2 Cosine-LS [{sample_index}/{len(samples)}] "
                    f"{sample['sample_id']}"
                ),
            ),
        )
        sample_output = (
            output_root / "fourier_ls/deterministic" / sample["sample_id"]
        )
        _save_result(sample_output, sample, result)
        utils.write_json(sample_output / "estimated_parameters.json", result["fourier_info"])
        metric_rows.append(_metric_row("fourier_ls", "deterministic", sample, result))

    unet_config = method_configs["unet_small"]
    for checkpoint_info in unet_config["checkpoints"]:
        seed = int(checkpoint_info["seed"])
        checkpoint = utils.to_absolute_path(checkpoint_info["checkpoint"])
        model, checkpoint_data = unet.load_checkpoint_model(checkpoint, device)
        parameter_count = utils.parameter_count(model)
        if abs(float(checkpoint_data["validation_loss"]) - float(checkpoint_info["validation_loss"])) > 1e-12:
            raise ValueError(f"Frozen U-Net checkpoint metadata changed: {checkpoint}")
        training_rows.append({"method": "unet_small", **checkpoint_info})

        for sample in samples:
            result = _with_common_residual(
                sample["observed"],
                run_method(
                    "unet_small",
                    sample["observed"],
                    unet_config,
                    unet_model=model,
                    unet_parameter_count=parameter_count,
                    device=device,
                ),
            )
            _save_result(
                output_root / "unet_small" / f"seed_{seed}" / sample["sample_id"],
                sample,
                result,
            )
            metric_rows.append(_metric_row("unet_small", seed, sample, result))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pinn_config = method_configs["pinn"]
    for seed in pinn_config["test_seeds"]:
        seed = int(seed)
        for sample in samples:
            result = _with_common_residual(
                sample["observed"],
                run_method(
                    "pinn",
                    sample["observed"],
                    pinn_config,
                    device=device,
                    seed=seed,
                    return_state=True,
                ),
            )
            sample_output = output_root / "pinn" / f"seed_{seed}" / sample["sample_id"]
            _save_result(sample_output, sample, result)
            pinn_result = result["pinn_result"]
            utils.save_rows_csv(sample_output / "history.csv", pinn_result["history"])
            utils.write_json(sample_output / "final_parameters.json", _pinn_parameters(pinn_result))
            if "state" in pinn_result:
                torch.save(
                    {
                        "model_state": pinn_result["state"]["model_state"],
                        "forcing_state": pinn_result["state"]["forcing_state"],
                        "step": pinn_result["state"]["step"],
                    },
                    sample_output / "final_model_forcing_state.pt",
                )
            metric_rows.append(_metric_row("pinn", seed, sample, result))

    summary = utils.summarize_rows(metric_rows, "method")
    grouped_table = _grouped_results(metric_rows)
    main_table = _main_result_table(metric_rows, training_rows)
    failure_cases = _failure_cases(metric_rows)
    low_high_consistency = _low_high_consistency(output_root, samples, metric_rows)

    utils.save_rows_csv(output_root / "metrics.csv", metric_rows)
    utils.save_rows_csv(output_root / "training.csv", training_rows)
    utils.save_rows_csv(output_root / "grouped_results.csv", grouped_table)
    utils.save_rows_csv(output_root / "main_result_table.csv", main_table)
    utils.save_rows_csv(output_root / "failure_cases.csv", failure_cases)
    utils.save_rows_csv(output_root / "low_high_consistency.csv", low_high_consistency)
    utils.write_json(output_root / "summary.json", summary)
    print(f"E2 完成：{output_root / 'main_result_table.csv'}")


def _with_common_residual(observed, method_result):
    """Attach R_hat outside the Method boundary using the shared reconstruction."""
    result = dict(method_result)
    result["residual_pred"] = utils.reconstruct_residual(
        observed, result["background_pred"]
    )
    return result


def _save_result(output_dir, sample, result):
    utils.save_prediction_arrays(
        output_dir,
        sample["observed"],
        result["background_pred"],
        result["residual_pred"],
    )


def _metric_row(method, seed, sample, result):
    scores = metrics.evaluate_synthetic(sample, result)
    background = np.asarray(result["background_pred"], dtype=np.float32)
    row = {
        "method": method,
        "seed": seed,
        "sample_id": sample["sample_id"],
        "base_id": sample["base_id"],
        "background_type": sample["background_type"],
        "intensity_level": sample["intensity_level"],
        "runtime_s": result["runtime_s"],
        "peak_vram_mb": result["peak_vram_mb"],
        "parameter_count": result["parameter_count"],
        "background_below_zero_fraction": float(np.mean(background < 0.0)),
        "background_above_one_fraction": float(np.mean(background > 1.0)),
        **scores,
    }
    if "fourier_info" in result:
        row.update(
            {
                "fourier_objective": result["fourier_info"]["objective"],
                "fourier_smooth_objective": result["fourier_info"]["smooth_objective"],
                "fourier_sparse_objective": result["fourier_info"]["sparse_objective"],
                "fourier_profile_nfev": result["fourier_info"]["profile_nfev"],
                "fourier_effective_component_count": result["fourier_info"]["effective_component_count"],
                "fourier_sigma_x": result["fourier_info"]["sigma_x"],
                "fourier_sigma_y": result["fourier_info"]["sigma_y"],
                "fourier_theta": result["fourier_info"]["theta"],
            }
        )
    return row



def _grouped_results(rows):
    table = []
    background_types = sorted({row["background_type"] for row in rows})
    intensity_levels = sorted({row["intensity_level"] for row in rows})
    for method in METHODS:
        for background_type in background_types:
            for intensity_level in intensity_levels:
                subset = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["background_type"] == background_type
                    and row["intensity_level"] == intensity_level
                ]
                if not subset:
                    continue
                table.append(
                    {
                        "method": method,
                        "background_type": background_type,
                        "intensity_level": intensity_level,
                        "count": len(subset),
                        "bg_mae_mean": np.mean([row["bg_mae"] for row in subset]),
                        "bg_mae_std": np.std([row["bg_mae"] for row in subset]),
                        "star_f1_mean": np.mean([row["star_f1"] for row in subset]),
                        "flux_error_mean": reporting.finite_mean(
                            subset, "matched_star_mean_relative_flux_error"
                        ),
                        "flux_error_valid_count": len(
                            reporting.finite_values(
                                subset, "matched_star_mean_relative_flux_error"
                            )
                        ),
                    }
                )
    return table


def _failure_cases(rows):
    cases = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        selected = [
            ("largest_bg_mae", max(subset, key=lambda row: row["bg_mae"])),
            ("lowest_star_f1", min(subset, key=lambda row: row["star_f1"])),
        ]
        seen = set()
        for reason, row in selected:
            identity = (row["seed"], row["sample_id"])
            if identity in seen:
                continue
            seen.add(identity)
            seed_dir = (
                "deterministic"
                if row["seed"] == "deterministic"
                else f"seed_{row['seed']}"
            )
            cases.append(
                {
                    "method": method,
                    "reason": reason,
                    "seed": row["seed"],
                    "sample_id": row["sample_id"],
                    "background_type": row["background_type"],
                    "intensity_level": row["intensity_level"],
                    "bg_mae": row["bg_mae"],
                    "star_f1": row["star_f1"],
                    "matched_star_mean_relative_flux_error": row["matched_star_mean_relative_flux_error"],
                    "prediction_dir": f"{method}/{seed_dir}/{row['sample_id']}",
                }
            )
    return cases


def _main_result_table(rows, training_rows):
    table = []
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        train_times = [
            row["training_time_s"]
            for row in training_rows
            if row["method"] == method and "training_time_s" in row
        ]
        table.append(
            {
                "method": method,
                "bg_mae": np.mean([row["bg_mae"] for row in subset]),
                "bg_mae_std": np.std([row["bg_mae"] for row in subset]),
                "bg_psnr": np.mean([row["bg_psnr"] for row in subset]),
                "bg_psnr_std": np.std([row["bg_psnr"] for row in subset]),
                "bg_ssim": np.mean([row["bg_ssim"] for row in subset]),
                "bg_ssim_std": np.std([row["bg_ssim"] for row in subset]),
                "residual_psnr": np.mean([row["residual_psnr"] for row in subset]),
                "residual_psnr_std": np.std([row["residual_psnr"] for row in subset]),
                "residual_ssim": np.mean([row["residual_ssim"] for row in subset]),
                "residual_ssim_std": np.std([row["residual_ssim"] for row in subset]),
                "star_f1": np.mean([row["star_f1"] for row in subset]),
                "star_f1_std": np.std([row["star_f1"] for row in subset]),
                "matched_star_mean_relative_flux_error": reporting.finite_mean(
                    subset, "matched_star_mean_relative_flux_error"
                ),
                "flux_error_std": reporting.finite_std(
                    subset, "matched_star_mean_relative_flux_error"
                ),
                "flux_error_valid_count": len(
                    reporting.finite_values(
                        subset, "matched_star_mean_relative_flux_error"
                    )
                ),
                "background_below_zero_fraction": np.mean(
                    [row["background_below_zero_fraction"] for row in subset]
                ),
                "background_above_one_fraction": np.mean(
                    [row["background_above_one_fraction"] for row in subset]
                ),
                "train_time_s": np.mean(train_times) if train_times else "N/A",
                "time_per_image_s": np.mean([row["runtime_s"] for row in subset]),
                "peak_vram_mb": _mean_available(subset, "peak_vram_mb"),
                "parameter_count": np.mean([row["parameter_count"] for row in subset]),
            }
        )
    return table


def _low_high_consistency(output_root, samples, metric_rows):
    """Check whether predicted high-low background change matches injected truth.

    This is a diagnostic requested after the Stage-9 pilot exposed a native-sky
    offset.  It does not replace the main absolute background metrics.
    """
    sample_map = {sample["sample_id"]: sample for sample in samples}
    rows = []
    groups = {}
    for row in metric_rows:
        key = (row["method"], str(row["seed"]), row["base_id"], row["background_type"])
        groups.setdefault(key, {})[row["intensity_level"]] = row
    for (method, seed, base_id, background_type), levels in groups.items():
        if "low" not in levels or "high" not in levels:
            continue
        low_row, high_row = levels["low"], levels["high"]
        seed_dir = "deterministic" if seed == "deterministic" else f"seed_{seed}"
        low_path = output_root / method / seed_dir / low_row["sample_id"] / "background_pred.tif"
        high_path = output_root / method / seed_dir / high_row["sample_id"] / "background_pred.tif"
        pred_difference = utils.load_float_tiff(high_path) - utils.load_float_tiff(low_path)
        true_difference = (
            sample_map[high_row["sample_id"]]["background_true"]
            - sample_map[low_row["sample_id"]]["background_true"]
        )
        rows.append(
            {
                "method": method,
                "seed": seed,
                "base_id": base_id,
                "background_type": background_type,
                "low_sample_id": low_row["sample_id"],
                "high_sample_id": high_row["sample_id"],
                "high_minus_low_bg_mae": float(np.mean(np.abs(pred_difference - true_difference))),
                "predicted_difference_mean": float(np.mean(pred_difference)),
                "true_difference_mean": float(np.mean(true_difference)),
            }
        )
    return rows


def _mean_available(rows, key):
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float)) and np.isfinite(row[key])
    ]
    return np.mean(values) if values else "N/A"


def _pinn_parameters(result):
    return {
        key: result[key]
        for key in (
            "step",
            "total_loss",
            "data_loss",
            "physics_loss",
            "sparse_loss",
            "effective_component_count",
            "components",
            "alpha",
            "center_x",
            "center_y",
            "sigma_x",
            "sigma_y",
            "theta",
            "runtime_s",
            "peak_vram_mb",
            "parameter_count",
        )
    }



def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace the E2 output")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(force=args.force)
