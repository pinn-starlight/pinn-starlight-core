"""E2 synthetic benchmark."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.baselines import fft_gaussian as fft
from experiments.common.baselines import unet_small as unet
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics

METHODS = ("fft_gaussian", "unet_small", "pinn")


def run_method(method: str, sample: dict, locked_config: dict, **kwargs) -> dict:
    """Run one method and return its prediction and measurements."""
    observed = sample["observed"]
    device = kwargs.get("device", torch.device("cpu"))
    utils.reset_peak_vram(device)
    started = time.perf_counter()
    if method == "fft_gaussian":
        background_pred = fft.estimate_background(
            observed,
            float(locked_config["fft_sigma"]),
        )
        extra = {"parameter_count": 0, "peak_vram_mb": None}
    elif method == "unet_small":
        background_pred = unet.predict_with_model(
            kwargs["unet_model"],
            observed,
            device=kwargs["device"],
            tile_size=int(locked_config["unet_tile_size"]),
            overlap=int(locked_config["unet_overlap"]),
        )
        extra = {
            "parameter_count": kwargs["unet_parameter_count"],
            "peak_vram_mb": utils.peak_vram_mb(device),
        }
    elif method == "pinn":
        pinn_result = pinn.train_background(
            observed,
            config=locked_config["pinn_config"],
            device=kwargs["device"],
            seed=kwargs["seed"],
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
        "residual_pred": observed - background_pred,
        "runtime_s": float(runtime_s),
        "peak_vram_mb": extra["peak_vram_mb"],
        **extra,
    }


def main():
    args = _parse_args()
    output_root = utils.prepare_output_root(args.output_root, force=args.force)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = tuple(args.seeds)

    train_rows = utils.load_manifest(args.manifest, split="train")
    validation_rows = utils.load_manifest(args.manifest, split="validation")
    test_rows = utils.load_manifest(args.manifest, split="test")
    if args.max_test_samples > 0:
        test_rows = test_rows[: args.max_test_samples]
    if not train_rows or not validation_rows or not test_rows:
        raise ValueError("train, validation, and test splits must all be non-empty")

    pinn_config = json.loads(Path(args.pinn_config).read_text(encoding="utf-8"))
    validation_samples = [utils.load_synthetic_sample(row) for row in validation_rows]
    fft_sigma = fft.select_sigma(validation_samples)
    locked_config = {
        "fft_sigma": fft_sigma,
        "pinn_config": pinn_config,
        "unet_tile_size": args.patch_size,
        "unet_overlap": min(32, args.patch_size // 4),
        "star_threshold": args.star_threshold,
        "matching_radius": args.matching_radius,
    }
    utils.save_run_metadata(
        output_root,
        {
            "experiment": "E2",
            "manifest": utils.project_relative(args.manifest),
            "pinn_config": utils.project_relative(args.pinn_config),
            "seeds": seeds,
            "locked_config": locked_config,
            "test_samples": [row["sample_id"] for row in test_rows],
            "unet_training": {
                "epochs": args.epochs,
                "steps_per_epoch": args.steps_per_epoch,
                "batch_size": args.unet_batch_size,
                "patch_size": args.patch_size,
                "base_channels": args.base_channels,
                "patience": args.patience,
            },
        },
    )

    samples = [utils.load_synthetic_sample(row) for row in test_rows]
    metric_rows = []
    training_rows = []

    for sample in samples:
        result = run_method("fft_gaussian", sample, locked_config)
        _save_result(
            output_root / "fft_gaussian/deterministic" / sample["sample_id"],
            sample,
            result,
        )
        metric_rows.append(
            _metric_row(
                "fft_gaussian",
                "deterministic",
                sample,
                result,
                args,
            )
        )

    train_pairs = _pairs(train_rows)
    validation_pairs = _pairs(validation_rows)
    unet_checkpoints = []
    for seed in seeds:
        checkpoint_dir = output_root / "unet_small" / f"seed_{seed}" / "checkpoint"
        utils.reset_peak_vram(device)
        started = time.perf_counter()
        checkpoint = unet.train(
            train_pairs,
            validation_pairs,
            checkpoint_dir,
            device=device,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.unet_batch_size,
            patch_size=args.patch_size,
            base_channels=args.base_channels,
            patience=args.patience,
            seed=seed,
        )
        training_time = time.perf_counter() - started
        model, checkpoint_data = unet.load_checkpoint_model(checkpoint, device)
        parameter_count = utils.parameter_count(model)
        training_rows.append(
            {
                "method": "unet_small",
                "seed": seed,
                "training_time_s": training_time,
                "peak_vram_mb": utils.peak_vram_mb(device),
                "validation_loss": checkpoint_data["validation_loss"],
                "best_epoch": checkpoint_data["epoch"],
                "parameter_count": parameter_count,
                "checkpoint": utils.project_relative(checkpoint),
            }
        )
        unet_checkpoints.append(
            (float(checkpoint_data["validation_loss"]), seed, checkpoint)
        )

        for sample in samples:
            result = run_method(
                "unet_small",
                sample,
                locked_config,
                unet_model=model,
                unet_parameter_count=parameter_count,
                device=device,
            )
            _save_result(
                output_root / "unet_small" / f"seed_{seed}" / sample["sample_id"],
                sample,
                result,
            )
            metric_rows.append(_metric_row("unet_small", seed, sample, result, args))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for seed in seeds:
        for sample in samples:
            result = run_method(
                "pinn",
                sample,
                locked_config,
                device=device,
                seed=seed,
            )
            sample_output = output_root / "pinn" / f"seed_{seed}" / sample["sample_id"]
            _save_result(sample_output, sample, result)
            pinn_result = result["pinn_result"]
            utils.save_rows_csv(sample_output / "history.csv", pinn_result["history"])
            utils.write_json(sample_output / "final_parameters.json", _pinn_parameters(pinn_result))
            metric_rows.append(_metric_row("pinn", seed, sample, result, args))

    summary = utils.summarize_rows(metric_rows, "method")
    grouped_table = _grouped_results(metric_rows)
    main_table = _main_result_table(metric_rows, training_rows)
    failure_cases = _failure_cases(metric_rows)
    best_validation, best_seed, best_checkpoint = min(unet_checkpoints)
    e2_locked = {
        "fft_sigma": fft_sigma,
        "unet_checkpoint": utils.project_relative(best_checkpoint),
        "unet_seed": best_seed,
        "unet_validation_loss": best_validation,
        "unet_tile_size": args.patch_size,
        "unet_overlap": min(32, args.patch_size // 4),
        "star_threshold": args.star_threshold,
        "matching_radius": args.matching_radius,
    }

    utils.save_rows_csv(output_root / "metrics.csv", metric_rows)
    utils.save_rows_csv(output_root / "training.csv", training_rows)
    utils.save_rows_csv(output_root / "grouped_results.csv", grouped_table)
    utils.save_rows_csv(output_root / "main_result_table.csv", main_table)
    utils.save_rows_csv(output_root / "failure_cases.csv", failure_cases)
    utils.write_json(output_root / "summary.json", summary)
    utils.write_json(output_root / "locked_e2_config.json", e2_locked)
    print(f"E2 完成：{output_root / 'main_result_table.csv'}")


def _save_result(output_dir, sample, result):
    utils.save_prediction_arrays(
        output_dir,
        sample["observed"],
        result["background_pred"],
        result["residual_pred"],
    )


def _metric_row(method, seed, sample, result, args):
    scores = metrics.evaluate_synthetic(
        sample,
        result,
        star_threshold=args.star_threshold,
        matching_radius=args.matching_radius,
    )
    return {
        "method": method,
        "seed": seed,
        "sample_id": sample["sample_id"],
        "background_type": sample["background_type"],
        "intensity_level": sample["metadata"]["intensity_level"],
        "runtime_s": result["runtime_s"],
        "peak_vram_mb": result["peak_vram_mb"],
        "parameter_count": result["parameter_count"],
        **scores,
    }


def _pairs(rows):
    return [
        (utils.to_absolute_path(row["observed"]), utils.to_absolute_path(row["background_true"]))
        for row in rows
    ]


def _grouped_results(rows):
    table = []
    for method in METHODS:
        for background_type in sorted({row["background_type"] for row in rows}):
            for intensity_level in sorted({row["intensity_level"] for row in rows}):
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
                        "residual_psnr_mean": np.mean(
                            [row["residual_psnr"] for row in subset]
                        ),
                        "star_f1_mean": np.mean([row["star_f1"] for row in subset]),
                        "flux_error_mean": np.mean(
                            [row["flux_error"] for row in subset]
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
                    "flux_error": row["flux_error"],
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
            if row["method"] == method
        ]
        table.append(
            {
                "method": method,
                "bg_mae": np.mean([row["bg_mae"] for row in subset]),
                "bg_mae_std": np.std([row["bg_mae"] for row in subset]),
                "bg_ssim": np.mean([row["bg_ssim"] for row in subset]),
                "bg_ssim_std": np.std([row["bg_ssim"] for row in subset]),
                "residual_psnr": np.mean([row["residual_psnr"] for row in subset]),
                "residual_psnr_std": np.std([row["residual_psnr"] for row in subset]),
                "residual_ssim": np.mean([row["residual_ssim"] for row in subset]),
                "residual_ssim_std": np.std([row["residual_ssim"] for row in subset]),
                "star_f1": np.mean([row["star_f1"] for row in subset]),
                "star_f1_std": np.std([row["star_f1"] for row in subset]),
                "flux_error": np.mean([row["flux_error"] for row in subset]),
                "flux_error_std": np.std([row["flux_error"] for row in subset]),
                "train_time_s": np.mean(train_times) if train_times else "N/A",
                "time_per_image_s": np.mean([row["runtime_s"] for row in subset]),
                "peak_vram_mb": _mean_available(subset, "peak_vram_mb"),
                "parameter_count": np.mean([row["parameter_count"] for row in subset]),
            }
        )
    return table


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
    parser = argparse.ArgumentParser(description="E2 synthetic benchmark")
    parser.add_argument("--manifest", default=str(utils.SYNTHETIC_MANIFEST))
    parser.add_argument(
        "--pinn-config",
        default=str(utils.OUTPUT_ROOT / "e1_tuning/locked_pinn_config.json"),
    )
    parser.add_argument("--output-root", default=str(utils.OUTPUT_ROOT / "e2_synthetic"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear the selected experiments/outputs subdirectory before running",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(utils.SEEDS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--unet-batch-size", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--star-threshold", type=float, default=0.03)
    parser.add_argument("--matching-radius", type=int, default=3)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
