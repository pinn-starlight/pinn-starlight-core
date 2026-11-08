"""E3: Stage-10 PINN ablation on the held-out synthetic test split."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.configs.formal_experiment import FORMAL_EXECUTION_CONFIG, PINN_CONFIG

VARIANTS = tuple(PINN_CONFIG["ablation"]["variants"])
MANIFEST = utils.SYNTHETIC_MANIFEST
GLOBAL_LOCK = utils.GLOBAL_CONFIGURATION_LOCK
OUTPUT_ROOT = utils.OUTPUT_ROOT / "e3_ablation"
MAX_TEST_SAMPLES = 0


def build_variant_config(name: str, locked_pinn_config: dict) -> dict:
    if name not in VARIANTS:
        raise ValueError(f"unknown ablation variant: {name}")
    config = dict(locked_pinn_config)
    if name == "data_only":
        config["physics_weight"] = 0.0
        config["sparse_weight"] = 0.0
    elif name == "single_component":
        config["forcing_max_components"] = 1
    elif name == "multi_fixed_centers":
        config["center_refinement_enabled"] = False
    elif name == "multi_bounded_centers":
        pass
    changed = {
        key
        for key in set(locked_pinn_config) | set(config)
        if locked_pinn_config.get(key) != config.get(key)
    }
    allowed = set(PINN_CONFIG["ablation"]["allowed_changed_keys"])
    if not changed <= allowed:
        raise AssertionError(f"ablation changed non-approved keys: {sorted(changed)}")
    return dict(pinn.normalized_config(config))


def _formal_device():
    """Resolve the frozen single-GPU Stage10-A benchmark device."""
    device = torch.device(str(FORMAL_EXECUTION_CONFIG["benchmark_device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Formal Stage10-A requires CUDA for the frozen benchmark policy; CUDA is not available."
        )
    return device


def main(*, force: bool = False):
    print("E3 Stage-10 PINN ablation")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    device = _formal_device()
    global_lock = utils.load_global_configuration_lock(
        GLOBAL_LOCK, verify=True, require_formal_ready=True
    )
    locked_config = global_lock["methods"]["pinn"]["config"]
    seeds = tuple(int(seed) for seed in global_lock["methods"]["pinn"]["test_seeds"])
    test_rows = utils.load_manifest(MANIFEST, split="test")
    if MAX_TEST_SAMPLES > 0:
        test_rows = test_rows[:MAX_TEST_SAMPLES]
    if not test_rows:
        raise ValueError("test split is empty")
    utils.save_run_metadata(
        output_root,
        {
            "experiment": "E3 Stage-10 PINN ablation",
            "manifest": utils.project_relative(MANIFEST),
            "global_configuration_lock": utils.project_relative(GLOBAL_LOCK),
            "global_configuration_lock_sha256": utils.file_sha256(GLOBAL_LOCK),
            "variants": VARIANTS,
            "seeds": seeds,
            "compute_device": str(device),
            "formal_execution_policy": dict(FORMAL_EXECUTION_CONFIG),
            "force_overwrite_output": force,
            "max_test_samples": MAX_TEST_SAMPLES,
            "test_samples": [row["sample_id"] for row in test_rows],
            "center_error_subset": "single_eccentric",
            "center_error_rule": "nearest active effective component to injected centre; evaluation only",
            "evaluation_schema_version": metrics.EVALUATION_SCHEMA_VERSION,
            "star_detection_rule": metrics.DETECTION_IMAGE_RULE,
            "flux_metric": metrics.MATCHED_STAR_FLUX_METRIC,
        },
    )

    samples = [utils.load_synthetic_sample(row) for row in test_rows]
    rows = []
    for variant in VARIANTS:
        variant_config = build_variant_config(variant, locked_config)
        utils.write_json(output_root / variant / "variant_config.json", variant_config)
        for seed in seeds:
            for sample in samples:
                result = pinn.train_background(
                    sample["observed"],
                    config=variant_config,
                    device=device,
                    seed=seed,
                    show_progress=True,
                    laplacian_backend=FORMAL_EXECUTION_CONFIG["pinn_laplacian_backend"],
                )
                prediction = {
                    "background_pred": result["background_pred"],
                    "residual_pred": utils.reconstruct_residual(
                        sample["observed"], result["background_pred"]
                    ),
                }
                sample_output = output_root / variant / f"seed_{seed}" / sample["sample_id"]
                utils.save_prediction_arrays(
                    sample_output,
                    sample["observed"],
                    prediction["background_pred"],
                    prediction["residual_pred"],
                )
                utils.save_rows_csv(sample_output / "history.csv", result["history"])
                utils.write_json(sample_output / "final_parameters.json", _final_parameters(result))

                scores = metrics.evaluate_synthetic(sample, prediction)
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "sample_id": sample["sample_id"],
                        "background_type": sample["background_type"],
                        "intensity_level": sample["intensity_level"],
                        "runtime_s": result["runtime_s"],
                        "peak_vram_mb": result["peak_vram_mb"],
                        "center_error": _center_error(variant, sample, result),
                        "center_x": result["center_x"],
                        "center_y": result["center_y"],
                        "effective_component_count": result["effective_component_count"],
                        **scores,
                    }
                )

    summary = utils.summarize_rows(rows, "variant")
    table = _ablation_table(rows)
    utils.save_rows_csv(output_root / "metrics.csv", rows)
    utils.save_rows_csv(output_root / "ablation_table.csv", table)
    utils.write_json(output_root / "summary.json", summary)
    print(f"E3 complete: {output_root / 'ablation_table.csv'}")


def _center_error(variant, sample, result):
    if variant == "data_only" or sample["background_type"] != "single_eccentric":
        return None
    truth = sample["metadata"].get(
        "generation_truth", sample["metadata"].get("background_params", {})
    )
    centers = truth.get("source_centers", [])
    if len(centers) != 1:
        return None
    candidates = [item for item in result["components"] if item.get("active")]
    if not candidates:
        candidates = list(result["components"])
    return min(
        metrics.center_error(
            item["center_x"],
            item["center_y"],
            centers[0]["x"],
            centers[0]["y"],
        )
        for item in candidates
    )


def _ablation_table(rows):
    table = []
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        center_errors = [
            float(row["center_error"])
            for row in subset
            if isinstance(row.get("center_error"), (int, float, np.number))
        ]
        bg_mae_values = [row["bg_mae"] for row in subset]
        table.append(
            {
                "variant": variant,
                "bg_mae": np.mean(bg_mae_values),
                "bg_mae_std": np.std(bg_mae_values),
                "star_f1": np.mean([row["star_f1"] for row in subset]),
                "star_f1_std": np.std([row["star_f1"] for row in subset]),
                "matched_star_mean_relative_flux_error": _finite_metric_mean(
                    subset, "matched_star_mean_relative_flux_error"
                ),
                "flux_error_std": _finite_metric_std(
                    subset, "matched_star_mean_relative_flux_error"
                ),
                "flux_error_valid_count": len(
                    _finite_metric_values(subset, "matched_star_mean_relative_flux_error")
                ),
                "center_error": np.mean(center_errors) if center_errors else "N/A",
                "center_error_std": np.std(center_errors) if center_errors else "N/A",
                "mean_effective_component_count": np.mean(
                    [row["effective_component_count"] for row in subset]
                ),
                "stability": f"Bg MAE std={float(np.std(bg_mae_values)):.6f}",
            }
        )
    return table


def _finite_metric_values(rows, key):
    return [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float, np.number)) and np.isfinite(row[key])
    ]


def _finite_metric_mean(rows, key):
    values = _finite_metric_values(rows, key)
    return float(np.mean(values)) if values else float("nan")


def _finite_metric_std(rows, key):
    values = _finite_metric_values(rows, key)
    return float(np.std(values)) if values else float("nan")


def _final_parameters(result):
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
        )
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace the E3 output")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(force=args.force)
