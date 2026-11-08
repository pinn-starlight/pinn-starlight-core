"""E3 参数消融"""

import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics

FORCE_OVERWRITE_OUTPUT = True

VARIANTS = (
    "data_only",
    "center_fixed",
    "bright_init_fixed",
    "bright_init_learnable",
)

MANIFEST = utils.SYNTHETIC_MANIFEST
PINN_CONFIG = utils.OUTPUT_ROOT / "e1_tuning/locked_pinn_config.json"
OUTPUT_ROOT = utils.OUTPUT_ROOT / "e3_ablation"
SEEDS = tuple(utils.SEEDS)
# 暂时保留成这个数据
STAR_THRESHOLD = 0.1
MATCHING_RADIUS = 3

MAX_TEST_SAMPLES = 3


def build_variant_config(name: str, locked_pinn_config: dict) -> dict:
    """切换消融模式"""
    if name not in VARIANTS:
        raise ValueError(f"不存在该消融类型{name}")
    config = dict(locked_pinn_config)
    if name == "data_only":
        config["physics_weight"] = 0.0
        config["center_mode"] = "bright_init_learnable"
    elif name == "center_fixed":
        config["center_mode"] = "origin_fixed"
    elif name == "bright_init_fixed":
        config["center_mode"] = "bright_init_fixed"
    else:
        config["center_mode"] = "bright_init_learnable"
    changed = {
        key
        for key in set(locked_pinn_config) | set(config)
        if locked_pinn_config.get(key) != config.get(key)
    }
    if not changed <= {"physics_weight", "center_mode"}:
        raise AssertionError(f"消融失败：{sorted(changed)}")
    return config


def main():
    print("E3")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=FORCE_OVERWRITE_OUTPUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locked_config = json.loads(Path(PINN_CONFIG).read_text(encoding="utf-8"))
    test_rows = utils.load_manifest(MANIFEST, split="test")
    if MAX_TEST_SAMPLES > 0:
        test_rows = test_rows[:MAX_TEST_SAMPLES]
    if not test_rows:
        raise ValueError("测试集为空")
    utils.save_run_metadata(
        output_root,
        {
            "experiment": "E3",
            "manifest": utils.project_relative(MANIFEST),
            "pinn_config": utils.project_relative(PINN_CONFIG),
            "variants": VARIANTS,
            "seeds": SEEDS,
            "force_overwrite_output": FORCE_OVERWRITE_OUTPUT,
            "star_threshold": STAR_THRESHOLD,
            "matching_radius": MATCHING_RADIUS,
            "max_test_samples": MAX_TEST_SAMPLES,
            "test_samples": [row["sample_id"] for row in test_rows],
            "center_error_subset": "single_eccentric",
        },
    )

    samples = [
        _attach_reference_stars(utils.load_synthetic_sample(row))
        for row in test_rows
    ]
    rows = []
    for variant in VARIANTS:
        variant_config = build_variant_config(variant, locked_config)
        utils.write_json(output_root / variant / "variant_config.json", variant_config)
        for seed in SEEDS:
            for sample in samples:
                result = pinn.train_background(
                    sample["observed"],
                    config=variant_config,
                    device=device,
                    seed=seed,
                    show_progress=False
                )
                sample_output = output_root / variant / f"seed_{seed}" / sample["sample_id"]
                utils.save_prediction_arrays(
                    sample_output,
                    result["observed"],
                    result["background_pred"],
                    result["residual_pred"],
                )
                utils.save_rows_csv(sample_output / "history.csv", result["history"])
                utils.write_json(
                    sample_output / "final_parameters.json",
                    _final_parameters(result),
                )

                scores = metrics.evaluate_synthetic(
                    sample,
                    result,
                    star_threshold=STAR_THRESHOLD,
                    matching_radius=MATCHING_RADIUS,
                )
                error = _center_error(variant, sample, result)
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "sample_id": sample["sample_id"],
                        "background_type": sample["background_type"],
                        "runtime_s": result["runtime_s"],
                        "peak_vram_mb": result["peak_vram_mb"],
                        "center_error": error,
                        "center_x": result["center_x"],
                        "center_y": result["center_y"],
                        **scores,
                    }
                )

    summary = utils.summarize_rows(rows, "variant")
    table = _ablation_table(rows)
    utils.save_rows_csv(output_root / "metrics.csv", rows)
    utils.save_rows_csv(output_root / "ablation_table.csv", table)
    utils.write_json(output_root / "summary.json", summary)
    print(f"E3:生成完毕")


def _center_error(variant, sample, result):
    if variant == "data_only" or sample["background_type"] != "single_eccentric":
        return None
    centers = sample["metadata"]["background_params"]["source_centers"]
    if len(centers) != 1:
        return None
    return metrics.center_error(
        result["center_x"],
        result["center_y"],
        centers[0]["x"],
        centers[0]["y"],
    )


def _attach_reference_stars(sample):
    metadata = dict(sample.get("metadata", {}))
    metadata["star_reference"] = {
        "threshold": STAR_THRESHOLD,
        "matching_radius": MATCHING_RADIUS,
        "stars": metrics.extract_stars(
            sample["clean_true"], threshold=STAR_THRESHOLD
        ),
    }
    return {**sample, "metadata": metadata}


def _ablation_table(rows):
    table = []
    for variant in VARIANTS:
        subset = [row for row in rows if row["variant"] == variant]
        center_errors = [
            row["center_error"]
            for row in subset
            if isinstance(row.get("center_error"), (int, float))
        ]
        bg_mae_values = [row["bg_mae"] for row in subset]
        table.append(
            {
                "variant": variant,
                "bg_mae": np.mean(bg_mae_values),
                "bg_mae_std": np.std(bg_mae_values),
                "star_f1": np.mean([row["star_f1"] for row in subset]),
                "star_f1_std": np.std([row["star_f1"] for row in subset]),
                "flux_error": np.mean([row["flux_error"] for row in subset]),
                "flux_error_std": np.std([row["flux_error"] for row in subset]),
                "center_error": np.mean(center_errors) if center_errors else "N/A",
                "center_error_std": np.std(center_errors) if center_errors else "N/A",
                "stability": f"Bg MAE std={float(np.std(bg_mae_values)):.6f}",
            }
        )
    return table


def _final_parameters(result):
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
        )
    }



if __name__ == "__main__":
    main()
