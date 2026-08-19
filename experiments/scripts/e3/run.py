"""E3：PDE 与源点中心的四组最小消融。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics

VARIANTS = (
    "data_only",
    "center_fixed",
    "bright_init_fixed",
    "bright_init_learnable",
)


def build_variant_config(name: str, locked_pinn_config: dict) -> dict:
    """只改变消融允许变化的 physics_weight 与 center_mode。"""
    if name not in VARIANTS:
        raise ValueError(f"未知消融版本：{name}")
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
        raise AssertionError(f"消融修改了不允许的配置项：{sorted(changed)}")
    return config


def main():
    args = _parse_args()
    output_root = utils.prepare_output_root(args.output_root, force=args.force)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    locked_config = json.loads(Path(args.pinn_config).read_text(encoding="utf-8"))
    test_rows = utils.load_manifest(args.manifest, split="test")
    if args.max_test_samples > 0:
        test_rows = test_rows[: args.max_test_samples]
    if not test_rows:
        raise ValueError("测试集为空")

    utils.save_run_metadata(
        output_root,
        {
            "experiment": "E3",
            "manifest": utils.project_relative(args.manifest),
            "pinn_config": utils.project_relative(args.pinn_config),
            "variants": VARIANTS,
            "seeds": args.seeds,
            "test_samples": [row["sample_id"] for row in test_rows],
            "center_error_subset": "single_eccentric",
        },
    )

    samples = [utils.load_synthetic_sample(row) for row in test_rows]
    rows = []
    for variant in VARIANTS:
        variant_config = build_variant_config(variant, locked_config)
        utils.write_json(output_root / variant / "variant_config.json", variant_config)
        for seed in args.seeds:
            for sample in samples:
                result = pinn.train_background(
                    sample["observed"],
                    config=variant_config,
                    device=device,
                    seed=seed,
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
                    star_threshold=args.star_threshold,
                    matching_radius=args.matching_radius,
                )
                error = _center_error(variant, sample, result)
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "sample_id": sample["sample_id"],
                        "background_type": sample["background_type"],
                        "intensity_level": sample["metadata"]["intensity_level"],
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
    print(f"E3 完成：{output_root / 'ablation_table.csv'}")


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
                "residual_psnr": np.mean([row["residual_psnr"] for row in subset]),
                "residual_psnr_std": np.std([row["residual_psnr"] for row in subset]),
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


def _parse_args():
    parser = argparse.ArgumentParser(description="E3：PDE 与中心消融")
    parser.add_argument("--manifest", default=str(utils.SYNTHETIC_MANIFEST))
    parser.add_argument(
        "--pinn-config",
        default=str(utils.OUTPUT_ROOT / "e1_tuning/locked_pinn_config.json"),
    )
    parser.add_argument("--output-root", default=str(utils.OUTPUT_ROOT / "e3_ablation"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="清空指定的 experiments/outputs 子目录后重跑",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(utils.SEEDS))
    parser.add_argument("--star-threshold", type=float, default=0.03)
    parser.add_argument("--matching-radius", type=int, default=3)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
