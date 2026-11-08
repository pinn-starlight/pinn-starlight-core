"""Stage11-B frozen follow-up validation for the final improved PINN."""
from __future__ import annotations

import argparse

import torch

from experiments.common.baselines import sgc_pinn as final_pinn
from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import stage_b_metrics
from experiments.configs.formal_experiment import (
    FINAL_PINN_FOLLOWUP_CONFIG,
    FORMAL_EXECUTION_CONFIG,
)
from experiments.scripts.final_pinn.common import (
    forced_reference_metrics,
    paired_comparisons,
    write_csv,
)
from experiments.scripts.stage_b.formal_run import _load_lock


OUTPUT_NAME = "stage_b/final_pinn_validation"


def _load_sample(row):
    sample = utils.load_synthetic_sample(row)
    sample["base_clean"] = utils.load_float_tiff(row["base_clean"])
    sample["protected_true"] = utils.load_float_tiff(row["protected_true"])
    sample["scene_type"] = row.get("scene_type", row["background_type"])
    return sample


def main(*, force=False, smoke=False):
    lock = _load_lock()
    frozen = lock["stage_b_config"].get("final_pinn_followup")
    if frozen != FINAL_PINN_FOLLOWUP_CONFIG:
        raise ValueError("final PINN config differs from the Stage-B Global Lock")

    output_root = utils.OUTPUT_ROOT / (OUTPUT_NAME + ("_smoke" if smoke else ""))
    output_root = utils.prepare_output_root(output_root, force=force)

    rows = utils.load_manifest(STAGE_B_MANIFEST, split="test")
    if smoke:
        rows = rows[:1]
    if not rows:
        raise ValueError("Stage11-B test split is empty")
    samples = [_load_sample(row) for row in rows]

    device = torch.device(str(FORMAL_EXECUTION_CONFIG["benchmark_device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Stage11-B final PINN validation requires CUDA")

    utils.save_run_metadata(
        output_root,
        {
            "experiment": "Stage11-B final PINN follow-up validation",
            "status": "frozen_followup_confirmation",
            "manifest": utils.project_relative(STAGE_B_MANIFEST),
            "stage_b_lock": utils.project_relative(
                utils.OUTPUT_ROOT / "stage_b/configuration_lock/stage_b_global_configuration_lock.json"
            ),
            "config": FINAL_PINN_FOLLOWUP_CONFIG,
            "smoke": bool(smoke),
            "test_samples": [r["sample_id"] for r in rows],
        },
    )

    metric_rows = []
    for seed in FINAL_PINN_FOLLOWUP_CONFIG["test_seeds"]:
        seed = int(seed)
        for sample_index, sample in enumerate(samples, 1):
            print(
                f"\n[Stage11-B final PINN {sample_index}/{len(samples)}] "
                f"seed={seed} {sample['sample_id']}",
                flush=True,
            )
            result = final_pinn.run_paired(
                sample["observed"],
                config=FINAL_PINN_FOLLOWUP_CONFIG,
                device=device,
                seed=seed,
                show_progress=True,
            )
            pair_root = output_root / f"seed_{seed}" / sample["sample_id"]
            pair_root.mkdir(parents=True, exist_ok=True)

            shared = result["shared"]
            utils.save_rows_csv(pair_root / "stage1_history.csv", shared["stage1_history"])
            utils.write_json(pair_root / "forcing_diagnostics.json", shared["forcing_diagnostics"])
            torch.save(shared["stage1_model_state"], pair_root / "stage1_model_state.pt")
            torch.save(shared["forcing_state"], pair_root / "forcing_state.pt")

            for method in (final_pinn.CONTROL_NAME, final_pinn.METHOD_NAME):
                method_result = result[method]
                residual = utils.reconstruct_residual(
                    sample["observed"], method_result["background_pred"]
                )
                method_dir = pair_root / method
                utils.save_prediction_arrays(
                    method_dir,
                    sample["observed"],
                    method_result["background_pred"],
                    residual,
                )
                utils.save_rows_csv(method_dir / "training_history.csv", method_result["history"])
                torch.save(method_result["model_state"], method_dir / "final_model_state.pt")

                scores = stage_b_metrics.evaluate_stage_b(
                    sample,
                    {
                        "background_pred": method_result["background_pred"],
                        "residual_pred": residual,
                    },
                )
                scores.update(forced_reference_metrics(sample, residual))

                row = {
                    "method": method,
                    "seed": seed,
                    "sample_id": sample["sample_id"],
                    "base_id": sample["base_id"],
                    "scene_type": sample["scene_type"],
                    "background_type": sample["scene_type"],
                    "intensity_level": sample["intensity_level"],
                    "runtime_s": method_result["runtime_s_total"],
                    "stage1_runtime_s": shared["stage1_runtime_s"],
                    "forcing_runtime_s": (
                        shared["forcing_runtime_s"]
                        if method == final_pinn.METHOD_NAME
                        else 0.0
                    ),
                    "stage3_runtime_s": method_result["stage3_runtime_s"],
                    "peak_vram_mb": method_result["peak_vram_mb"],
                    "parameter_count": method_result["parameter_count"],
                    "gradient_conflict_fraction": method_result["gradient_conflict_fraction"],
                    "mean_lambda_use": method_result["mean_lambda_use"],
                    "mean_scaled_physics_to_data_ratio": method_result[
                        "mean_scaled_physics_to_data_ratio"
                    ],
                    "mean_target_scaled_physics_to_data_ratio": method_result[
                        "mean_target_scaled_physics_to_data_ratio"
                    ],
                    "mean_absolute_balance_error": method_result[
                        "mean_absolute_balance_error"
                    ],
                    **scores,
                }
                metric_rows.append(row)
                utils.write_json(method_dir / "final_metrics.json", row)

    summary = utils.summarize_rows(metric_rows, "method")
    comparisons = paired_comparisons(metric_rows)
    utils.save_rows_csv(output_root / "metrics.csv", metric_rows)
    utils.write_json(output_root / "summary.json", summary)
    write_csv(output_root / "paired_comparisons.csv", comparisons)

    print("\nStage11-B final PINN validation complete.")
    print(" metrics:", output_root / "metrics.csv")
    print(" paired :", output_root / "paired_comparisons.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(force=args.force, smoke=args.smoke)
