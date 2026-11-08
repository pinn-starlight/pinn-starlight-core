"""Shared runner for the frozen final-PINN follow-up validation.

``stage_a`` and ``stage_b`` differ only in where the Global Lock comes from,
which manifest and metric evaluator they use, and a handful of metadata/row
fields.  Everything else — device check, artifact layout, the seed x sample
loop, summary aggregation — is identical and lives here.

The numerical work stays in :func:`sgc_pinn.run_paired` and the per-stage
``evaluate_*`` functions; this module only wires it together.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from experiments.common.baselines import sgc_pinn as final_pinn
from experiments.common.utils import experiment_utils as utils
from experiments.configs.formal_experiment import (
    FINAL_PINN_FOLLOWUP_CONFIG,
    FORMAL_EXECUTION_CONFIG,
)
from experiments.scripts.final_pinn.common import (
    forced_reference_metrics,
    paired_comparisons,
    write_csv,
)


@dataclass(frozen=True)
class FollowupSpec:
    """Stage-specific variation points of the final-PINN follow-up run."""

    name: str
    """Human-readable stage label, e.g. ``"Stage11-A"``."""

    output_name: str
    """Output directory name relative to ``utils.OUTPUT_ROOT``."""

    load_lock: Callable[[], dict]
    """Return the verified Global Lock of this stage."""

    config_key: tuple[str, ...]
    """Key path locating the frozen final-PINN config inside the lock."""

    manifest: Path
    """Test-split manifest of this stage."""

    load_sample: Callable[[dict], dict]
    """Turn one manifest row into the sample dict used by the runner."""

    evaluate: Callable[[dict, dict], dict]
    """Stage-specific metric evaluator."""

    build_metadata: Callable[[list, bool], dict]
    """Build the ``run_metadata.json`` payload from rows and the smoke flag."""

    drop_row_fields: tuple[str, ...] = ()
    """Row keys this stage does not report (e.g. stage-B omits lambda stats)."""

    extra_row_fields: Callable[[dict], dict] | None = None
    """Additional or overriding row fields, as a function of the sample."""


def _frozen_config(lock: dict, key_path: tuple[str, ...]):
    """Read the frozen config, tolerating a missing leaf key."""
    value = lock
    for key in key_path[:-1]:
        value = value[key]
    return value.get(key_path[-1])


def run_followup(spec: FollowupSpec, *, force: bool = False, smoke: bool = False) -> None:
    """Run the frozen follow-up validation for one stage and write its outputs."""
    lock = spec.load_lock()
    frozen = _frozen_config(lock, spec.config_key)
    if frozen != FINAL_PINN_FOLLOWUP_CONFIG:
        raise ValueError(f"final PINN config differs from the {spec.name} Global Lock")

    output_root = utils.OUTPUT_ROOT / (spec.output_name + ("_smoke" if smoke else ""))
    output_root = utils.prepare_output_root(output_root, force=force)

    rows = utils.load_manifest(spec.manifest, split="test")
    if smoke:
        rows = rows[:1]
    if not rows:
        raise ValueError(f"{spec.name} test split is empty")
    samples = [spec.load_sample(row) for row in rows]

    device = torch.device(str(FORMAL_EXECUTION_CONFIG["benchmark_device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"{spec.name} final PINN validation requires CUDA")

    utils.save_run_metadata(output_root, spec.build_metadata(rows, smoke))

    metric_rows = []
    for seed in FINAL_PINN_FOLLOWUP_CONFIG["test_seeds"]:
        seed = int(seed)
        for sample_index, sample in enumerate(samples, 1):
            print(
                f"\n[{spec.name} final PINN {sample_index}/{len(samples)}] "
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

                scores = spec.evaluate(
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
                    "background_type": sample["background_type"],
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
                    "median_lambda_use": method_result["median_lambda_use"],
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
                for key in spec.drop_row_fields:
                    row.pop(key, None)
                if spec.extra_row_fields is not None:
                    row.update(spec.extra_row_fields(sample))

                metric_rows.append(row)
                utils.write_json(method_dir / "final_metrics.json", row)

    summary = utils.summarize_rows(metric_rows, "method")
    comparisons = paired_comparisons(metric_rows)
    utils.save_rows_csv(output_root / "metrics.csv", metric_rows)
    utils.write_json(output_root / "summary.json", summary)
    write_csv(output_root / "paired_comparisons.csv", comparisons)

    print(f"\n{spec.name} final PINN validation complete.")
    print(" metrics:", output_root / "metrics.csv")
    print(" paired :", output_root / "paired_comparisons.csv")


def cli_main(spec: FollowupSpec) -> None:
    """Standard ``--force`` / ``--smoke`` entry point used by both stage scripts."""
    import argparse

    parser = argparse.ArgumentParser(description=f"{spec.name} final PINN follow-up.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_followup(spec, force=args.force, smoke=args.smoke)
