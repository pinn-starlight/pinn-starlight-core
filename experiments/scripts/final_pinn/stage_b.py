"""Stage11-B frozen follow-up validation for the final improved PINN."""

from __future__ import annotations

from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import stage_b_metrics
from experiments.configs.formal_experiment import FINAL_PINN_FOLLOWUP_CONFIG
from experiments.scripts.final_pinn.followup import FollowupSpec, cli_main, run_followup
from experiments.scripts.stage_b.formal_run import load_stage_b_lock


def _load_sample(row):
    sample = utils.load_synthetic_sample(row)
    sample["base_clean"] = utils.load_float_tiff(row["base_clean"])
    sample["protected_true"] = utils.load_float_tiff(row["protected_true"])
    sample["scene_type"] = row.get("scene_type", row["background_type"])
    return sample


def _build_metadata(rows, smoke):
    return {
        "experiment": "Stage11-B final PINN follow-up validation",
        "status": "frozen_followup_confirmation",
        "manifest": utils.project_relative(STAGE_B_MANIFEST),
        "stage_b_lock": utils.project_relative(
            utils.OUTPUT_ROOT / "stage_b/configuration_lock/stage_b_global_configuration_lock.json"
        ),
        "config": FINAL_PINN_FOLLOWUP_CONFIG,
        "smoke": bool(smoke),
        "test_samples": [r["sample_id"] for r in rows],
    }


def _scene_fields(sample):
    """Stage-B reports the scene type and keys ``background_type`` off it."""
    return {
        "scene_type": sample["scene_type"],
        "background_type": sample["scene_type"],
    }


SPEC = FollowupSpec(
    name="Stage11-B",
    output_name="stage_b/final_pinn_validation",
    load_lock=load_stage_b_lock,
    config_key=("stage_b_config", "final_pinn_followup"),
    manifest=STAGE_B_MANIFEST,
    load_sample=_load_sample,
    evaluate=stage_b_metrics.evaluate_stage_b,
    build_metadata=_build_metadata,
    drop_row_fields=("median_lambda_use",),
    extra_row_fields=_scene_fields,
)


def main(*, force=False, smoke=False):
    run_followup(SPEC, force=force, smoke=smoke)


if __name__ == "__main__":
    cli_main(SPEC)
