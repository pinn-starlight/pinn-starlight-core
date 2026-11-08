"""Stage11-A frozen follow-up validation: matched data-only vs final improved PINN.

The current split context may be standard or one Stage-A CV5 fold.  This runner
requires the corresponding Stage-A Global Configuration Lock to exist first.
No truth array is used by either method; truth is opened only for evaluation.
"""
from __future__ import annotations

from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.configs.formal_experiment import FINAL_PINN_FOLLOWUP_CONFIG
from experiments.scripts.final_pinn.followup import FollowupSpec, cli_main, run_followup


def _load_lock():
    return utils.load_global_configuration_lock(
        utils.GLOBAL_CONFIGURATION_LOCK,
        verify=True,
        require_formal_ready=True,
    )


def _build_metadata(rows, smoke):
    return {
        "experiment": "Stage11-A final PINN follow-up validation",
        "status": "frozen_followup_confirmation",
        "split_mode": utils.SPLIT_MODE,
        "cv_fold": utils.CV_FOLD,
        "manifest": utils.project_relative(utils.SYNTHETIC_MANIFEST),
        "global_lock": utils.project_relative(utils.GLOBAL_CONFIGURATION_LOCK),
        "global_lock_sha256": utils.file_sha256(utils.GLOBAL_CONFIGURATION_LOCK),
        "config": FINAL_PINN_FOLLOWUP_CONFIG,
        "smoke": bool(smoke),
        "test_samples": [r["sample_id"] for r in rows],
    }


SPEC = FollowupSpec(
    name="Stage11-A",
    output_name="final_pinn_validation",
    load_lock=_load_lock,
    config_key=("formal_config", "snapshot", "final_pinn_followup"),
    manifest=utils.SYNTHETIC_MANIFEST,
    load_sample=utils.load_synthetic_sample,
    evaluate=metrics.evaluate_synthetic,
    build_metadata=_build_metadata,
)


def main(*, force=False, smoke=False):
    run_followup(SPEC, force=force, smoke=smoke)


if __name__ == "__main__":
    cli_main(SPEC)
