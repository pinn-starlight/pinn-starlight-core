"""Create the Stage10-B Global Lock after all B selectors finish."""

from __future__ import annotations

import argparse
from datetime import datetime

from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST, STAGE_B_ROOT
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import stage_b_provenance as provenance
from experiments.common.utils.stage_b_anchor import load_stage_a_anchor
from experiments.configs.stage_b_experiment import (
    STAGE_B_TEST_SEEDS,
    stage_b_config_snapshot,
    stage_b_selection_config,
)
from experiments.scripts.stage_b import (
    select_fourier_ls,
    select_gaussian,
    select_pinn,
)

OUTPUT_ROOT = utils.OUTPUT_ROOT / "stage_b/configuration_lock"
LOCK_PATH = OUTPUT_ROOT / "stage_b_global_configuration_lock.json"

FORMAL_SOURCE_FILES = (
    "experiments/configs/stage_b_experiment.py",
    "experiments/configs/split_protocols.py",
    "experiments/common/data/generate_stage_b.py",
    "experiments/common/data/stage_b_backgrounds.py",
    "experiments/common/utils/stage_b_metrics.py",
    "experiments/common/utils/stage_b_provenance.py",
    "experiments/common/utils/stage_b_anchor.py",
    "experiments/common/utils/metrics.py",
    "experiments/common/utils/experiment_utils.py",
    "experiments/common/baselines/fft_gaussian.py",
    "experiments/common/baselines/cosine_ls.py",
    "experiments/common/baselines/coordinate_pinn.py",
    "experiments/common/baselines/sgc_pinn.py",
    "src/pinn_starlight_core/physics/forcing.py",
    "src/pinn_starlight_core/nn/physics_model.py",
    "src/pinn_starlight_core/nn/pinn_layers.py",
    "src/pinn_starlight_core/nn/pinn_loss.py",
    "experiments/scripts/stage_b/formal_run.py",
    "experiments/scripts/final_pinn/common.py",
    "experiments/scripts/final_pinn/followup.py",
    "experiments/scripts/final_pinn/stage_b.py",
)


def main(*, force=False):
    utils.prepare_output_root(OUTPUT_ROOT, force=force)

    # Relevant A/B scientific compatibility only.  Do not run the whole A formal-config verifier.
    a_lock = load_stage_a_anchor()

    artifacts = {
        "fft_gaussian": provenance.verify_selection_artifact(
            select_gaussian.ARTIFACT,
            method="fft_gaussian",
            selection_config=stage_b_selection_config("fft_gaussian"),
            manifest=STAGE_B_MANIFEST,
            required_splits=("validation",),
            source_files=select_gaussian.SOURCE_FILES,
        ),
        "fourier_ls": provenance.verify_selection_artifact(
            select_fourier_ls.ARTIFACT,
            method="fourier_ls",
            selection_config=stage_b_selection_config("fourier_ls"),
            manifest=STAGE_B_MANIFEST,
            required_splits=("validation",),
            source_files=select_fourier_ls.SOURCE_FILES,
        ),
        "pinn": provenance.verify_selection_artifact(
            select_pinn.ARTIFACT,
            method="pinn",
            selection_config=stage_b_selection_config("pinn"),
            manifest=STAGE_B_MANIFEST,
            required_splits=("validation",),
            source_files=select_pinn.SOURCE_FILES,
        ),
    }

    train_rows = utils.load_manifest(STAGE_B_MANIFEST, split="train")
    val_rows = utils.load_manifest(STAGE_B_MANIFEST, split="validation")

    # Only IDs are read from the manifest; test image arrays are still unopened.
    test_rows = utils.load_manifest(STAGE_B_MANIFEST, split="test")

    methods = {
        "fft_gaussian": artifacts["fft_gaussian"]["selected"],
        "fourier_ls": artifacts["fourier_ls"]["selected"],
        "pinn": {
            **artifacts["pinn"]["selected"],
            "test_seeds": list(STAGE_B_TEST_SEEDS),
        },
    }

    config_snapshot = stage_b_config_snapshot()
    lock = {
        "schema_version": 3,
        "lock_state": "FROZEN",
        "formal_test_ready": True,
        "split_protocol": {
            "mode": utils.SPLIT_MODE,
            "cv_fold": utils.CV_FOLD,
            "output_namespace": utils.OUTPUT_NAMESPACE,
            "independent_group": "camera_reference_base_id",
        },
        "created_at": datetime.now().astimezone().isoformat(),
        "boundary": {
            "selection_splits": ["train", "validation"],
            "formal_test_split": "test",
            "test_arrays_loaded_during_lock": False,
            "selection_or_training_performed_during_lock": False,
            "rule": "No Stage-B global hyperparameter/checkpoint selection after this lock.",
        },
        "stage_b_config": config_snapshot,
        "stage_b_config_sha256": provenance.stable_json_sha256(config_snapshot),
        "data_provenance": {
            "manifest": utils.project_relative(STAGE_B_MANIFEST),
            "manifest_sha256": utils.file_sha256(STAGE_B_MANIFEST),
            "dataset_protocol": utils.project_relative(STAGE_B_ROOT / "dataset_protocol.json"),
            "dataset_protocol_sha256": utils.file_sha256(STAGE_B_ROOT / "dataset_protocol.json"),
            "train_sample_ids": [r["sample_id"] for r in train_rows],
            "validation_sample_ids": [r["sample_id"] for r in val_rows],
            "test_sample_ids": [r["sample_id"] for r in test_rows],
        },
        "selection_artifacts": {
            method: {
                "path": utils.project_relative(path),
                "sha256": utils.file_sha256(path),
            }
            for method, path in {
                "fft_gaussian": select_gaussian.ARTIFACT,
                "fourier_ls": select_fourier_ls.ARTIFACT,
                "pinn": select_pinn.ARTIFACT,
            }.items()
        },
        "methods": methods,
        "source_provenance": provenance.source_snapshot(FORMAL_SOURCE_FILES),
        "stage10_a_reference": {
            "global_lock": utils.project_relative(utils.GLOBAL_CONFIGURATION_LOCK),
            "global_lock_sha256": utils.file_sha256(utils.GLOBAL_CONFIGURATION_LOCK),
            "lock_state": a_lock["lock_state"],
            "formal_test_ready": bool(a_lock["formal_test_ready"]),
            "compatibility_rule": "method-specific selection/training configs and core method source matched Stage10-A",
        },
    }
    utils.write_json(LOCK_PATH, lock)
    print("Stage-B Global Lock written:", LOCK_PATH)
    print("formal_test_ready=True")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
