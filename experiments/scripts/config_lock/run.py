"""Pure Stage-10 Global Configuration Lock gate.

This script does not train, tune, or load synthetic image arrays. It verifies
three pre-test selection artifacts, verifies their method-local dependencies,
snapshots formal source/config state, and then opens the held-out test gate.
"""

from __future__ import annotations

import argparse

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.baselines import cosine_ls as fourier_ls
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import selection_artifacts as artifacts
from experiments.configs.formal_experiment import (
    FORMAL_CONFIG_SCHEMA_VERSION,
    PINN_CONFIG,
    formal_config_snapshot,
    selection_dependency_config,
)

MANIFEST = utils.SYNTHETIC_MANIFEST
OUTPUT_ROOT = utils.OUTPUT_ROOT / "configuration_lock"
LOCK_PATH = utils.GLOBAL_CONFIGURATION_LOCK

SELECTORS = {
    "pinn": {
        "path": utils.PINN_SELECTION_ARTIFACT,
        "splits": ("validation",),
        "command": "python -m experiments.scripts.selection.pinn",
    },
    "fft_gaussian": {
        "path": utils.GAUSSIAN_SELECTION_ARTIFACT,
        "splits": ("validation",),
        "command": "python -m experiments.scripts.selection.gaussian",
    },
    "fourier_ls": {
        "path": utils.FOURIER_LS_SELECTION_ARTIFACT,
        "splits": ("validation",),
        "command": "python -m experiments.scripts.selection.fourier_ls",
    },
}


def main(*, force: bool = False):
    print("Global Configuration Lock (pure freeze gate)")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Formal synthetic manifest missing: {MANIFEST}")

    selected_artifacts = {}
    for method, spec in SELECTORS.items():
        if not spec["path"].is_file():
            raise FileNotFoundError(
                f"Missing {method} selection artifact: {spec['path']}. "
                f"Run `{spec['command']}` first."
            )
        selected_artifacts[method] = artifacts.verify_selection_artifact(
            spec["path"],
            method=method,
            formal_config=selection_dependency_config(method),
            required_splits=spec["splits"],
            verify_sources=True,
        )

    train_rows = utils.load_manifest(MANIFEST, split="train")
    validation_rows = utils.load_manifest(MANIFEST, split="validation")
    if not train_rows or not validation_rows:
        raise ValueError("train and validation splits must be non-empty")

    gaussian = selected_artifacts["fft_gaussian"]["selected"]
    fourier = selected_artifacts["fourier_ls"]["selected"]
    pinn_selected = selected_artifacts["pinn"]["selected"]

    fourier_ls.validate_frozen_config(fourier)
    pinn_config = dict(pinn.normalized_config(pinn_selected["config"]))

    config_snapshot = formal_config_snapshot()
    selection_snapshot = {
        method: {
            "path": utils.project_relative(spec["path"]),
            "sha256": utils.file_sha256(spec["path"]),
            "formal_config_sha256": selected_artifacts[method]["formal_config_sha256"],
            "source_bundle_sha256": selected_artifacts[method]["source_provenance"][
                "bundle_sha256"
            ],
        }
        for method, spec in SELECTORS.items()
    }
    lock = {
        "schema_version": 6,
        "lock_state": "FROZEN",
        "formal_test_ready": True,
        "split_protocol": {
            "mode": utils.SPLIT_MODE,
            "cv_fold": utils.CV_FOLD,
            "output_namespace": utils.OUTPUT_NAMESPACE,
            "independent_group": "camera_reference_base_id",
        },
        "boundary": {
            "selection_splits": ["train", "validation"],
            "formal_test_split": "test",
            "test_arrays_loaded_during_lock": False,
            "selection_or_training_performed_during_lock": False,
            "rule": "No global hyperparameter/checkpoint selection after this lock.",
        },
        "formal_config": {
            "schema_version": FORMAL_CONFIG_SCHEMA_VERSION,
            "sha256": artifacts.stable_json_sha256(config_snapshot),
            "snapshot": config_snapshot,
        },
        "data_provenance": {
            "manifest": utils.project_relative(MANIFEST),
            "manifest_sha256": utils.file_sha256(MANIFEST),
            "dataset_protocol": utils.project_relative(
                utils.SYNTHETIC_ROOT / "dataset_protocol.json"
            ),
            "dataset_protocol_sha256": utils.file_sha256(
                utils.SYNTHETIC_ROOT / "dataset_protocol.json"
            ),
            "train_sample_ids": [row["sample_id"] for row in train_rows],
            "validation_sample_ids": [row["sample_id"] for row in validation_rows],
            "test_sample_ids": [
                row["sample_id"] for row in utils.load_manifest(MANIFEST, split="test")
            ],
        },
        "selection_artifacts": selection_snapshot,
        "source_provenance": artifacts.source_provenance(artifacts.FORMAL_SOURCE_DEPENDENCIES),
        "methods": {
            "fft_gaussian": gaussian,
            "fourier_ls": fourier,
            "pinn": {
                "config": pinn_config,
                "selection_metric": pinn_selected["selection_metric"],
                "selection_score": pinn_selected["selection_score"],
                "selection_seed": int(pinn_selected["selection_seed"]),
                "test_seeds": [int(seed) for seed in PINN_CONFIG["test"]["seeds"]],
            },
        },
        "next_stage": {
            "fourier_ls_pending": False,
            "note": "All four pre-test artifacts verified; formal test gate is open.",
        },
    }

    utils.save_run_metadata(
        output_root,
        {
            "experiment": "Global Configuration Lock",
            "manifest": utils.project_relative(MANIFEST),
            "formal_config_sha256": lock["formal_config"]["sha256"],
            "selection_artifacts": selection_snapshot,
            "force_overwrite_output": force,
            "formal_test_ready": True,
        },
    )
    utils.write_json(LOCK_PATH, lock)
    print(f"Global Configuration Lock written: {LOCK_PATH}")
    print(
        "Frozen:",
        f"Gaussian sigma={gaussian['sigma']}",
        f"Cosine-LS lambda_LS={fourier['lambda_ls']}",
        f"Cosine-LS sparse={fourier['sparse_weight']}",
        f"PINN steps={pinn_config['steps']}",
        "formal_test_ready=True",
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace only the Global Lock output")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(force=args.force)
