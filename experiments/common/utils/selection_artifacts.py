"""Pre-test selection artifact provenance and dependency verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.common.utils import experiment_utils as utils

SELECTION_ARTIFACT_SCHEMA_VERSION = 1

# Source dependencies are deliberately method-local. A change to one method or
# its selector makes only that selected artifact stale before a new Global Lock.
_COMMON_DATA_DEPENDENCY = "experiments/common/utils/experiment_utils.py"
SELECTION_SOURCE_DEPENDENCIES = {
    "pinn": (
        "experiments/scripts/e1/run.py",
        "experiments/common/baselines/coordinate_pinn.py",
        "experiments/common/utils/metrics.py",
        "src/pinn_starlight_core/data/image_loader.py",
        "src/pinn_starlight_core/nn/physics_model.py",
        "src/pinn_starlight_core/physics/forcing.py",
        "src/pinn_starlight_core/nn/pinn_layers.py",
        "src/pinn_starlight_core/nn/pinn_loss.py",
        _COMMON_DATA_DEPENDENCY,
    ),
    "fft_gaussian": (
        "experiments/scripts/selection/gaussian/run.py",
        "experiments/common/baselines/fft_gaussian.py",
        _COMMON_DATA_DEPENDENCY,
    ),
    "fourier_ls": (
        "experiments/scripts/selection/fourier_ls/run.py",
        "experiments/common/baselines/fft_screened_poisson.py",
        "src/pinn_starlight_core/physics/forcing.py",
        _COMMON_DATA_DEPENDENCY,
    ),
    "unet_small": (
        "experiments/scripts/selection/unet/run.py",
        "experiments/common/baselines/unet_small.py",
        _COMMON_DATA_DEPENDENCY,
    ),
}

# After the Global Lock exists, any change to formal code must invalidate the
# lock before test is touched. This closes the remaining source-edit freeze gap.
FORMAL_SOURCE_DEPENDENCIES = tuple(
    dict.fromkeys(
        (
            "experiments/configs/formal_experiment.py",
            "experiments/configs/split_protocols.py",
            "experiments/common/utils/selection_artifacts.py",
            "experiments/common/utils/metrics.py",
            "experiments/common/data/generate_synthetic.py",
            "experiments/common/data/native_background.py",
            "experiments/common/data/synthetic_backgrounds.py",
            "experiments/scripts/preflight/run.py",
            "experiments/scripts/config_lock/run.py",
            "experiments/scripts/e2/run.py",
            "experiments/scripts/e3/run.py",
            "experiments/scripts/e4/run.py",
            "experiments/common/baselines/decoupled_gradient_pinn.py",
            "experiments/scripts/final_pinn/common.py",
            "experiments/scripts/final_pinn/stage_a.py",
            "experiments/scripts/final_pinn/aggregate_cv5.py",
        )
        + tuple(
            path
            for dependencies in SELECTION_SOURCE_DEPENDENCIES.values()
            for path in dependencies
        )
    )
)


def stable_json_sha256(value) -> str:
    """Hash a JSON-compatible value independent of dict insertion order."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_provenance(relative_paths) -> dict:
    """Fingerprint a deterministic list of project source files."""
    paths = tuple(str(path).replace("\\", "/") for path in relative_paths)
    records = []
    for relative in paths:
        absolute = utils.PROJECT_ROOT / relative
        if not absolute.is_file():
            raise FileNotFoundError(f"Source dependency missing: {absolute}")
        records.append({"path": relative, "sha256": utils.file_sha256(absolute)})
    return {
        "files": records,
        "bundle_sha256": stable_json_sha256(records),
    }


def verify_source_provenance(provenance: dict, expected_paths=None) -> None:
    """Reject a source snapshot whose path set or file content has changed."""
    if not isinstance(provenance, dict):
        raise ValueError("Missing source provenance")
    records = provenance.get("files", [])
    if not isinstance(records, list) or not records:
        raise ValueError("Source provenance contains no files")
    actual_paths = tuple(record.get("path") for record in records)
    if expected_paths is not None:
        normalized_expected = tuple(
            str(path).replace("\\", "/") for path in expected_paths
        )
        if actual_paths != normalized_expected:
            raise ValueError("Source dependency list changed; rerun the affected selector")
    if stable_json_sha256(records) != provenance.get("bundle_sha256"):
        raise ValueError("Stored source provenance bundle is internally inconsistent")
    for record in records:
        relative = record.get("path")
        expected_sha = record.get("sha256")
        absolute = utils.PROJECT_ROOT / str(relative)
        if not absolute.is_file():
            raise FileNotFoundError(f"Frozen source file missing: {absolute}")
        if utils.file_sha256(absolute) != expected_sha:
            raise ValueError(f"Source changed after selection/lock: {relative}")


def build_selection_artifact(
    *,
    method: str,
    formal_config: dict,
    selected: dict,
    manifest,
    rows_by_split: dict[str, list[dict]],
    metadata: dict | None = None,
) -> dict:
    """Build one method-local artifact using train/validation rows only."""
    if method not in SELECTION_SOURCE_DEPENDENCIES:
        raise ValueError(f"Unknown selection method: {method}")
    loaded_splits = list(rows_by_split)
    if "test" in loaded_splits:
        raise ValueError("Pre-test selection artifact cannot include the test split")
    sample_ids = {
        split: [str(row["sample_id"]) for row in rows]
        for split, rows in rows_by_split.items()
    }
    return {
        "schema_version": SELECTION_ARTIFACT_SCHEMA_VERSION,
        "artifact_state": "SELECTED",
        "method": method,
        "boundary": {
            "loaded_splits": loaded_splits,
            "test_arrays_loaded_during_selection": False,
        },
        "data_provenance": {
            "manifest": utils.project_relative(manifest),
            "manifest_sha256": utils.file_sha256(manifest),
            "sample_ids": sample_ids,
        },
        "formal_config": formal_config,
        "formal_config_sha256": stable_json_sha256(formal_config),
        "source_provenance": source_provenance(SELECTION_SOURCE_DEPENDENCIES[method]),
        "selected": selected,
        "metadata": metadata or {},
    }


def verify_selection_artifact(
    path,
    *,
    method: str,
    formal_config: dict,
    required_splits,
    verify_sources: bool = True,
) -> dict:
    """Load a selected artifact and verify config, data, split, and source deps."""
    path = utils.to_absolute_path(path)
    artifact = utils.read_json(path)
    if int(artifact.get("schema_version", 0)) != SELECTION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported selection artifact schema: {path}")
    if artifact.get("artifact_state") != "SELECTED":
        raise ValueError(f"Selection artifact is not SELECTED: {path}")
    if artifact.get("method") != method:
        raise ValueError(f"Selection artifact method mismatch: {path}")

    stored_config = artifact.get("formal_config")
    stored_hash = artifact.get("formal_config_sha256")
    if stable_json_sha256(stored_config) != stored_hash:
        raise ValueError(f"Selection artifact config snapshot is inconsistent: {path}")
    current_hash = stable_json_sha256(formal_config)
    if current_hash != stored_hash:
        raise ValueError(
            f"{method} formal config changed after selection; rerun only this selector"
        )

    boundary = artifact.get("boundary", {})
    loaded_splits = tuple(boundary.get("loaded_splits", []))
    required_splits = tuple(required_splits)
    if loaded_splits != required_splits:
        raise ValueError(
            f"{method} selection split mismatch: {loaded_splits} != {required_splits}"
        )
    if "test" in loaded_splits or boundary.get("test_arrays_loaded_during_selection") is not False:
        raise RuntimeError(f"{method} selection artifact violates the pre-test boundary")

    data = artifact.get("data_provenance", {})
    manifest = utils.to_absolute_path(data.get("manifest", ""))
    if manifest.resolve() != utils.SYNTHETIC_MANIFEST.resolve():
        raise ValueError(f"{method} selection used a different synthetic manifest")
    if utils.file_sha256(manifest) != data.get("manifest_sha256"):
        raise ValueError(f"Synthetic manifest changed after {method} selection")
    expected_ids = {
        split: [str(row["sample_id"]) for row in utils.load_manifest(manifest, split=split)]
        for split in required_splits
    }
    if data.get("sample_ids") != expected_ids:
        raise ValueError(f"{method} selection sample IDs no longer match the manifest")

    if verify_sources:
        verify_source_provenance(
            artifact.get("source_provenance", {}),
            SELECTION_SOURCE_DEPENDENCIES[method],
        )
    if not isinstance(artifact.get("selected"), dict):
        raise ValueError(f"{method} selection artifact has no selected payload")
    return artifact
