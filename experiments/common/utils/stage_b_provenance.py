"""Selection-artifact and lock provenance for Stage10-B."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.common.utils import experiment_utils as utils


def stable_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_snapshot(relative_paths) -> dict:
    records = []
    for relative in tuple(str(p).replace("\\", "/") for p in relative_paths):
        absolute = utils.PROJECT_ROOT / relative
        if not absolute.is_file():
            raise FileNotFoundError(f"Stage-B source dependency missing: {absolute}")
        records.append({"path": relative, "sha256": utils.file_sha256(absolute)})
    return {"files": records, "bundle_sha256": stable_json_sha256(records)}


def verify_source_snapshot(snapshot: dict, expected_paths=None):
    records = snapshot.get("files", [])
    if not records or stable_json_sha256(records) != snapshot.get("bundle_sha256"):
        raise ValueError("Stage-B source snapshot is missing or internally inconsistent")
    if expected_paths is not None:
        expected = tuple(str(p).replace("\\", "/") for p in expected_paths)
        actual = tuple(record["path"] for record in records)
        if actual != expected:
            raise ValueError("Stage-B source dependency list changed")
    for record in records:
        path = utils.PROJECT_ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Frozen Stage-B source missing: {path}")
        if utils.file_sha256(path) != record["sha256"]:
            raise ValueError(f"Stage-B source changed after selection/lock: {record['path']}")


def build_selection_artifact(
    *,
    method,
    selection_config,
    selected,
    manifest,
    rows_by_split,
    source_files,
    metadata=None,
):
    loaded = tuple(rows_by_split)
    if "test" in loaded:
        raise RuntimeError("Stage-B selector is not allowed to load test rows")
    return {
        "schema_version": 1,
        "artifact_state": "SELECTED",
        "method": method,
        "boundary": {
            "loaded_splits": list(loaded),
            "test_arrays_loaded_during_selection": False,
        },
        "data_provenance": {
            "manifest": utils.project_relative(manifest),
            "manifest_sha256": utils.file_sha256(manifest),
            "sample_ids": {
                split: [str(row["sample_id"]) for row in rows]
                for split, rows in rows_by_split.items()
            },
        },
        "selection_config": selection_config,
        "selection_config_sha256": stable_json_sha256(selection_config),
        "selected": selected,
        "source_provenance": source_snapshot(source_files),
        "metadata": metadata or {},
    }


def verify_selection_artifact(
    path,
    *,
    method,
    selection_config,
    manifest,
    required_splits,
    source_files,
):
    artifact = utils.read_json(path)
    if artifact.get("artifact_state") != "SELECTED" or artifact.get("method") != method:
        raise ValueError(f"Invalid Stage-B selection artifact: {path}")
    if tuple(artifact.get("boundary", {}).get("loaded_splits", [])) != tuple(required_splits):
        raise ValueError(f"Stage-B selector split mismatch for {method}")
    if artifact.get("boundary", {}).get("test_arrays_loaded_during_selection") is not False:
        raise RuntimeError(f"Stage-B {method} selector touched test arrays")

    current_cfg = stable_json_sha256(selection_config)
    if artifact.get("selection_config_sha256") != current_cfg:
        raise ValueError(f"Stage-B {method} selection config changed")

    data = artifact.get("data_provenance", {})
    if Path(utils.to_absolute_path(data["manifest"])).resolve() != Path(manifest).resolve():
        raise ValueError(f"Stage-B {method} selector used the wrong manifest")
    if utils.file_sha256(manifest) != data.get("manifest_sha256"):
        raise ValueError(f"Stage-B manifest changed after {method} selection")
    expected_ids = {
        split: [str(row["sample_id"]) for row in utils.load_manifest(manifest, split=split)]
        for split in required_splits
    }
    if data.get("sample_ids") != expected_ids:
        raise ValueError(f"Stage-B {method} selector sample IDs changed")

    verify_source_snapshot(artifact["source_provenance"], source_files)
    return artifact
