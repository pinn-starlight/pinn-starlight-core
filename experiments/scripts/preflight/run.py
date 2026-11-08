"""Read-only preflight for the Stage-10 formal experiment pipeline.

Run once before rebuilding the synthetic dataset and again immediately after it
is generated.  The script never modifies data or outputs.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.common.data import generate_synthetic as generator
from experiments.common.data.synthetic_backgrounds import BACKGROUND_TYPES
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.configs.formal_experiment import SYNTHETIC_DATA_CONFIG

EXPECTED_REAL_IMAGE_COUNT = 16
EXPECTED_SAMPLE_COUNT = 30
EXPECTED_SPLIT_COUNTS = {"train": 18, "validation": 6, "test": 6}


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _check_collection_manifest():
    path = utils.COLLECTION_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"collections manifest missing: {path}")
    rows = _read_csv(path)
    by_id = {row.get("image_id"): row for row in rows}
    missing = [base_id for base_id in generator.FIXED_BASE_SPLIT if base_id not in by_id]
    if missing:
        raise ValueError("missing formal camera-reference bases: " + ", ".join(missing))
    for base_id in generator.FIXED_BASE_SPLIT:
        row = by_id[base_id]
        if row.get("usage") != "synthetic_base_candidate":
            raise ValueError(f"{base_id} must have usage=synthetic_base_candidate")
        image_path = utils.resolve_collection_path(row["current_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"camera reference missing: {base_id}: {image_path}")
    return rows


def _check_raw_layout():
    if not utils.NATIVE_TEST_IMAGE.is_file():
        raise FileNotFoundError(f"E0 native test image missing: {utils.NATIVE_TEST_IMAGE}")
    real_images = utils.supported_image_files(utils.REAL_HIGH_POLLUTION_ROOT)
    if len(real_images) != EXPECTED_REAL_IMAGE_COUNT:
        raise ValueError(
            f"expected {EXPECTED_REAL_IMAGE_COUNT} E4 images in "
            f"{utils.REAL_HIGH_POLLUTION_ROOT}; found {len(real_images)}"
        )
    if utils.SYNTHETIC_ROOT.resolve() == utils.LEGACY_SYNTHETIC_ROOT.resolve():
        raise AssertionError("formal and legacy synthetic roots are not isolated")
    return real_images


def _check_generated_synthetic():
    manifest = utils.SYNTHETIC_MANIFEST
    if not utils.SYNTHETIC_ROOT.exists():
        return {"state": "not_generated"}
    if not manifest.is_file():
        raise FileNotFoundError(
            f"formal synthetic root exists but manifest is missing: {manifest}"
        )

    protocol_path = utils.SYNTHETIC_ROOT / "dataset_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Stage-10 dataset protocol missing: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if int(protocol.get("protocol_version", 0)) != int(
        SYNTHETIC_DATA_CONFIG["protocol_version"]
    ):
        raise ValueError("synthetic dataset protocol version is not Stage 10")
    protocol_split = protocol.get("base_split", {})
    if protocol_split != generator.FIXED_BASE_SPLIT:
        raise ValueError(
            "synthetic dataset base split does not match the active ordinary/CV fold protocol"
        )
    model = protocol.get("observation_model", {})
    if model.get("additional_synthetic_noise") is not False:
        raise ValueError("Stage-10 main benchmark must not inject extra synthetic noise")
    if model.get("skyglow_induced_shot_noise_simulated") is not False:
        raise ValueError("Stage-10 main benchmark must not invent calibrated shot noise")
    if model.get("scientific_clipping") is not False:
        raise ValueError("Stage-10 scientific arrays must remain unclipped")

    rows = _read_csv(manifest)
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"formal synthetic manifest has {len(rows)} rows; expected 30")
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample_id in formal synthetic manifest")
    split_counts = Counter(row["split"] for row in rows)
    if dict(split_counts) != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"formal split counts are {dict(split_counts)}; expected {EXPECTED_SPLIT_COUNTS}"
        )
    if set(row["background_type"] for row in rows) != set(BACKGROUND_TYPES):
        raise ValueError("formal synthetic background types do not match Stage-10 protocol")
    if set(row["intensity_level"] for row in rows) != set(generator.INTENSITY_LEVELS):
        raise ValueError("formal synthetic intensity levels do not match Stage-10 protocol")
    for row in rows:
        base_id = str(row.get("base_id", ""))
        expected_split = generator.FIXED_BASE_SPLIT.get(base_id)
        if expected_split is None or row.get("split") != expected_split:
            raise ValueError(
                f"formal synthetic row split mismatch for {base_id}: "
                f"{row.get('split')} != {expected_split}"
            )

    forbidden = {"noise_std", "noise_seed"}
    checked_native_audits = set()
    for row in rows:
        if forbidden.intersection(row):
            raise ValueError("legacy synthetic-noise fields found in formal manifest")
        for key in ("clean_true", "background_true", "observed", "metadata"):
            path = utils.to_absolute_path(row[key])
            if not path.is_file():
                raise FileNotFoundError(f"formal synthetic file missing: {path}")

        # Before the fold/global lock, test rows are checked only for manifest/path
        # integrity.  Scientific test arrays are deliberately left unopened.
        if row["split"] == "test":
            continue

        sample = utils.load_synthetic_sample(row)
        identity = float(
            np.max(
                np.abs(
                    sample["observed"]
                    - sample["clean_true"]
                    - sample["background_true"]
                )
            )
        )
        if identity >= 1e-6:
            raise ValueError(f"additive identity failed for {row['sample_id']}: {identity}")

        metadata = sample["metadata"]
        observation_model = metadata.get("observation_model", {})
        if observation_model.get("additional_synthetic_noise") is not False:
            raise ValueError("formal metadata must explicitly disable extra synthetic noise")
        if "noise_std" in observation_model or "noise_seed" in observation_model:
            raise ValueError("legacy synthetic-noise fields found in observation metadata")
        if observation_model.get("low_high_semantics") != (
            "skyglow_intensity_and_spatial_scale_not_noise_level"
        ):
            raise ValueError("low/high must be documented as skyglow levels, not noise levels")

        star_reference = metadata.get("star_reference", {})
        if int(star_reference.get("evaluation_schema_version", 0)) != metrics.EVALUATION_SCHEMA_VERSION:
            raise ValueError("star-reference evaluation schema mismatch")
        if star_reference.get("catalog_source") != "native_background_flattened_camera_reference":
            raise ValueError("star reference must come from the flattened camera reference")

        native = metadata.get("preprocessing", {}).get("native_background", {})
        audit_path = native.get("audit_path")
        if not audit_path:
            raise ValueError("sample metadata is missing native-background audit provenance")
        audit_path = utils.to_absolute_path(audit_path)
        if audit_path not in checked_native_audits:
            audit_record = json.loads(audit_path.read_text(encoding="utf-8"))
            if bool(SYNTHETIC_DATA_CONFIG["native_background"].get("strict_audit", True)):
                if audit_record.get("audit", {}).get("pass") is not True:
                    raise ValueError(f"native-background audit is not PASS: {audit_path}")
            checked_native_audits.add(audit_path)

    return {
        "state": "complete",
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "native_audits": len(checked_native_audits),
        "numeric_arrays_checked_before_lock": "train+validation_only",
        "test_arrays_opened_before_lock": False,
    }


def main():
    _check_collection_manifest()
    real_images = _check_raw_layout()
    synthetic = _check_generated_synthetic()
    print("Stage-10 formal experiment preflight: PASS")
    print(f"data_root={utils.DATA_ROOT}")
    print(f"camera_reference_bases={len(generator.FIXED_BASE_SPLIT)}")
    print(f"planned_synthetic={EXPECTED_SAMPLE_COUNT} (5 x 3 x 2)")
    print(f"e4_real_images={len(real_images)}")
    print(f"legacy_synthetic_present={utils.LEGACY_SYNTHETIC_ROOT.exists()}")
    print(f"formal_synthetic={synthetic}")
    if synthetic["state"] == "not_generated":
        print("Next: rebuild the Stage-10 synthetic dataset with --force, then preflight again.")
    else:
        print("pre-lock numeric arrays checked: train + validation only")
        print("test arrays opened before lock: False")
        print(
            "Next: optional E0 smoke; rerun all four pre-test selectors; rebuild the "
            "Global Configuration Lock; then run E2/E3/E4. Stage-9 artifacts are invalid."
        )


if __name__ == "__main__":
    main()
