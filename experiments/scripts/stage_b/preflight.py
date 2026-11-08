"""Preflight for Stage10-B.

This preflight deliberately does NOT open Stage-B test arrays.
It verifies:
- manifest structure and 24/8/8 split;
- existence of all generated files;
- numeric/additive integrity on train + validation only;
- low/high pairing on train + validation;
- Stage10-A method/config compatibility relevant to Stage-B.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils.stage_b_anchor import load_stage_a_anchor
from experiments.configs.stage_b_experiment import STAGE_B_CONFIG

EXPECTED_COUNTS = {"train": 24, "validation": 8, "test": 8}


def _load_extra(row, key):
    if key not in row or not row[key]:
        raise ValueError(f"{row.get('sample_id')}: manifest is missing {key}")
    return utils.load_float_tiff(row[key])


def _max_abs(x):
    return float(np.max(np.abs(np.asarray(x, dtype=np.float64))))


def main():
    if not STAGE_B_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Stage-B manifest not found: {STAGE_B_MANIFEST}\n"
            "Run: python -m experiments.common.data.generate_stage_b --force"
        )

    protocol_path = STAGE_B_MANIFEST.parent / "dataset_protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Stage-B dataset protocol missing: {protocol_path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("base_split", {}) != STAGE_B_CONFIG["base_split"]:
        raise ValueError(
            "Stage-B dataset base split does not match the active ordinary/CV fold protocol"
        )

    rows = utils.load_manifest(STAGE_B_MANIFEST)
    if len(rows) != 40:
        raise ValueError(f"Stage-B expected 40 manifest rows, found {len(rows)}")

    counts = Counter(str(row["split"]) for row in rows)
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Stage-B split counts mismatch: {dict(counts)} != {EXPECTED_COUNTS}")
    for row in rows:
        base_id = str(row.get("base_id", ""))
        expected_split = STAGE_B_CONFIG["base_split"].get(base_id)
        if expected_split is None or row.get("split") != expected_split:
            raise ValueError(
                f"Stage-B row split mismatch for {base_id}: {row.get('split')} != {expected_split}"
            )

    # File existence for ALL splits is safe; do not decode test image arrays here.
    required_keys = (
        "observed",
        "clean_true",
        "background_true",
        "base_clean",
        "protected_true",
    )
    for row in rows:
        for key in required_keys:
            value = row.get(key)
            if not value:
                raise ValueError(f"{row['sample_id']}: missing path {key}")
            path = utils.to_absolute_path(value)
            if not path.is_file():
                raise FileNotFoundError(f"{row['sample_id']}: missing file {path}")

    # Numeric checks use only train + validation.
    checked_rows = [r for r in rows if r["split"] in ("train", "validation")]
    pairs = defaultdict(dict)

    for row in checked_rows:
        sample = utils.load_synthetic_sample(row)
        base_clean = _load_extra(row, "base_clean")
        protected = _load_extra(row, "protected_true")

        observed = np.asarray(sample["observed"], dtype=np.float64)
        clean = np.asarray(sample["clean_true"], dtype=np.float64)
        background = np.asarray(sample["background_true"], dtype=np.float64)
        base_clean = np.asarray(base_clean, dtype=np.float64)
        protected = np.asarray(protected, dtype=np.float64)

        if _max_abs(observed - clean - background) > 2e-6:
            raise ValueError(f"{row['sample_id']}: observed != clean_true + background_true")
        if _max_abs(clean - base_clean - protected) > 2e-6:
            raise ValueError(f"{row['sample_id']}: clean_true != base_clean + protected_true")

        key = (str(row["base_id"]), str(row.get("scene_type", row["background_type"])))
        pairs[key][str(row["intensity_level"])] = {
            "clean": clean,
            "protected": protected,
            "background": background,
        }

    for key, levels in pairs.items():
        if set(levels) != {"low", "high"}:
            raise ValueError(f"{key}: missing low/high pair")
        low, high = levels["low"], levels["high"]
        if _max_abs(low["clean"] - high["clean"]) > 2e-6:
            raise ValueError(f"{key}: low/high must share the same clean target")
        if _max_abs(low["protected"] - high["protected"]) > 2e-6:
            raise ValueError(f"{key}: low/high must share protected_true")

        low_peak = float(STAGE_B_CONFIG["intensity_levels"]["low"])
        high_peak = float(STAGE_B_CONFIG["intensity_levels"]["high"])
        expected_ratio = high_peak / low_peak
        denom = np.maximum(np.abs(low["background"]), 1e-8)
        valid = np.abs(low["background"]) > 1e-6
        if np.any(valid):
            ratio_err = np.median(np.abs(high["background"][valid] / denom[valid] - expected_ratio))
            if float(ratio_err) > 2e-3:
                raise ValueError(f"{key}: high/low pollution scaling is inconsistent")

    # A compatibility: relevant configs + method source only.
    a_lock = load_stage_a_anchor()

    print("Stage10-B preflight PASS")
    print(" manifest rows: 40")
    print(" split counts:", EXPECTED_COUNTS)
    print(" numeric arrays checked before B lock: train + validation only (32 rows)")
    print(" B test arrays opened before B lock: False")
    print(
        " Stage10-A scientific compatibility: PASS "
        f"(lock={a_lock.get('lock_state')}, formal_test_ready={a_lock.get('formal_test_ready')})"
    )


if __name__ == "__main__":
    main()
