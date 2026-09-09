"""Validation-only raw/nonnegative/legacy detector comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics


def _find_validation_row(sample_id, manifest):
    rows = utils.load_manifest(manifest)
    matches = [row for row in rows if row["sample_id"] == sample_id]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {sample_id}, found {len(matches)}")
    row = matches[0]
    if row.get("split") != "validation":
        raise RuntimeError(
            "detection-mode debug is validation-only; refusing "
            f"{sample_id} (split={row.get('split')})"
        )
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=utils.SYNTHETIC_MANIFEST)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    row = _find_validation_row(args.sample_id, args.manifest)
    sample = utils.load_synthetic_sample(row)
    residual = utils.load_float_tiff(args.residual)
    if residual.shape != sample["clean_true"].shape:
        raise ValueError(
            f"residual shape mismatch: {residual.shape} vs {sample['clean_true'].shape}"
        )

    reference = sample["metadata"]["star_reference"]
    results = metrics.compare_star_detection_modes(
        residual,
        threshold=float(reference["threshold"]),
        fwhm=float(reference["fwhm"]),
        matching_radius=float(reference["matching_radius"]),
        reference_stars=reference["stars"],
        reference_count_capped=bool(reference.get("count_capped", False)),
        photometry=reference["photometry"],
    )
    payload = {
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "formal_detection_mode": metrics.FORMAL_DETECTION_MODE,
        "debug_modes": list(metrics.DETECTION_MODES),
        "photometry_array": "raw residual for every mode",
        "results": results,
    }
    output = args.output or (
        utils.OUTPUT_ROOT / "detection_mode_debug" / f"{sample['sample_id']}.json"
    )
    utils.write_json(output, payload)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
