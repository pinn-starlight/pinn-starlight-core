"""Select the FFT-Gaussian sigma on validation only and persist an artifact."""

from __future__ import annotations

import argparse

import numpy as np

from experiments.common.baselines import fft_gaussian as fft
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import selection_artifacts as artifacts
from experiments.configs.formal_experiment import (
    GAUSSIAN_CONFIG,
    selection_dependency_config,
)

OUTPUT_ROOT = utils.GAUSSIAN_SELECTION_ROOT
ARTIFACT_PATH = utils.GAUSSIAN_SELECTION_ARTIFACT
MANIFEST = utils.SYNTHETIC_MANIFEST


def validation_mse_scores(validation_samples, candidates):
    samples = list(validation_samples)
    if not samples:
        raise ValueError("validation_samples must not be empty")
    candidates = tuple(float(value) for value in candidates)
    if not candidates:
        raise ValueError("Gaussian sigma candidates must not be empty")

    rows = []
    for candidate in candidates:
        losses = []
        for sample in samples:
            observed = sample["observed"] if isinstance(sample, dict) else sample[0]
            background_true = sample["background_true"] if isinstance(sample, dict) else sample[1]
            observed = np.asarray(observed, dtype=np.float32)
            background_true = np.asarray(background_true, dtype=np.float32)
            if observed.shape != background_true.shape:
                raise ValueError("observed and background_true must have the same shape")
            prediction = fft.estimate_background(observed, candidate)
            losses.append(float(np.mean((prediction - background_true) ** 2)))
        rows.append(
            {
                "sigma": float(candidate),
                "validation_background_mse": float(np.mean(losses)),
            }
        )
    return rows


def main(*, force: bool = False):
    print("Gaussian validation selection")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    validation_rows = utils.load_manifest(MANIFEST, split="validation")
    if not validation_rows:
        raise ValueError("validation split must be non-empty")
    validation_samples = [utils.load_synthetic_sample(row) for row in validation_rows]

    config = selection_dependency_config("fft_gaussian")
    candidates = validation_mse_scores(
        validation_samples,
        config["selection"]["sigma_candidates"],
    )
    best = min(candidates, key=lambda row: row["validation_background_mse"])
    selected = {
        "sigma": float(best["sigma"]),
        "selection_metric": config["selection"]["selection_metric"],
        "candidates": candidates,
        "deterministic": bool(GAUSSIAN_CONFIG["method"]["deterministic"]),
    }
    artifact = artifacts.build_selection_artifact(
        method="fft_gaussian",
        formal_config=config,
        selected=selected,
        manifest=MANIFEST,
        rows_by_split={"validation": validation_rows},
    )
    utils.save_rows_csv(output_root / "candidates.csv", candidates)
    utils.write_json(ARTIFACT_PATH, artifact)
    print(f"Gaussian selected sigma={selected['sigma']}: {ARTIFACT_PATH}")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace this selector output")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(force=args.force)
