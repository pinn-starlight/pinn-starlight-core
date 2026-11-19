"""Stage-B Gaussian validation selector."""

from __future__ import annotations

import argparse

from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import stage_b_provenance as provenance
from experiments.configs.stage_b_experiment import stage_b_selection_config
from experiments.scripts.selection.gaussian import validation_mse_scores

OUTPUT_ROOT = utils.OUTPUT_ROOT / "stage_b/selection/fft_gaussian"
ARTIFACT = OUTPUT_ROOT / "gaussian_selected.json"
SOURCE_FILES = (
    "experiments/scripts/stage_b/select_gaussian.py",
    "experiments/scripts/selection/gaussian.py",
    "experiments/configs/stage_b_experiment.py",
    "experiments/common/data/generate_stage_b.py",
    "experiments/common/data/stage_b_backgrounds.py",
    "experiments/common/baselines/fft_gaussian.py",
    "experiments/common/utils/experiment_utils.py",
)


def main(*, force=False):
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    rows = utils.load_manifest(STAGE_B_MANIFEST, split="validation")
    samples = [utils.load_synthetic_sample(row) for row in rows]
    config = stage_b_selection_config("fft_gaussian")
    candidates = validation_mse_scores(
        samples,
        config["selection"]["sigma_candidates"],
    )
    best = min(candidates, key=lambda x: x["validation_background_mse"])
    selected = {
        "sigma": float(best["sigma"]),
        "selection_metric": config["selection"]["selection_metric"],
        "candidates": candidates,
        "deterministic": bool(config["method"]["deterministic"]),
    }
    artifact = provenance.build_selection_artifact(
        method="fft_gaussian",
        selection_config=config,
        selected=selected,
        manifest=STAGE_B_MANIFEST,
        rows_by_split={"validation": rows},
        source_files=SOURCE_FILES,
    )
    utils.save_rows_csv(output_root / "candidates.csv", candidates)
    utils.write_json(ARTIFACT, artifact)
    print(f"Stage-B Gaussian selected sigma={selected['sigma']}: {ARTIFACT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
