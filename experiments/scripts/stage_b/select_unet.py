"""Stage-B U-Net train/validation selector.

Reuses the Stage-A training loop and payload assembly (identical architectures
and budgets); only the manifest, config source and artifact schema differ.
"""
from __future__ import annotations
import argparse

import torch

from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import stage_b_provenance as provenance
from experiments.configs.stage_b_experiment import stage_b_selection_config
from experiments.scripts.selection.unet import (
    selected_payload,
    train_seed_candidates,
)

OUTPUT_ROOT = utils.OUTPUT_ROOT / "stage_b/selection/unet_small"
ARTIFACT = OUTPUT_ROOT / "unet_selected.json"
SOURCE_FILES = (
    "experiments/scripts/stage_b/select_unet.py",
    "experiments/scripts/selection/unet.py",
    "experiments/configs/stage_b_experiment.py",
    "experiments/common/data/generate_stage_b.py",
    "experiments/common/data/stage_b_backgrounds.py",
    "experiments/common/baselines/unet_small.py",
    "experiments/common/utils/experiment_utils.py",
)


def main(*, force=False):
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    train_rows = utils.load_manifest(STAGE_B_MANIFEST, split="train")
    validation_rows = utils.load_manifest(STAGE_B_MANIFEST, split="validation")
    if not train_rows or not validation_rows:
        raise ValueError("Stage-B train/validation splits must be non-empty")

    config = stage_b_selection_config("unet_small")
    training = config["training"]
    inference = config["inference"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    candidates = train_seed_candidates(
        train_rows=train_rows,
        validation_rows=validation_rows,
        training=training,
        inference=inference,
        output_root=output_root,
        device=device,
    )
    selected = selected_payload(candidates, training, inference, config["selection"])
    artifact = provenance.build_selection_artifact(
        method="unet_small",
        selection_config=config,
        selected=selected,
        manifest=STAGE_B_MANIFEST,
        rows_by_split={"train": train_rows, "validation": validation_rows},
        source_files=SOURCE_FILES,
        metadata={"device": str(device)},
    )
    utils.save_rows_csv(output_root / "candidates.csv", candidates)
    utils.write_json(ARTIFACT, artifact)
    print(
        f"Stage-B U-Net selected seed={selected['e4_seed']} "
        f"validation_loss={selected['e4_validation_loss']:.8g}: {ARTIFACT}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
