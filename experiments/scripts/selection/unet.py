"""Train/select formal U-Net seeds using train+validation only."""

from __future__ import annotations

import argparse
import time

import torch

from experiments.common.baselines import unet_small as unet
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import selection_artifacts as artifacts
from experiments.configs.formal_experiment import selection_dependency_config

OUTPUT_ROOT = utils.UNET_SELECTION_ROOT
ARTIFACT_PATH = utils.UNET_SELECTION_ARTIFACT
MANIFEST = utils.SYNTHETIC_MANIFEST


def train_seed_candidates(
    *,
    train_rows,
    validation_rows,
    training: dict,
    inference: dict,
    output_root,
    device,
):
    """Train one U-Net per configured seed and describe each checkpoint.

    Shared by the Stage-A and Stage-B selectors: they differ only in which
    manifest/config they feed in and how the resulting artifact is persisted.
    """
    candidates = []
    for seed in training["seeds"]:
        seed = int(seed)
        checkpoint_dir = output_root / "checkpoints" / f"seed_{seed}"
        utils.reset_peak_vram(device)
        started = time.perf_counter()
        checkpoint = unet.train(
            train_rows,
            validation_rows,
            checkpoint_dir,
            device=device,
            epochs=int(training["epochs"]),
            steps_per_epoch=int(training["steps_per_epoch"]),
            batch_size=int(training["batch_size"]),
            patch_size=int(training["patch_size"]),
            learning_rate=float(training["learning_rate"]),
            base_channels=int(training["base_channels"]),
            patience=int(training["patience"]),
            min_delta_relative=float(training["min_delta_relative"]),
            use_amp=bool(training["amp_requested"]),
            inference_mode=str(inference["mode"]),
            inference_tile_size=int(inference["tile_size"]),
            inference_overlap=int(inference["overlap"]),
            seed=seed,
            show_progress=True,
        )
        training_time = time.perf_counter() - started
        model, checkpoint_data = unet.load_checkpoint_model(checkpoint, device)
        candidates.append(
            {
                "seed": seed,
                "checkpoint": utils.project_relative(checkpoint),
                "sha256": utils.file_sha256(checkpoint),
                "validation_loss": float(checkpoint_data["validation_loss"]),
                "best_epoch": int(checkpoint_data["epoch"]),
                "training_time_s": float(training_time),
                "peak_vram_mb": utils.peak_vram_mb(device),
                "parameter_count": int(utils.parameter_count(model)),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return candidates


def selected_payload(candidates, training: dict, inference: dict, selection: dict):
    """Assemble the `selected` payload shared by both stages.

    The E4 checkpoint is the seed with the lowest validation loss.
    """
    best = min(candidates, key=lambda row: row["validation_loss"])
    return {
        "training": dict(training),
        "inference": dict(inference),
        "checkpoints": candidates,
        "e4_checkpoint": best["checkpoint"],
        "e4_checkpoint_sha256": best["sha256"],
        "e4_seed": int(best["seed"]),
        "e4_validation_loss": float(best["validation_loss"]),
        "selection": dict(selection),
    }


def main(*, force: bool = False):
    print("U-Net train/validation selection")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    train_rows = utils.load_manifest(MANIFEST, split="train")
    validation_rows = utils.load_manifest(MANIFEST, split="validation")
    if not train_rows or not validation_rows:
        raise ValueError("train and validation splits must be non-empty")
    if any(row.get("split") == "test" for row in train_rows + validation_rows):
        raise AssertionError("test row entered U-Net pre-test training/selection")

    config = selection_dependency_config("unet_small")
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
    artifact = artifacts.build_selection_artifact(
        method="unet_small",
        formal_config=config,
        selected=selected,
        manifest=MANIFEST,
        rows_by_split={"train": train_rows, "validation": validation_rows},
        metadata={"device": str(device)},
    )
    utils.save_rows_csv(output_root / "candidates.csv", candidates)
    utils.write_json(ARTIFACT_PATH, artifact)
    print(
        f"U-Net selected E4 seed={selected['e4_seed']} "
        f"validation_loss={selected['e4_validation_loss']:.8g}: {ARTIFACT_PATH}"
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace this selector output")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(force=args.force)
