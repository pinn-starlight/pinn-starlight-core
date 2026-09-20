"""Stage10-B U-Net train/validation selector."""
from __future__ import annotations
import argparse
import time
import torch

from experiments.common.baselines import unet_small as unet
from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import stage_b_provenance as provenance
from experiments.configs.stage_b_experiment import stage_b_selection_config

OUTPUT_ROOT = utils.OUTPUT_ROOT / "stage_b/selection/unet_small"
ARTIFACT = OUTPUT_ROOT / "unet_selected.json"
SOURCE_FILES = (
    "experiments/scripts/stage_b/select_unet.py",
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
        candidates.append({
            "seed": seed,
            "checkpoint": utils.project_relative(checkpoint),
            "sha256": utils.file_sha256(checkpoint),
            "validation_loss": float(checkpoint_data["validation_loss"]),
            "best_epoch": int(checkpoint_data["epoch"]),
            "training_time_s": float(training_time),
            "peak_vram_mb": utils.peak_vram_mb(device),
            "parameter_count": int(utils.parameter_count(model)),
        })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best = min(candidates, key=lambda x: x["validation_loss"])
    selected = {
        "training": dict(training),
        "inference": dict(inference),
        "checkpoints": candidates,
        "e4_checkpoint": best["checkpoint"],
        "e4_checkpoint_sha256": best["sha256"],
        "e4_seed": int(best["seed"]),
        "e4_validation_loss": float(best["validation_loss"]),
        "selection": dict(config["selection"]),
    }
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
