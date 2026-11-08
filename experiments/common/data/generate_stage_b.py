"""Generate the Stage10-B frequency-overlap stress-test dataset.

This module only adds new Stage-B data.  It does not modify Stage10-A data,
selection artifacts, checkpoints, or the Stage10-A Global Configuration Lock.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.common.data import generate_synthetic as stage_a_generator
from experiments.common.data.stage_b_backgrounds import SCENE_TYPES, generate_stage_b_scene
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.configs.stage_b_experiment import (
    STAGE_B_BASE_SPLIT,
    STAGE_B_CONFIG,
    STAGE_B_SEED,
)

STAGE_B_DATA_ROOT = utils.COLLECTION_ROOT / "synthetic_stage_b"
if utils.SPLIT_MODE == "standard":
    STAGE_B_ROOT = STAGE_B_DATA_ROOT
else:
    STAGE_B_ROOT = STAGE_B_DATA_ROOT / "cv5" / f"fold_{utils.CV_FOLD}"
STAGE_B_MANIFEST = STAGE_B_ROOT / "manifest.csv"


def _stable_seed(*parts) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _base_candidates(manifest_path):
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    by_id = {row["image_id"]: row for row in rows if row.get("usage") == "synthetic_base_candidate"}
    missing = [base_id for base_id in STAGE_B_BASE_SPLIT if base_id not in by_id]
    if missing:
        raise ValueError("Stage-B missing required base images: " + ", ".join(missing))
    return [by_id[base_id] for base_id in STAGE_B_BASE_SPLIT]


def _base_star_coordinates(base_clean):
    photometry = metrics.default_photometry_config()
    stars, capped = metrics.build_star_catalog(
        base_clean,
        threshold=stage_a_generator.STAR_THRESHOLD,
        fwhm=stage_a_generator.STAR_FWHM,
        max_stars=metrics.STAR_COUNT_LIMIT,
        photometry=photometry,
        detection_mode=metrics.REFERENCE_DETECTION_MODE,
    )
    return stars, bool(capped), photometry


def _scene_star_reference(clean_true, base_stars, count_capped, photometry):
    # Coordinates are frozen from base_clean, while the reference aperture flux
    # is measured on the actual Stage-B clean target (base + protected structure).
    stars = metrics.attach_raw_aperture_fluxes(clean_true, base_stars, photometry=photometry)
    return {
        "evaluation_schema_version": metrics.EVALUATION_SCHEMA_VERSION,
        "catalog_source": "stage_b_base_clean_coordinates",
        "coordinate_source": "native_background_flattened_camera_reference",
        "photometry_source": "stage_b_clean_true_base_plus_protected_structure",
        "threshold": stage_a_generator.STAR_THRESHOLD,
        "fwhm": stage_a_generator.STAR_FWHM,
        "matching_radius": stage_a_generator.MATCHING_RADIUS,
        "max_stars": metrics.STAR_COUNT_LIMIT,
        "count_capped": bool(count_capped),
        "detection_image": {
            "reference_mode": metrics.REFERENCE_DETECTION_MODE,
            "reference_rule": metrics.DETECTION_IMAGE_RULES[metrics.REFERENCE_DETECTION_MODE],
            "prediction_mode": metrics.FORMAL_DETECTION_MODE,
            "prediction_rule": metrics.DETECTION_IMAGE_RULE,
            "scientific_residual_clipping": False,
        },
        "photometry": photometry,
        "stars": stars,
    }


def _save_array(sample_dir: Path, name: str, sample_id: str, image) -> str:
    tiff_path = sample_dir / f"{name}_{sample_id}.tif"
    utils.save_float_tiff(tiff_path, image)
    # Scientific TIFF remains unclipped; PNG is display-only.
    utils.save_display_png(sample_dir / f"{name}_{sample_id}.png", image, clip=True)
    return utils.project_relative(tiff_path)


def build_dataset(
    *,
    collection_manifest=utils.COLLECTION_MANIFEST,
    output_root=STAGE_B_ROOT,
    seed=STAGE_B_SEED,
    center_crop_size=0,
    force=False,
):
    candidates = _base_candidates(collection_manifest)
    output_root = stage_a_generator._prepare_synthetic_output_root(output_root, force=force)
    rows = []
    dataset_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")

    for candidate in candidates:
        base_id = candidate["image_id"]
        split = STAGE_B_BASE_SPLIT[base_id]
        input_file = utils.resolve_collection_path(candidate["current_path"])
        base_clean, native_background_record = stage_a_generator._prepare_native_reference(
            input_file, output_root, base_id, center_crop_size
        )
        base_stars, count_capped, photometry = _base_star_coordinates(base_clean)

        for scene_type in SCENE_TYPES:
            scene_seed = _stable_seed(seed, base_id, scene_type, "stage_b_scene")
            clean_cache = None
            protected_cache = None
            star_reference_cache = None
            for intensity_level in STAGE_B_CONFIG["intensity_levels"]:
                background_true, protected_true, generation_truth = generate_stage_b_scene(
                    base_clean.shape[0],
                    base_clean.shape[1],
                    scene_type,
                    intensity_level,
                    scene_seed,
                )
                clean_true = (base_clean + protected_true).astype(np.float32)
                observed = (clean_true + background_true).astype(np.float32)

                # Low/high variants share identical geometry and protected target.
                if clean_cache is None:
                    clean_cache = clean_true.copy()
                    protected_cache = protected_true.copy()
                    star_reference_cache = _scene_star_reference(
                        clean_true, base_stars, count_capped, photometry
                    )
                else:
                    if not np.array_equal(clean_true, clean_cache):
                        raise AssertionError("Stage-B low/high clean target changed unexpectedly")
                    if not np.array_equal(protected_true, protected_cache):
                        raise AssertionError(
                            "Stage-B low/high protected field changed unexpectedly"
                        )

                identity_error = float(
                    np.max(np.abs(observed - base_clean - protected_true - background_true))
                )
                clean_identity_error = float(
                    np.max(np.abs(clean_true - base_clean - protected_true))
                )
                if identity_error > 2e-6 or clean_identity_error > 2e-6:
                    raise AssertionError("Stage-B additive identity failed")

                sample_id = f"{base_id}_{scene_type}_{intensity_level}"
                sample_dir = Path(output_root) / sample_id
                sample_dir.mkdir(parents=True, exist_ok=True)
                paths = {
                    "base_clean": _save_array(sample_dir, "base_clean", sample_id, base_clean),
                    "protected_true": _save_array(
                        sample_dir, "protected_true", sample_id, protected_true
                    ),
                    "clean_true": _save_array(sample_dir, "clean_true", sample_id, clean_true),
                    "background_true": _save_array(
                        sample_dir, "background_true", sample_id, background_true
                    ),
                    "observed": _save_array(sample_dir, "observed", sample_id, observed),
                }
                metadata = {
                    "sample_id": sample_id,
                    "base_id": base_id,
                    "split": split,
                    "scene_type": scene_type,
                    "background_type": scene_type,
                    "intensity_level": intensity_level,
                    "scene_seed": int(scene_seed),
                    "stage_b_protocol": STAGE_B_CONFIG,
                    "generation_truth": generation_truth,
                    "observation_model": {
                        "formula": "observed = base_clean + protected_true + background_true",
                        "clean_true_formula": "clean_true = base_clean + protected_true",
                        "additive_identity_max_abs": identity_error,
                        "clean_identity_max_abs": clean_identity_error,
                        "additional_synthetic_noise": False,
                        "scientific_clipping": False,
                        "observed_min": float(observed.min()),
                        "observed_max": float(observed.max()),
                        "outside_0_1_fraction": float(np.mean((observed < 0.0) | (observed > 1.0))),
                    },
                    "preprocessing": {
                        "raw_loader_downsample": stage_a_generator.RAW_DOWNSAMPLE,
                        "center_crop_size": int(center_crop_size),
                        "native_background": native_background_record,
                    },
                    "star_reference": star_reference_cache,
                }
                metadata_path = sample_dir / f"metadata_{sample_id}.json"
                utils.write_json(metadata_path, metadata)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "dataset_id": dataset_id,
                        "base_id": base_id,
                        "split": split,
                        "background_type": scene_type,
                        "scene_type": scene_type,
                        "intensity_level": intensity_level,
                        "scene_seed": int(scene_seed),
                        **paths,
                        "metadata": utils.project_relative(metadata_path),
                    }
                )

    expected = len(STAGE_B_BASE_SPLIT) * len(SCENE_TYPES) * len(STAGE_B_CONFIG["intensity_levels"])
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} Stage-B samples, generated {len(rows)}")

    manifest_path = Path(output_root) / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    utils.write_json(Path(output_root) / "base_split.json", STAGE_B_BASE_SPLIT)
    utils.write_json(
        Path(output_root) / "dataset_protocol.json",
        {
            **STAGE_B_CONFIG,
            "dataset_id": dataset_id,
            "manifest": utils.project_relative(manifest_path),
            "sample_count": len(rows),
            "split_counts": {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "validation", "test")
            },
            "seed_policy": {
                "scene": "sha256(seed, base_id, scene_type, stage_b_scene)",
                "low_high_share_geometry": True,
                "low_high_share_protected_structure": True,
            },
        },
    )
    return manifest_path, rows


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=STAGE_B_ROOT)
    parser.add_argument("--seed", type=int, default=STAGE_B_SEED)
    parser.add_argument("--center-crop-size", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    manifest, rows = build_dataset(
        output_root=args.output_root,
        seed=args.seed,
        center_crop_size=args.center_crop_size,
        force=args.force,
    )
    print(f"Stage10-B generated: {len(rows)} samples -> {manifest}")
