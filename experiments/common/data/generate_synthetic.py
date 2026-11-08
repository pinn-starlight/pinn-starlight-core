"""Build the Stage-10 formal semi-synthetic light-pollution dataset.

Protocol summary
----------------
* split camera-acquired bases first (fixed 3/1/1 split), then generate variants;
* estimate each base's native low-frequency sky with a fixed method-independent
  ``photutils.Background2D`` rule and subtract it once;
* preserve the camera-scale fluctuations and negative values remaining after the
  fixed low-frequency estimate is subtracted (no clipping and no min-max rescaling);
* 3 background types x 2 intensity/scale levels per base;
* ``observed = clean_true + background_true`` with no extra Gaussian/Poisson
  noise and no numeric clipping in the scientific float-TIFF path;
* background generation truth is stored for evaluation only and preprocessing
  products are stored for audit/reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.common.data.native_background import audit_flattening, flatten_reference
from experiments.common.data.synthetic_backgrounds import (
    BACKGROUND_TYPES,
    generate_background,
)
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.configs.formal_experiment import SYNTHETIC_DATA_CONFIG
from experiments.configs.split_protocols import active_base_split
from pinn_starlight_core.data.image_loader import ImageLoader


RAW_DOWNSAMPLE = 2
STAR_THRESHOLD = 0.03
STAR_FWHM = 3.0
MATCHING_RADIUS = 3.0

INTENSITY_LEVELS = {
    "low": {
        "intensity": 0.10,
        "scale": 0.55,
    },
    "high": {
        "intensity": 0.20,
        "scale": 0.85,
    },
}

# The split is part of the experiment protocol, not a generator side effect.
# Default imports use the ordinary fixed split; CV subprocesses select one fold
# through PINN_STARLIGHT_SPLIT_MODE=cv5 and PINN_STARLIGHT_CV_FOLD.
FIXED_BASE_SPLIT = active_base_split()


def load_clean_image(input_file, center_crop_size=0):
    """Load a clean base image as a [0,1] linear grayscale float array."""
    loader = ImageLoader(input_file, device="cpu", downsample=RAW_DOWNSAMPLE)
    _, brightness, width, height = loader.get_gray_data()
    if hasattr(brightness, "detach"):
        brightness = brightness.detach().cpu().numpy()
    image = np.asarray(brightness, dtype=np.float32).reshape(height, width)
    return center_crop(image, center_crop_size)


def center_crop(image, crop_size=0):
    crop_size = int(crop_size)
    if crop_size == 0:
        return image
    if crop_size < 0 or crop_size > min(image.shape):
        raise ValueError(f"invalid center_crop_size: {crop_size}")
    height, width = image.shape
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[top : top + crop_size, left : left + crop_size].copy()


def build_star_reference(clean_true):
    """Build and freeze the per-base catalogue from the flattened reference only."""
    photometry = metrics.default_photometry_config()
    stars, count_capped = metrics.build_star_catalog(
        clean_true,
        threshold=STAR_THRESHOLD,
        fwhm=STAR_FWHM,
        max_stars=metrics.STAR_COUNT_LIMIT,
        photometry=photometry,
        detection_mode=metrics.REFERENCE_DETECTION_MODE,
    )
    return {
        "evaluation_schema_version": metrics.EVALUATION_SCHEMA_VERSION,
        "catalog_source": "native_background_flattened_camera_reference",
        "legacy_variable_name": "clean_true",
        "reference_semantics": (
            "camera-acquired reference after fixed native low-frequency background "
            "subtraction; remaining camera fluctuations and negative values retained"
        ),
        "threshold": STAR_THRESHOLD,
        "fwhm": STAR_FWHM,
        "matching_radius": MATCHING_RADIUS,
        "max_stars": metrics.STAR_COUNT_LIMIT,
        "count_capped": bool(count_capped),
        "detection_image": {
            "reference_mode": metrics.REFERENCE_DETECTION_MODE,
            "reference_rule": metrics.DETECTION_IMAGE_RULES[
                metrics.REFERENCE_DETECTION_MODE
            ],
            "debug_candidate_prediction_modes": list(metrics.DETECTION_MODES),
            "scientific_residual_clipping": False,
        },
        "photometry": photometry,
        "stars": stars,
    }


def synthesize_from_clean(
    clean_true,
    background_type,
    intensity_level,
    geometry_seed,
):
    """Construct one strict additive semi-synthetic observation and its truth.

    Formal model:
        observed = clean_true + background_true

    ``clean_true`` is the camera-acquired reference after fixed native
    low-frequency background subtraction. Camera-scale fluctuations remain except for the deliberately removed low-frequency
    estimate. No additional Gaussian, Poisson, or Poisson-Gaussian noise is injected.

    ``geometry_seed`` is shared by low/high variants of one
    ``base_id × background_type`` pair. The level changes skyglow intensity
    and spatial scale, not a separate noise level.
    """
    if intensity_level not in INTENSITY_LEVELS:
        raise ValueError(f"invalid intensity_level: {intensity_level}")
    if background_type not in BACKGROUND_TYPES:
        raise ValueError(f"invalid background_type: {background_type}")

    clean_true = np.asarray(clean_true, dtype=np.float32)
    if clean_true.ndim != 2 or not np.isfinite(clean_true).all():
        raise ValueError("clean_true must be a finite 2-D array")

    config = INTENSITY_LEVELS[intensity_level]
    geometry_seed = int(geometry_seed)
    height, width = clean_true.shape

    background_true, generation_truth = generate_background(
        height,
        width,
        background_type=background_type,
        intensity=float(config["intensity"]),
        scale=float(config["scale"]),
        seed=geometry_seed,
    )

    # No numerical clipping: float TIFFs preserve the exact additive model.
    observed = (
        clean_true.astype(np.float32, copy=False)
        + background_true.astype(np.float32, copy=False)
    ).astype(np.float32)

    observation_model = {
        "formula": "observed = clean_true + background_true",
        "benchmark_type": "semi_synthetic_reference_plus_added_skyglow",
        "reference_image_semantics": (
            "native-background-flattened camera reference; remaining camera fluctuations "
            "retained without clipping/rescaling; not assumed noiseless"
        ),
        "background_semantics": "synthetically_added_skyglow_component",
        "task_scope": "light_pollution_background_estimation_and_removal_not_denoising",
        "intensity": float(config["intensity"]),
        "scale": float(config["scale"]),
        "scale_coordinate_system": "normalized_xy_minus1_to_1",
        "additional_synthetic_noise": False,
        "skyglow_induced_shot_noise_simulated": False,
        "shot_noise_limitation_reason": (
            "per-image gain/exposure/electron-or-ADU/black-level/read-noise calibration "
            "is unavailable; normalized intensities do not define a physical Poisson rate"
        ),
        "low_high_semantics": "skyglow_intensity_and_spatial_scale_not_noise_level",
        "geometry_seed": int(geometry_seed),
        "background_seed": int(geometry_seed),
        "numeric_clipping": False,
        "multiplicative_rescaling": False,
        "additive_identity_max_abs": float(
            np.max(np.abs(observed - clean_true - background_true))
        ),
        "observed_min": float(observed.min()),
        "observed_max": float(observed.max()),
        "outside_0_1_fraction": float(np.mean((observed < 0.0) | (observed > 1.0))),
    }
    return background_true, observed, observation_model, generation_truth


def save_sample(
    input_file,
    output_root,
    sample_id,
    base_id,
    split,
    background_type,
    intensity_level,
    geometry_seed,
    clean_true,
    star_reference,
    native_background_record,
    center_crop_size=0,
):
    background_true, observed, observation_model, generation_truth = synthesize_from_clean(
        clean_true,
        background_type,
        intensity_level,
        geometry_seed,
    )
    sample_dir = Path(output_root) / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, image in {
        "clean_true": clean_true,
        "background_true": background_true,
        "observed": observed,
    }.items():
        tiff_path = sample_dir / f"{name}_{sample_id}.tif"
        utils.save_float_tiff(tiff_path, image)
        utils.save_display_png(sample_dir / f"{name}_{sample_id}.png", image, clip=True)
        paths[name] = utils.project_relative(tiff_path)

    metadata = {
        "sample_id": sample_id,
        "base_id": base_id,
        "source_file": utils.project_relative(input_file),
        "split": split,
        "background_type": background_type,
        "intensity_level": intensity_level,
        "geometry_seed": int(geometry_seed),
        "observation_model": observation_model,
        "generation_truth": generation_truth,
        # Compatibility alias for older E3 code / existing notes.  New code
        # should prefer generation_truth.
        "background_params": generation_truth,
        "preprocessing": {
            "raw_loader_downsample": RAW_DOWNSAMPLE,
            "center_crop_size": int(center_crop_size),
            "native_background": native_background_record,
        },
        "star_reference": star_reference,
    }
    metadata_path = sample_dir / f"metadata_{sample_id}.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "sample_id": sample_id,
        "base_id": base_id,
        "split": split,
        "background_type": background_type,
        "intensity_level": intensity_level,
        "geometry_seed": int(observation_model["geometry_seed"]),
        "background_seed": int(observation_model["background_seed"]),
        **paths,
        "metadata": utils.project_relative(metadata_path),
    }


def _clean_candidates(manifest_path):
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    by_id = {
        row["image_id"]: row
        for row in rows
        if row.get("usage") == "synthetic_base_candidate"
    }
    missing = [base_id for base_id in FIXED_BASE_SPLIT if base_id not in by_id]
    if missing:
        raise ValueError(
            "formal synthetic split is missing required base images: "
            + ", ".join(missing)
        )
    return [by_id[base_id] for base_id in FIXED_BASE_SPLIT]


def _prepare_synthetic_output_root(output_root, force=False):
    """Return an empty dataset root, refusing stale data unless explicitly forced.

    Destructive ``force`` cleanup is intentionally restricted to children of
    the resolved ``collections`` directory so a mistyped path cannot remove an
    arbitrary folder.
    """
    path = Path(output_root)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    if not path.is_dir():
        raise NotADirectoryError(f"synthetic output path is not a directory: {path}")
    if not any(path.iterdir()):
        return path
    if not force:
        raise FileExistsError(
            f"synthetic output directory is not empty: {path}. "
            "Refusing to mix stale and current protocol data; rerun with --force "
            "to rebuild the formal dataset."
        )
    if path.is_symlink():
        raise ValueError("--force does not clean a symlinked synthetic output directory")

    resolved_path = path.resolve()
    allowed_root = utils.COLLECTION_ROOT.resolve()
    if resolved_path == allowed_root or allowed_root not in resolved_path.parents:
        raise ValueError(
            "--force synthetic cleanup is only allowed below the resolved collections root: "
            f"{resolved_path}"
        )
    shutil.rmtree(resolved_path)
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def _prepare_native_reference(input_file, output_root, base_id, center_crop_size):
    """Flatten one camera reference once and persist every audit product."""
    reference = load_clean_image(input_file, center_crop_size)
    config = dict(SYNTHETIC_DATA_CONFIG["native_background"])
    clean_true, native_background, estimator_info = flatten_reference(reference, config)
    audit = audit_flattening(reference, clean_true, native_background, config)

    audit_root = Path(output_root) / "_native_background" / str(base_id)
    audit_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, image in {
        "reference_original": reference,
        "native_background_estimate": native_background,
        "clean_true_flattened": clean_true,
    }.items():
        path = audit_root / f"{name}.tif"
        utils.save_float_tiff(path, image)
        utils.save_display_png(audit_root / f"{name}.png", image, clip=(name != "clean_true_flattened"))
        paths[name] = utils.project_relative(path)
    audit_path = audit_root / "audit.json"
    audit_record = {
        "stage": 10,
        "base_id": str(base_id),
        "estimator": estimator_info,
        "audit": audit,
        "paths": paths,
        "scientific_clipping": False,
        "multiplicative_rescaling": False,
        "camera_fluctuations_retained_after_low_frequency_subtraction": True,
    }
    audit_record["audit_path"] = utils.project_relative(audit_path)

    # Always persist the audit before a strict-audit failure is raised. Re-open
    # the file once so a failure cannot misleadingly point to a JSON that was
    # never actually written.
    utils.write_json(audit_path, audit_record)
    if not audit_path.is_file():
        raise OSError(f"native-background audit JSON was not created: {audit_path.resolve()}")
    try:
        persisted_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OSError(
            f"native-background audit JSON could not be read back: {audit_path.resolve()}"
        ) from exc
    if persisted_audit.get("base_id") != str(base_id) or "audit" not in persisted_audit:
        raise OSError(
            f"native-background audit JSON is incomplete: {audit_path.resolve()}"
        )

    if bool(config.get("strict_audit", True)) and not bool(audit["pass"]):
        failed_checks = [
            name for name, passed in audit.get("checks", {}).items() if not bool(passed)
        ]
        failed_text = ", ".join(failed_checks) if failed_checks else "unknown"
        raise RuntimeError(
            f"Stage-10 native-background audit failed for {base_id}: "
            f"{audit_path.resolve()}. "
            "The audit JSON was written successfully before this exception. "
            f"Failed checks: {failed_text}. "
            "Inspect the audit before changing any threshold or estimator setting."
        )
    return clean_true, audit_record


def build_dataset(
    collection_manifest=utils.COLLECTION_MANIFEST,
    output_root=utils.SYNTHETIC_ROOT,
    seed=20260728,
    center_crop_size=0,
    force=False,
):
    """Build the 30-sample formal synthetic dataset."""
    candidates = _clean_candidates(collection_manifest)
    output_root = _prepare_synthetic_output_root(output_root, force=force)
    rows = []
    dataset_id = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    for candidate in candidates:
        base_id = candidate["image_id"]
        split = FIXED_BASE_SPLIT[base_id]
        input_file = utils.resolve_collection_path(candidate["current_path"])
        clean_true, native_background_record = _prepare_native_reference(
            input_file, output_root, base_id, center_crop_size
        )
        star_reference = build_star_reference(clean_true)

        for background_type in BACKGROUND_TYPES:
            geometry_seed = _stable_seed(seed, base_id, background_type, "geometry")
            for intensity_level in INTENSITY_LEVELS:
                sample_id = f"{base_id}_{background_type}_{intensity_level}"
                row = save_sample(
                        input_file=input_file,
                        output_root=output_root,
                        sample_id=sample_id,
                        base_id=base_id,
                        split=split,
                        background_type=background_type,
                        intensity_level=intensity_level,
                        geometry_seed=geometry_seed,
                        clean_true=clean_true,
                        star_reference=star_reference,
                        native_background_record=native_background_record,
                        center_crop_size=center_crop_size,
                        )
                row["dataset_id"] = dataset_id
                rows.append(row)

    expected_samples = len(FIXED_BASE_SPLIT) * len(BACKGROUND_TYPES) * len(INTENSITY_LEVELS)
    if len(rows) != expected_samples:
        raise AssertionError(f"expected {expected_samples} samples, generated {len(rows)}")

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (output_root / "base_split.json").write_text(
        json.dumps(FIXED_BASE_SPLIT, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_root / "dataset_protocol.json").write_text(
        json.dumps(
            {
                "protocol_version": int(SYNTHETIC_DATA_CONFIG["protocol_version"]),
                "protocol_name": "native_flattened_camera_reference_plus_synthetic_skyglow_no_extra_noise",
                "background_types": list(BACKGROUND_TYPES),
                "intensity_levels": INTENSITY_LEVELS,
                "base_split": FIXED_BASE_SPLIT,
                "observation_model": {
                    "formula": "observed = clean_true + background_true",
                    "clean_true_semantics": SYNTHETIC_DATA_CONFIG["observation_model"][
                        "clean_true_semantics"
                    ],
                    "background_true_semantics": "synthetically added skyglow component",
                    "additional_synthetic_noise": False,
                    "skyglow_induced_shot_noise_simulated": False,
                    "shot_noise_limitation": (
                        "No calibrated photon/electron rate is available after normalized "
                        "camera loading, so a physically parameterized skyglow Poisson term "
                        "is intentionally not invented in the main benchmark."
                    ),
                    "low_high_semantics": "skyglow intensity/scale levels, not noise levels",
                    "task_scope": (
                        "background estimation and light-pollution removal, not denoising"
                    ),
                    "scientific_clipping": False,
                    "multiplicative_rescaling_after_flattening": False,
                },
                "star_reference": {
                    "threshold": STAR_THRESHOLD,
                    "fwhm": STAR_FWHM,
                    "matching_radius": MATCHING_RADIUS,
                    "evaluation_schema_version": metrics.EVALUATION_SCHEMA_VERSION,
                    "reference_detection_mode": metrics.REFERENCE_DETECTION_MODE,
                    "debug_candidate_prediction_modes": list(metrics.DETECTION_MODES),
                },
                "native_background_preprocessing": SYNTHETIC_DATA_CONFIG["native_background"],
                "numeric_clipping": False,
                "seed": int(seed),
                "seed_policy": {
                    "geometry": "hash(seed, base_id, background_type, geometry)",
                    "low_high_share_geometry": True
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path, rows


def _stable_seed(*parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)


def _parse_args():
    parser = argparse.ArgumentParser(description="Build the formal synthetic dataset")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=utils.SYNTHETIC_ROOT,
        help="dataset output directory (default: collections/synthetic_formal)",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--center-crop-size", type=int, default=0)
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete an existing non-empty synthetic dataset root before rebuilding",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    manifest_path, rows = build_dataset(
        output_root=args.output_root,
        seed=args.seed,
        center_crop_size=args.center_crop_size,
        force=args.force,
    )
    print(f"generated: {manifest_path}")
    print(f"number of samples: {len(rows)}")


if __name__ == "__main__":
    main()
