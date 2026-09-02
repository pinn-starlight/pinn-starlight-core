"""Generate synthetic clean/skyglow/observed image triples.

This is a 2-D toy model inspired by the scattering terms in Kocifaj et al.
It is not a 3-D radiative-transfer implementation.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile as tif

from experiments.common.utils import experiment_utils as utils
from experiments.common.utils.metrics import extract_stars
from pinn_starlight_core.data.image_loader import ImageLoader

RAW_DOWNSAMPLE = 2
BACKGROUND_TYPES = ("single_radial", "single_eccentric", "multi_source")
PROFILES = (
    {"level": "low", "intensity": 0.10, "scale": 0.55, "noise_std": 0.005},
    {"level": "high", "intensity": 0.20, "scale": 0.85, "noise_std": 0.015},
)

def kocifaj_background(
    height,
    width,
    intensity=0.15,
    source_x=1.15,
    source_y=0.20,
    sigma_x=0.85,
    sigma_y=0.45,
    theta_degrees=15.0,
    aerosol_optical_depth=0.20,
    scattering_height=0.35,
    asymmetry_g=0.60,
    distance_power=1.0,
    ambient=0.0,
):
    """Return one normalized 2-D source using a simple scattering approximation."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if sigma_x <= 0 or sigma_y <= 0 or scattering_height <= 0:
        raise ValueError("sigma_x, sigma_y and scattering_height must be positive")
    if not -1.0 < asymmetry_g < 1.0:
        raise ValueError("asymmetry_g must be between -1 and 1")

    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    dx = xx - float(source_x)
    dy = yy - float(source_y)

    theta = np.deg2rad(theta_degrees)
    dx_rot = np.cos(theta) * dx + np.sin(theta) * dy
    dy_rot = -np.sin(theta) * dx + np.cos(theta) * dy
    source_profile = np.exp(
        -0.5 * ((dx_rot / sigma_x) ** 2 + (dy_rot / sigma_y) ** 2)
    )

    horizontal_distance = np.sqrt(dx * dx + dy * dy)
    source_to_scatter = np.sqrt(horizontal_distance**2 + scattering_height**2)
    distance = np.maximum(source_to_scatter, 1e-6)
    transmission = np.exp(-aerosol_optical_depth * (distance + scattering_height))
    cos_angle = horizontal_distance / distance
    phase_denominator = 1.0 + asymmetry_g**2 - 2.0 * asymmetry_g * cos_angle
    phase = (1.0 - asymmetry_g**2) / np.maximum(phase_denominator**1.5, 1e-6)

    profile = source_profile * transmission * phase / distance**distance_power
    profile /= max(float(profile.max()), 1e-8)
    return np.clip(ambient + intensity * profile, 0.0, 1.0).astype(np.float32)


def _source(height, width, center, sigma_x, sigma_y, theta, weight=1.0):
    """Generate one source and keep the parameters needed to reproduce it."""
    image = kocifaj_background(
        height,
        width,
        intensity=float(weight),
        source_x=center["x"],
        source_y=center["y"],
        sigma_x=sigma_x,
        sigma_y=sigma_y,
        theta_degrees=theta,
    )
    params = {
        "center": {"x": float(center["x"]), "y": float(center["y"])},
        "weight": float(weight),
        "sigma_x": float(sigma_x),
        "sigma_y": float(sigma_y),
        "theta_degrees": float(theta),
    }
    return image, params


def generate_background(height, width, background_type, intensity, scale, rng):
    """Generate a background and the parameters used to generate it."""
    if background_type == "single_radial":
        center = {"x": rng.uniform(-0.35, 0.35), "y": rng.uniform(-0.35, 0.35)}
        background, source = _source(height, width, center, scale, scale, 0.0)
        sources = [source]
    elif background_type == "single_eccentric":
        center = {"x": rng.uniform(-0.75, 0.75), "y": rng.uniform(-0.75, 0.75)}
        sigma_x = scale * rng.uniform(0.9, 1.2)
        sigma_y = scale * rng.uniform(0.45, 0.7)
        theta = rng.uniform(-45.0, 45.0)
        background, source = _source(height, width, center, sigma_x, sigma_y, theta)
        sources = [source]
    elif background_type == "multi_source":
        source_specs = []
        for x_range in ((-1.1, -0.25), (0.25, 1.1)):
            center = {"x": rng.uniform(*x_range), "y": rng.uniform(-0.8, 0.8)}
            source_specs.append(
                (center, scale, scale * 0.65, rng.uniform(-35.0, 35.0), rng.uniform(0.6, 1.0))
            )
        background = np.zeros((height, width), dtype=np.float32)
        sources = []
        for center, sigma_x, sigma_y, theta, weight in source_specs:
            source_image, source = _source(
                height, width, center, sigma_x, sigma_y, theta, weight
            )
            background += source_image
            sources.append(source)
        background /= max(float(background.max()), 1e-8)
    else:
        raise ValueError(f"unknown background_type: {background_type}")

    background = np.clip(intensity * background, 0.0, 1.0).astype(np.float32)
    return background, {
        "background_type": background_type,
        "intensity": float(intensity),
        "scale": float(scale),
        "sources": sources,
    }


def load_clean_image(input_file, center_crop_size=0):
    """Load a clean image as a 2-D float32 NumPy array in [0, 1]."""
    loader = ImageLoader(input_file, device="cpu", downsample=RAW_DOWNSAMPLE)
    _, brightness, width, height = loader.get_gray_data()
    if hasattr(brightness, "detach"):
        brightness = brightness.detach().cpu().numpy()
    image = np.asarray(brightness, dtype=np.float32).reshape(height, width)
    image = np.clip(image, 0.0, 1.0)
    return _center_crop(image, center_crop_size)


def save_float_tiff(path, image):
    """Save a 2-D float32 TIFF for metric computation."""
    path = Path(path)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("TIFF image must be a finite 2-D array")
    path.parent.mkdir(parents=True, exist_ok=True)
    tif.imwrite(path, image, dtype=np.float32)


def _center_crop(image, crop_size):
    crop_size = int(crop_size)
    if crop_size == 0:
        return image
    if crop_size < 0 or crop_size > min(image.shape):
        raise ValueError(f"invalid center_crop_size: {crop_size}")
    height, width = image.shape
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[top : top + crop_size, left : left + crop_size].copy()


def synthesize_from_clean(clean_true, background_type, intensity, scale, noise_std, seed):
    """Create background truth and observed image from one clean image."""
    clean_true = np.asarray(clean_true, dtype=np.float32)
    if clean_true.ndim != 2 or not np.isfinite(clean_true).all():
        raise ValueError("clean_true must be a finite 2-D array")
    clean_true = np.clip(clean_true, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    background_true, background_params = generate_background(
        *clean_true.shape, background_type, intensity, scale, rng
    )
    noise = rng.normal(0.0, noise_std, clean_true.shape).astype(np.float32)
    observed = np.clip(clean_true + background_true + noise, 0.0, 1.0)
    return clean_true, background_true, observed.astype(np.float32), background_params


def save_sample(input_file, output_root, sample_id, base_id, split, background_type,
                profile, seed, clean_true, stars=None, center_crop_size=0):
    clean_true, background_true, observed, background_params = synthesize_from_clean(
        clean_true, background_type, profile["intensity"], profile["scale"],
        profile["noise_std"], seed
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
        save_float_tiff(tiff_path, image)
        utils.save_display_png(sample_dir / f"{name}_{sample_id}.png", image)
        paths[name] = utils.project_relative(tiff_path)

    stars = extract_stars(clean_true, threshold=0.03) if stars is None else stars
    metadata = {
        "sample_id": sample_id,
        "base_id": base_id,
        "source_file": utils.project_relative(input_file),
        "split": split,
        "seed": int(seed),
        "equation": "observed = clean_true + background_true + noise",
        "background": background_params,
        "noise_std": float(profile["noise_std"]),
        "preprocessing": {"raw_loader_downsample": RAW_DOWNSAMPLE,
                          "center_crop_size": int(center_crop_size)},
        "star_reference": {"method": "photutils.DAOStarFinder", "threshold": 0.03,
                            "matching_radius": 3, "stars": stars},
    }
    metadata_path = sample_dir / f"metadata_{sample_id}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"sample_id": sample_id, "base_id": base_id, "split": split,
            "background_type": background_type, "intensity_level": profile["level"],
            "noise_std": profile["noise_std"], "seed": seed,
            **paths, "metadata": utils.project_relative(metadata_path)}


def _clean_candidates(manifest_path):
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    candidates = [row for row in rows if row.get("usage") == "synthetic_base_candidate"]
    return sorted(candidates, key=lambda row: row["image_id"])


def build_dataset(collection_manifest=utils.PROJECT_ROOT / "data/collections/manifest.csv",
                  output_root=utils.SYNTHETIC_ROOT, seed=20260728, center_crop_size=0):
    candidates = _clean_candidates(collection_manifest)
    if len(candidates) < 5:
        raise ValueError("at least 5 clean images are required for the 60/20/20 split")

    rng = np.random.default_rng(seed)
    candidates = [candidates[i] for i in rng.permutation(len(candidates))]
    train_count = max(1, round(len(candidates) * 0.6))
    validation_count = max(1, round(len(candidates) * 0.2))
    if train_count + validation_count >= len(candidates):
        train_count, validation_count = len(candidates) - 2, 1

    split_by_id = {}
    for index, row in enumerate(candidates):
        split_by_id[row["image_id"]] = (
            "train" if index < train_count else
            "validation" if index < train_count + validation_count else "test"
        )

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    sample_number = 0
    for candidate in candidates:
        base_id = candidate["image_id"]
        input_file = utils.convert_absol_path(
            Path("data/collections") / candidate["current_path"]
        )
        clean_true = load_clean_image(input_file, center_crop_size)
        stars = extract_stars(clean_true, threshold=0.03)
        for background_type in BACKGROUND_TYPES:
            for profile in PROFILES:
                sample_id = f"{base_id}_{background_type}_{profile['level']}"
                rows.append(save_sample(
                    input_file, output_root, sample_id, base_id, split_by_id[base_id],
                    background_type, profile, seed + sample_number, clean_true, stars,
                    center_crop_size,
                ))
                sample_number += 1

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "base_split.json").write_text(
        json.dumps(split_by_id, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path, rows


def main():
    parser = argparse.ArgumentParser(description="generate synthetic dataset")
    parser.add_argument("--collection-manifest", default=str(utils.PROJECT_ROOT / "data/collections/manifest.csv"))
    parser.add_argument("--output-root", default=str(utils.SYNTHETIC_ROOT))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--center-crop-size", type=int, default=0)
    args = parser.parse_args()
    manifest, rows = build_dataset(args.collection_manifest, args.output_root, args.seed, args.center_crop_size)
    counts = {name: sum(row["split"] == name for row in rows) for name in ("train", "validation", "test")}
    print(f"generated: {manifest}")
    print(f"train={counts['train']} validation={counts['validation']} test={counts['test']}")

if __name__ == "__main__":
    main()
