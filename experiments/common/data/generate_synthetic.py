"""生成正式合成数据和固定 manifest。

observed = clean_true + background_true + noise
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.common.utils import experiment_utils as utils
from experiments.common.utils.metrics import extract_stars
from pinn_starlight_core.data.image_loader import ImageLoader

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
    """生成受 Kocifaj 2025 启发的二维清空背景近似。

    这不是论文三维辐射传输与随机云模型的数值复现，只保留有限光源、
    双程大气透射、Henyey-Greenstein 前向散射和几何衰减。
    """
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    dx = xx - source_x
    dy = yy - source_y
    theta = np.deg2rad(theta_degrees)
    dx_rot = np.cos(theta) * dx + np.sin(theta) * dy
    dy_rot = -np.sin(theta) * dx + np.cos(theta) * dy
    source_profile = np.exp(
        -0.5 * ((dx_rot / sigma_x) ** 2 + (dy_rot / sigma_y) ** 2)
    )

    horizontal_distance = np.sqrt(dx**2 + dy**2)
    source_to_scatter = np.sqrt(horizontal_distance**2 + scattering_height**2)
    transmission = np.exp(
        -aerosol_optical_depth * (source_to_scatter + scattering_height)
    )
    cos_angle = horizontal_distance / np.maximum(source_to_scatter, 1e-6)
    phase = (1.0 - asymmetry_g**2) / np.maximum(
        (1.0 + asymmetry_g**2 - 2.0 * asymmetry_g * cos_angle) ** 1.5,
        1e-6,
    )
    spreading = np.maximum(source_to_scatter, 1e-6) ** distance_power
    profile = source_profile * transmission * phase / spreading
    profile = profile / max(float(profile.max()), 1e-8)
    return np.clip(ambient + intensity * profile, 0.0, 1.0).astype(np.float32)


def generate_background(height, width, background_type, intensity, scale, rng):
    """返回背景数组和用于复现实验的源项参数。"""
    if background_type == "single_radial":
        center_x = float(rng.uniform(-0.35, 0.35))
        center_y = float(rng.uniform(-0.35, 0.35))
        background = kocifaj_background(
            height,
            width,
            intensity=intensity,
            source_x=center_x,
            source_y=center_y,
            sigma_x=scale,
            sigma_y=scale,
            theta_degrees=0.0,
        )
        parameters = {
            "source_centers": [{"x": center_x, "y": center_y}],
            "sigma_x": scale,
            "sigma_y": scale,
            "theta_degrees": 0.0,
        }
    elif background_type == "single_eccentric":
        center_x = float(rng.uniform(-0.75, 0.75))
        center_y = float(rng.uniform(-0.75, 0.75))
        sigma_x = float(scale * rng.uniform(0.9, 1.2))
        sigma_y = float(scale * rng.uniform(0.45, 0.7))
        theta = float(rng.uniform(-45.0, 45.0))
        background = kocifaj_background(
            height,
            width,
            intensity=intensity,
            source_x=center_x,
            source_y=center_y,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            theta_degrees=theta,
        )
        parameters = {
            "source_centers": [{"x": center_x, "y": center_y}],
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "theta_degrees": theta,
        }
    elif background_type == "multi_source":
        centers = [
            {
                "x": float(rng.uniform(-1.1, -0.25)),
                "y": float(rng.uniform(-0.8, 0.8)),
            },
            {
                "x": float(rng.uniform(0.25, 1.1)),
                "y": float(rng.uniform(-0.8, 0.8)),
            },
        ]
        weights = rng.uniform(0.6, 1.0, size=2)
        background = np.zeros((height, width), dtype=np.float32)
        for center, weight in zip(centers, weights):
            background += float(weight) * kocifaj_background(
                height,
                width,
                intensity=1.0,
                source_x=center["x"],
                source_y=center["y"],
                sigma_x=scale,
                sigma_y=scale * 0.65,
                theta_degrees=float(rng.uniform(-35.0, 35.0)),
            )
        background = background / max(float(background.max()), 1e-8)
        background = np.clip(intensity * background, 0.0, 1.0).astype(np.float32)
        parameters = {
            "source_centers": centers,
            "source_weights": [float(value) for value in weights],
            "sigma_x": scale,
            "sigma_y": scale * 0.65,
        }
    else:
        raise ValueError(f"未知背景类型：{background_type}")

    parameters["intensity"] = float(intensity)
    parameters["scale"] = float(scale)
    return background, parameters


def generate_sample(
    input_file,
    background_type,
    intensity,
    scale,
    noise_std,
    seed,
    center_crop_size=0,
):
    """读取一张干净基础图，并合成一个带真值样本。"""
    clean_true = load_clean_image(input_file, center_crop_size=center_crop_size)
    return synthesize_from_clean(
        clean_true,
        background_type,
        intensity,
        scale,
        noise_std,
        seed,
    )


def load_clean_image(input_file, center_crop_size=0):
    """基础图只读取一次，避免同一 RAW 为六个变体重复解码。"""
    loader = ImageLoader(str(input_file), device="cpu")
    _, brightness, width, height = loader.get_gray_data()
    clean_true = brightness.reshape(height, width).numpy()
    clean_true = np.clip(clean_true, 0.0, 1.0).astype(np.float32)
    return _center_crop(clean_true, center_crop_size)


def _center_crop(image, crop_size):
    """可选固定中心裁剪；默认值 0 保持原始下采样尺寸。"""
    crop_size = int(crop_size)
    if crop_size == 0:
        return image
    if crop_size < 0:
        raise ValueError("center_crop_size 不能小于 0")

    height, width = image.shape
    if crop_size > min(height, width):
        raise ValueError(
            f"center_crop_size={crop_size} 超过图像尺寸 {(height, width)}"
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[top : top + crop_size, left : left + crop_size].copy()


def synthesize_from_clean(
    clean_true,
    background_type,
    intensity,
    scale,
    noise_std,
    seed,
):
    height, width = clean_true.shape
    rng = np.random.default_rng(seed)
    background_true, background_params = generate_background(
        height,
        width,
        background_type,
        intensity,
        scale,
        rng,
    )
    noise = rng.normal(0.0, noise_std, clean_true.shape).astype(np.float32)
    observed = np.clip(clean_true + background_true + noise, 0.0, 1.0)
    return clean_true, background_true, observed.astype(np.float32), background_params


def save_sample(
    input_file,
    output_root,
    sample_id,
    base_id,
    split,
    background_type,
    profile,
    seed,
    clean_true=None,
    stars=None,
    center_crop_size=0,
):
    if clean_true is None:
        clean_true = load_clean_image(
            input_file,
            center_crop_size=center_crop_size,
        )
    clean_true, background_true, observed, background_params = synthesize_from_clean(
        clean_true=clean_true,
        background_type=background_type,
        intensity=profile["intensity"],
        scale=profile["scale"],
        noise_std=profile["noise_std"],
        seed=seed,
    )
    sample_dir = Path(output_root) / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    paths = {}
    for name, image in {
        "clean_true": clean_true,
        "background_true": background_true,
        "observed": observed,
    }.items():
        path = sample_dir / f"{name}_{sample_id}.tif"
        utils.save_float_tiff(path, image)
        paths[name] = utils.project_relative(path)

    if stars is None:
        stars = extract_stars(clean_true, threshold=0.03, matching_radius=3)
    metadata = {
        "sample_id": sample_id,
        "base_id": base_id,
        "source_file": utils.project_relative(input_file),
        "split": split,
        "seed": seed,
        "equation": "observed = clean_true + background_true + noise",
        "background_type": background_type,
        "background_params": background_params,
        "intensity_level": profile["level"],
        "noise_std": profile["noise_std"],
        "preprocessing": {
            "raw_loader_downsample": 2,
            "center_crop_size": int(center_crop_size),
        },
        "star_reference": {
            "method": "local contrast maxima extracted once from clean_true",
            "threshold": 0.03,
            "matching_radius": 3,
            "stars": stars,
        },
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
        "intensity_level": profile["level"],
        "noise_std": profile["noise_std"],
        "seed": seed,
        "clean_true": paths["clean_true"],
        "background_true": paths["background_true"],
        "observed": paths["observed"],
        "metadata": utils.project_relative(metadata_path),
    }


def build_dataset(
    collection_manifest=utils.PROJECT_ROOT / "data/collections/manifest.csv",
    output_root=utils.SYNTHETIC_ROOT,
    seed=20260728,
    center_crop_size=0,
):
    """按基础图固定划分后生成全部正式合成样本。"""
    candidates = _clean_candidates(collection_manifest)
    if len(candidates) < 5:
        raise ValueError("正式 60/20/20 划分至少需要 5 张干净基础图")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates)).tolist()
    candidates = [candidates[index] for index in order]
    split_by_base = {}
    train_count = max(1, round(len(candidates) * 0.6))
    validation_count = max(1, round(len(candidates) * 0.2))
    if train_count + validation_count >= len(candidates):
        train_count = len(candidates) - 2
        validation_count = 1

    for index, candidate in enumerate(candidates):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        split_by_base[candidate["image_id"]] = split

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    sample_number = 0
    for candidate in candidates:
        base_id = candidate["image_id"]
        input_file = utils.resolve_path(
            Path("data/collections") / candidate["current_path"]
        )
        clean_true = load_clean_image(
            input_file,
            center_crop_size=center_crop_size,
        )
        stars = extract_stars(clean_true, threshold=0.03, matching_radius=3)
        for background_type in BACKGROUND_TYPES:
            for profile in PROFILES:
                sample_seed = seed + sample_number
                sample_id = f"{base_id}_{background_type}_{profile['level']}"
                rows.append(
                    save_sample(
                        input_file=input_file,
                        output_root=output_root,
                        sample_id=sample_id,
                        base_id=base_id,
                        split=split_by_base[base_id],
                        background_type=background_type,
                        profile=profile,
                        seed=sample_seed,
                        clean_true=clean_true,
                        stars=stars,
                        center_crop_size=center_crop_size,
                    )
                )
                sample_number += 1

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split_path = output_root / "base_split.json"
    split_path.write_text(
        json.dumps(split_by_base, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path, rows


def _clean_candidates(collection_manifest):
    collection_manifest = Path(collection_manifest)
    with collection_manifest.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    return sorted(
        [row for row in rows if row.get("usage") == "synthetic_base_candidate"],
        key=lambda row: row["image_id"],
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="生成正式合成数据和固定 manifest")
    parser.add_argument(
        "--collection-manifest",
        default=str(utils.PROJECT_ROOT / "data/collections/manifest.csv"),
    )
    parser.add_argument("--output-root", default=str(utils.SYNTHETIC_ROOT))
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--center-crop-size",
        type=int,
        default=0,
        help="下采样后固定中心裁剪边长；0 表示不裁剪。",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    manifest, rows = build_dataset(
        collection_manifest=args.collection_manifest,
        output_root=args.output_root,
        seed=args.seed,
        center_crop_size=args.center_crop_size,
    )
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}
    print(f"生成完成：{manifest}")
    print(f"train={counts['train']} validation={counts['validation']} test={counts['test']}")


if __name__ == "__main__":
    main()
