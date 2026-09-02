"""
合成光污染的主文件
"""

import csv
import json
from pathlib import Path
from experiments.common.data.exponential_background import (
    generate_exponential_background,
)
from experiments.common.data.elliptical_background import (
    generate_elliptical_background,
)
import numpy as np

from experiments.common.utils import experiment_utils as utils
from pinn_starlight_core.data.image_loader import ImageLoader

RAW_DOWNSAMPLE = 2
BACKGROUND_TYPES = (
    "exponential",
    "elliptical",
)


def load_clean_image(input_file, center_crop_size=0):
    """读取图片"""
    loader = ImageLoader(input_file, device="cpu", downsample=RAW_DOWNSAMPLE)
    _, brightness, width, height = loader.get_gray_data()
    if hasattr(brightness, "detach"):
        brightness = brightness.detach().cpu().numpy()
    image = np.asarray(brightness, dtype=np.float32).reshape(height, width)
    return center_crop(image, center_crop_size)


def center_crop(image, crop_size=0):
    """裁剪图片，一般用不到"""
    crop_size = int(crop_size)
    if crop_size == 0:
        return image
    if crop_size < 0 or crop_size > min(image.shape):
        raise ValueError(f"invalid center_crop_size: {crop_size}")
    height, width = image.shape
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[top : top + crop_size, left : left + crop_size].copy()


def synthesize_from_clean(clean_true, background_type, seed):
    H, W = clean_true.shape
    if background_type == "exponential":
        background_true = generate_exponential_background(
            H, W, seed
        )
    elif background_type == "elliptical":
        background_true = generate_elliptical_background(
            H, W, seed
        )
    else:
        raise ValueError(f"invalid background_type: {background_type}")

    observed = np.clip(clean_true + background_true, 0, 1)
    background_param = {
        "model": background_type,
        "seed": int(seed)
    }
    return clean_true, background_true, observed, background_param


def save_sample(
    input_file,
    output_root,
    sample_id,
    base_id,
    split,
    background_type,
    seed,
    clean_true,
    center_crop_size=0,
):
    """保存合成后的数据并输出metadata"""
    clean_true, background_true, observed, background_params = synthesize_from_clean(
        clean_true, background_type, seed
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
        utils.save_display_png(sample_dir / f"{name}_{sample_id}.png", image)
        paths[name] = utils.project_relative(tiff_path)

    metadata = {
        "sample_id": sample_id,
        "base_id": base_id,
        "source_file": utils.project_relative(input_file),
        "split": split,
        "seed": int(seed),
        "background_type": background_type,
        "background_params": background_params,
        "preprocessing": {
            "raw_loader_downsample": RAW_DOWNSAMPLE,
            "center_crop_size": int(center_crop_size),
        },
        "star_reference": {"status": "TODO"},
    }
    metadata_path = sample_dir / f"metadata_{sample_id}.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "sample_id": sample_id,
        "base_id": base_id,
        "split": split,
        "background_type": background_type,
        "seed": int(seed),
        **paths,
        "metadata": utils.project_relative(metadata_path),
    }


def _clean_candidates(manifest_path):
    """读取csv并在csv内寻找标记clean的星图"""
    with Path(manifest_path).open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    candidates = [
        row for row in rows if row.get("usage") == "synthetic_base_candidate"
    ]
    return sorted(candidates, key=lambda row: row["image_id"])


def _assign_splits(candidates, seed):
    """打乱图片顺序"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))
    shuffled = [candidates[index] for index in order]
    train_count = max(1, round(len(shuffled) * 0.6))
    validation_count = max(1, round(len(shuffled) * 0.2))
    if train_count + validation_count >= len(shuffled):
        train_count, validation_count = len(shuffled) - 2, 1

    splits = {}
    for index, row in enumerate(shuffled):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "validation"
        else:
            split = "test"
        splits[row["image_id"]] = split
    return shuffled, splits


def build_dataset(
    collection_manifest=utils.PROJECT_ROOT / "data/collections/manifest.csv",
    output_root=utils.SYNTHETIC_ROOT,
    seed=20260728,
    center_crop_size=0,
):
    """主调度"""
    candidates = _clean_candidates(collection_manifest)
    if len(candidates) < 5:
        raise ValueError("训练时最少5张照片")

    candidates, splits = _assign_splits(candidates, seed)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    sample_number = 0

    for candidate in candidates:
        base_id = candidate["image_id"]
        input_file = utils.to_absolute_path(
            Path("data/collections") / candidate["current_path"]
        )
        clean_true = load_clean_image(input_file, center_crop_size)
        for background_type in BACKGROUND_TYPES:
            sample_id = f"{base_id}_{background_type}"
            rows.append(
                save_sample(
                    input_file,
                    output_root,
                    sample_id,
                    base_id,
                    splits[base_id],
                    background_type,
                    seed + sample_number,
                    clean_true,
                    center_crop_size,
                )
            )
            sample_number += 1

    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "base_split.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path, rows


def main():
    manifest_path, rows = build_dataset()

    print(f"generated: {manifest_path}")
    print(f"number of samples: {len(rows)}")


if __name__ == "__main__":
    main()
