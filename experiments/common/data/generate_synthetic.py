"""生成一张带光污染背景的合成图片。

observed = clean_true + background_true + noise
"""

import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from pinn_starlight_core.data.image_loader import ImageLoader


def kocifaj_background(
    height,
    width,
    intensity=0.35,
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
    """生成受 Kocifaj 2025 启发的二维光污染背景。

    这不是论文中三维辐射传输模型的完整复现，只保留光源分布、
    大气衰减、前向散射和距离衰减几个主要因素。
    """
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # 光源的位置和方向
    dx = xx - source_x
    dy = yy - source_y
    theta = np.deg2rad(theta_degrees)
    dx_rot = np.cos(theta) * dx + np.sin(theta) * dy
    dy_rot = -np.sin(theta) * dx + np.cos(theta) * dy

    source_profile = np.exp(
        -0.5 * ((dx_rot / sigma_x) ** 2 + (dy_rot / sigma_y) ** 2)
    )

    # 光经过大气时的衰减
    horizontal_distance = np.sqrt(dx**2 + dy**2)
    source_to_scatter = np.sqrt(horizontal_distance**2 + scattering_height**2)
    transmission = np.exp(
        -aerosol_optical_depth * (source_to_scatter + scattering_height)
    )

    # Henyey-Greenstein 前向散射近似
    cos_angle = horizontal_distance / np.maximum(source_to_scatter, 1e-6)
    phase = (1.0 - asymmetry_g**2) / np.maximum(
        (1.0 + asymmetry_g**2 - 2.0 * asymmetry_g * cos_angle) ** 1.5,
        1e-6,
    )

    distance = np.maximum(source_to_scatter, 1e-6) ** distance_power
    background = source_profile * transmission * phase / distance
    background = background / background.max()
    background = ambient + intensity * background

    return np.clip(background, 0.0, 1.0).astype(np.float32)


def _generate_sample(
    input_file, noise_std=0.0, seed=20260728
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """读取干净图片，再加入背景和噪声。"""
    loader = ImageLoader(input_file)
    _, brightness, width, height = loader.get_gray_data()

    clean_true = brightness.reshape(height, width).detach().cpu().numpy()
    clean_true = np.asarray(np.clip(clean_true, 0.0, 1.0), dtype=np.float32)
    background_true = kocifaj_background(height, width)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, clean_true.shape).astype(np.float32)
    observed = np.add(clean_true, background_true)
    observed = np.add(observed, noise)
    observed = np.asarray(np.clip(observed, 0.0, 1.0), dtype=np.float32)

    return clean_true, background_true, observed


def _save_sample(input_file, output_dir, noise_std=0.0, seed=20260728):
    """把三张图片和生成参数保存下来。"""
    clean_true, background_true, observed = _generate_sample(
        input_file, noise_std, seed
    )
    sample_name = Path(input_file).stem

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_true = (clean_true * 65535).round().astype(np.uint16)
    background_true = (background_true * 65535).round().astype(np.uint16)
    observed = (observed * 65535).round().astype(np.uint16)

    Image.fromarray(clean_true).save(output_dir / f"clean_true_{sample_name}.tif")
    Image.fromarray(background_true).save(
        output_dir / f"background_true_{sample_name}.tif"
    )
    Image.fromarray(observed).save(output_dir / f"observed_{sample_name}.tif")

    metadata = {
        "source_file": str(Path(input_file).resolve()),
        "noise_std": noise_std,
        "seed": seed,
    }
    metadata_file = output_dir / f"metadata_{sample_name}.json"
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    # 在这里换成你的干净图片和输出文件夹。
    input_file = ""
    output_dir = "experiments/output/synthetic"

    if input_file == "":
        print("请先在 main() 中填写 input_file")
        return

    _save_sample(input_file, output_dir, noise_std=0.01)
    print("生成完成：", output_dir)


if __name__ == "__main__":
    main()
