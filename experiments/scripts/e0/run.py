"""E0：先试试三种方法能不能跑，不进入论文结果。

目标：用两张合成样本和 native_test.tif，让 FFT-Gaussian、
U-Net-small 和 PINN 都完成一次最短运行。这里只检查整条流程是否可用，
不比较方法效果，也不把数值写入论文结果。

最简单的通过条件：
- 输入、background_pred、residual_pred 尺寸和数值范围正确；
- loss、指标和参数无 NaN/Inf；
- 至少能打印 loss，并保存一张背景图和一张残差图；
- 当前 PINN 加载器接口已修正，正式实验 alpha 固定为 0.5。
"""
from pathlib import Path

import numpy as np
import torch

import experiments.common.baselines.coordinate_pinn as pinn
import experiments.common.baselines.fft_gaussian as fft
import experiments.common.baselines.unet_small as unet
import experiments.common.utils.experiment_utils as utils
import experiments.common.utils.metrics as metrics
from experiments.common.baselines.coordinate_pinn import DEFAULT_CONFIG

FORCE_OVERWRITE_OUTPUT=False
PINN_STEP = 1_000
PINN_BATCH_SIZE = 256
FFT_SIGMA = 0.08
UNET_EPOCHS = 1
UNET_STEPS_PER_EPOCH = 10
UNET_BATCH_SIZE = 1
UNET_PATCH_SIZE = 256
UNET_BASE_CHANNELS = 8
E0_DOWNSAMPLE = 2
TEST_IMG_DIR = utils.PROJECT_ROOT / "data/test"
SYNTHETIC_DIR = utils.PROJECT_ROOT / "data/collections/synthetic"
OUTPUT_ROOT = utils.OUTPUT_ROOT / "e0"


def main():
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=FORCE_OVERWRITE_OUTPUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device.type}")
    test_images = _test_images(Path(TEST_IMG_DIR))
    _metrics_test()

    print("Hello PINN-Starlight-core!")
    _fft_test(output_root, test_images, E0_DOWNSAMPLE, FFT_SIGMA)
    _pinn_test(
        output_root,
        test_images,
        device,
        E0_DOWNSAMPLE,
        PINN_STEP,
        PINN_BATCH_SIZE,
    )
    _unet_test(
        output_root,
        test_images,
        Path(SYNTHETIC_DIR),
        device,
        E0_DOWNSAMPLE,
        UNET_EPOCHS,
        UNET_STEPS_PER_EPOCH,
        UNET_BATCH_SIZE,
        UNET_PATCH_SIZE,
        UNET_BASE_CHANNELS,
    )


def _pinn_test(output_root, test_images, device, downsample, steps, batch_size):
    config = DEFAULT_CONFIG.copy()
    config["steps"] = steps
    config["batch_size"] = batch_size

    for img in test_images:
        img_name = _sample_name(img)
        observed = utils.load_gray_image(img, downsample=downsample)
        result = pinn.train_background(
            observed,
            config=config,
            device=device,
        )
        predicted = result["background_pred"]
        residual = result["residual_pred"]
        _save_images(output_root, "coordinate_pinn", img_name, observed, predicted, residual)

    print("PINN Done!")


def _fft_test(output_root, test_images, downsample, sigma):
    for img in test_images:
        img_name = _sample_name(img)
        observed = utils.load_gray_image(img, downsample=downsample)
        predicted = fft.estimate_background(observed, sigma)
        residual = observed - predicted
        _save_images(output_root, "fft_gaussian", img_name, observed, predicted, residual)

    print("FFT-Gaussian Done!")


def _unet_test(
    output_root,
    test_images,
    synthetic_dir,
    device,
    downsample,
    epochs,
    steps_per_epoch,
    batch_size,
    patch_size,
    base_channels,
):
    pairs = _synthetic_pairs(synthetic_dir)
    if not pairs:
        raise FileNotFoundError(f"No synthetic samples found in: {synthetic_dir}")

    train_pairs = [pairs[0]]
    validation_pairs = [pairs[1] if len(pairs) > 1 else pairs[0]]
    method_output = output_root / "unet_small"
    checkpoint = unet.train(
        train_pairs,
        validation_pairs,
        method_output / "checkpoint",
        device=device,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        patch_size=patch_size,
        base_channels=base_channels,
        patience=1,
    )

    for img in test_images:
        img_name = _sample_name(img)
        observed = utils.load_gray_image(img, downsample=downsample)
        predicted = unet.predict(
            checkpoint,
            observed,
            device=device,
            tile_size=patch_size,
        )
        residual = observed - predicted
        _save_images(output_root, "unet_small", img_name, observed, predicted, residual)

    print("U-Net-small Done!")


def _metrics_test():
    """用完美预测检查合成数据评估器和输入形状。"""
    sample = _first_synthetic_sample()
    clean_true = sample["clean_true"]
    background_true = sample["background_true"]
    observed = sample["observed"]

    for name, image in {
        "clean_true": clean_true,
        "background_true": background_true,
        "observed": observed,
    }.items():
        if not isinstance(image, np.ndarray) or image.ndim != 2:
            raise ValueError(
                f"评估器输入 {name} 必须是二维 NumPy 数组，"
                f"实际为 {type(image).__name__}, {getattr(image, 'shape', None)}"
            )
        if not np.isfinite(image).all():
            raise ValueError(f"评估器输入 {name} 包含 NaN 或 Inf")

    # 用当前检测器生成参考星点，避免旧 metadata 的检测算法影响自检。
    reference_stars = metrics.extract_stars(clean_true, threshold=0.03)
    test_sample = {
        **sample,
        "metadata": {
            "star_reference": {
                "threshold": 0.3,
                "matching_radius": 3,
                "stars": reference_stars,
            }
        },
    }
    perfect_prediction = {
        "background_pred": background_true.copy(),
        "residual_pred": clean_true.copy(),
    }
    scores = metrics.evaluate_synthetic(test_sample, perfect_prediction)

    if scores["bg_mae"] > 1e-6 or scores["residual_mae"] > 1e-6:
        raise AssertionError(f"完美预测的 MAE 不为 0：{scores}")
    if not all(np.isfinite(value) for value in scores.values()):
        raise AssertionError(f"评估器输出包含 NaN 或 Inf：{scores}")

    print(
        "Metrics test passed: "
        f"shape={clean_true.shape}, "
        f"bg_mae={scores['bg_mae']:.2e}, "
        f"residual_mae={scores['residual_mae']:.2e}, "
        f"star_f1={scores['star_f1']:.3f},"
        f"star_counts={scores['star_counts']}"
    )


def _first_synthetic_sample():
    """直接从合成样本目录读取一张样本，不依赖 manifest。"""
    sample_dirs: list[Path] = list(SYNTHETIC_DIR.iterdir())
    for sample_dir in sorted(sample_dirs):
        if not sample_dir.is_dir():
            continue

        clean_files = list(sample_dir.glob("clean_true_*.tif"))
        background_files = list(sample_dir.glob("background_true_*.tif"))
        observed_files = list(sample_dir.glob("observed_*.tif"))
        if len(clean_files) != 1 or len(background_files) != 1 or len(observed_files) != 1:
            continue

        return {
            "clean_true": utils.load_gray_image(clean_files[0]),
            "background_true": utils.load_gray_image(background_files[0]),
            "observed": utils.load_gray_image(observed_files[0]),
        }

    raise FileNotFoundError(
        f"合成数据目录中没有完整样本：{SYNTHETIC_DIR}"
    )


def _synthetic_pairs(synthetic_dir: Path = SYNTHETIC_DIR):
    pairs = []
    if not synthetic_dir.is_dir():
        return pairs

    for sample_dir in sorted(synthetic_dir.iterdir()):
        if not sample_dir.is_dir():
            continue

        observed_files = list(sample_dir.glob("observed_*.tif"))
        background_files = list(sample_dir.glob("background_true_*.tif"))
        if len(observed_files) == 1 and len(background_files) == 1:
            pairs.append((observed_files[0], background_files[0]))

    return pairs


def _test_images(test_image_dir: Path):
    images = sorted(test_image_dir.glob("*.tif"))
    if not images:
        raise FileNotFoundError(f"No TIFF test images found in: {test_image_dir}")
    return images


def _sample_name(path: Path):
    return path.stem.removeprefix("observed_")


def _save_images(output_root, method, image_name, observed, predicted, residual):
    if observed.shape != predicted.shape or observed.shape != residual.shape:
        raise ValueError(
            f"{method} 输出尺寸不一致：observed={observed.shape}, "
            f"background={predicted.shape}, residual={residual.shape}"
        )
    method_output = Path(output_root) / method
    method_output.mkdir(exist_ok=True, parents=True)
    utils.save_display_png(
        method_output / f"observed_{image_name}.png",
        observed
    )
    utils.save_display_png(
        method_output / f"predicted_{image_name}.png",
        predicted
    )
    utils.save_display_png(
        method_output / f"residual_{image_name}.png",
        residual
    )



if __name__ == "__main__":
    main()
