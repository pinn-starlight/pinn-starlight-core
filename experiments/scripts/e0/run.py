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
import argparse
from pathlib import Path

import torch
from matplotlib import pyplot as plt

import experiments.common.baselines.coordinate_pinn as pinn
import experiments.common.baselines.fft_gaussian as fft
import experiments.common.baselines.unet_small as unet
import experiments.common.utils.experiment_utils as utils

STEP = 1_000
BATCH_SIZE = 256
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
    args = _parse_args()
    output_root = utils.prepare_output_root(args.output_root, force=args.force)
    device = _device()
    test_images = _test_images(Path(args.test_image_dir))

    print("Hello PINN-Starlight-core!")
    _fft_test(output_root, test_images, args.downsample, args.fft_sigma)
    _pinn_test(
        output_root,
        test_images,
        device,
        args.downsample,
        args.pinn_steps,
        args.pinn_batch_size,
    )
    _unet_test(
        output_root,
        test_images,
        Path(args.synthetic_dir),
        device,
        args.downsample,
        args.unet_epochs,
        args.unet_steps_per_epoch,
        args.unet_batch_size,
        args.unet_patch_size,
        args.unet_base_channels,
    )


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pinn_test(output_root, test_images, device, downsample, steps, batch_size):
    for img in test_images:
        img_name = _sample_name(img)
        observed = _load_e0_observed(img, downsample)
        observed, predicted, residual = pinn.single_train(
            observed,
            device,
            steps,
            batch_size,
        )
        _save_images(output_root, "coordinate_pinn", img_name, observed, predicted, residual)

    print("PINN Done!")


def _fft_test(output_root, test_images, downsample, sigma):
    for img in test_images:
        img_name = _sample_name(img)
        observed = _load_e0_observed(img, downsample)
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
        observed = _load_e0_observed(img, downsample)
        predicted = unet.predict(
            checkpoint,
            observed,
            device=device,
            tile_size=patch_size,
        )
        residual = observed - predicted
        _save_images(output_root, "unet_small", img_name, observed, predicted, residual)

    print("U-Net-small Done!")


def _synthetic_pairs(synthetic_dir: Path = SYNTHETIC_DIR) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
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


def _test_images(test_image_dir: Path) -> list[Path]:
    images = sorted(test_image_dir.glob("*.tif"))
    if not images:
        raise FileNotFoundError(f"No TIFF test images found in: {test_image_dir}")
    return images


def _load_e0_observed(path: Path, downsample: int = E0_DOWNSAMPLE):
    """Use one preprocessing path and resolution for all E0 methods."""
    return utils.load_gray_image(path, downsample=downsample)


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
    display_residual = residual.clip(0, 1)

    plt.imsave(
        method_output / f"observed_{image_name}.png",
        observed,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    plt.imsave(
        method_output / f"residual_{image_name}.png",
        display_residual,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )
    plt.imsave(
        method_output / f"predicted_{image_name}.png",
        predicted,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="E0：三种方法的流程检查，不产生论文结果")
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument(
        "--force",
        action="store_true",
        help="清空指定的 experiments/outputs 子目录后重跑",
    )
    parser.add_argument("--test-image-dir", default=str(TEST_IMG_DIR))
    parser.add_argument("--synthetic-dir", default=str(SYNTHETIC_DIR))
    parser.add_argument("--downsample", type=int, default=E0_DOWNSAMPLE)
    parser.add_argument("--fft-sigma", type=float, default=FFT_SIGMA)
    parser.add_argument("--pinn-steps", type=int, default=STEP)
    parser.add_argument("--pinn-batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--unet-epochs", type=int, default=UNET_EPOCHS)
    parser.add_argument("--unet-steps-per-epoch", type=int, default=UNET_STEPS_PER_EPOCH)
    parser.add_argument("--unet-batch-size", type=int, default=UNET_BATCH_SIZE)
    parser.add_argument("--unet-patch-size", type=int, default=UNET_PATCH_SIZE)
    parser.add_argument("--unet-base-channels", type=int, default=UNET_BASE_CHANNELS)
    return parser.parse_args()


if __name__ == "__main__":
    main()
