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
TEST_IMG_DIR = utils.PROJECT_ROOT / "data/test"
SYNTHETIC_DIR = utils.PROJECT_ROOT / "data/collections/synthetic"
OUTPUT_ROOT = utils.OUTPUT_ROOT / "e0"
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

def main():
    print("Hello PINN-Starlight-core!")
    _fft_test()
    _pinn_test()
    _unet_test()


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pinn_test():
    for img in TEST_IMG_DIR.glob("*.tif"):
        img_name = _sample_name(img)
        observed, predicted, residual = pinn.single_train(img, _device(), STEP, BATCH_SIZE)
        _save_images("coordinate_pinn", img_name, observed, predicted, residual)

    print("PINN Done!")


def _fft_test():
    for img in TEST_IMG_DIR.glob("*.tif"):
        img_name = _sample_name(img)
        observed, predicted, residual = fft.single_estimate(img, FFT_SIGMA)
        _save_images("fft_gaussian", img_name, observed, predicted, residual)

    print("FFT-Gaussian Done!")


def _unet_test():
    pairs = _synthetic_pairs()
    if not pairs:
        raise FileNotFoundError(f"No synthetic samples found in: {SYNTHETIC_DIR}")

    train_pairs = [pairs[0]]
    validation_pairs = [pairs[1] if len(pairs) > 1 else pairs[0]]
    method_output = OUTPUT_ROOT / "unet_small"
    checkpoint = unet.train(
        train_pairs,
        validation_pairs,
        method_output / "checkpoint",
        device=_device(),
        epochs=UNET_EPOCHS,
        steps_per_epoch=UNET_STEPS_PER_EPOCH,
        batch_size=UNET_BATCH_SIZE,
        patch_size=UNET_PATCH_SIZE,
        base_channels=UNET_BASE_CHANNELS,
        patience=1,
    )

    for img in TEST_IMG_DIR.glob("*.tif"):
        img_name = _sample_name(img)
        observed, predicted, residual = unet.single_predict(
            checkpoint,
            img,
            device=_device(),
            tile_size=UNET_PATCH_SIZE,
        )
        _save_images("unet_small", img_name, observed, predicted, residual)

    print("U-Net-small Done!")


def _synthetic_pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    if not SYNTHETIC_DIR.is_dir():
        return pairs

    for sample_dir in SYNTHETIC_DIR.iterdir():
        if not sample_dir.is_dir():
            continue

        observed_files = list(sample_dir.glob("observed_*.tif"))
        background_files = list(sample_dir.glob("background_true_*.tif"))
        if len(observed_files) == 1 and len(background_files) == 1:
            pairs.append((observed_files[0], background_files[0]))

    return pairs


def _sample_name(path: Path):
    return path.stem.removeprefix("observed_")


def _save_images(method, image_name, observed, predicted, residual):
    method_output = OUTPUT_ROOT / method
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


if __name__ == "__main__":
    main()
