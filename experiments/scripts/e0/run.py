"""E0：四方法冒烟测试，不进入论文结果。

目标：在正式 selection 前，用最小预算确认 FFT-Gaussian、Fourier-LS、
U-Net-small 和 PINN 的输入 -> background -> common residual -> 保存链路都能跑通。
这里只检查运行时、shape、finite 值和 I/O，不比较方法效果，也不把数值写入论文结果。

Fourier-LS 的 E0 分支故意使用 smoke-only 的小 Powell/BCD 预算；正式 selection / E2 / E4
仍只使用 ``formal_experiment.py`` 中冻结的完整参数。E0 不参与任何 selection artifact
或 Global Configuration Lock。
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import experiments.common.baselines.coordinate_pinn as pinn
import experiments.common.baselines.fft_gaussian as fft
import experiments.common.baselines.fft_screened_poisson as fourier_ls
import experiments.common.baselines.unet_small as unet
import experiments.common.utils.experiment_utils as utils
import experiments.common.utils.metrics as metrics
from experiments.common.baselines.coordinate_pinn import DEFAULT_CONFIG

PINN_STEP = 1_000
PINN_BATCH_SIZE = 256
FFT_SIGMA = 0.08
# E0-only Fourier-LS smoke settings. These are runtime test budgets, not formal
# experiment choices, and therefore must never be copied into formal_experiment.py.
FOURIER_LS_LAMBDA = 0.1
FOURIER_LS_SPARSE_WEIGHT = 1e-3
FOURIER_LS_DOWNSAMPLE = 8
FOURIER_LS_MAX_SWEEPS = 1
FOURIER_LS_COMPONENT_MAXITER = 2
UNET_EPOCHS = 1
UNET_STEPS_PER_EPOCH = 10
UNET_BATCH_SIZE = 1
UNET_PATCH_SIZE = 256
UNET_BASE_CHANNELS = 8
E0_DOWNSAMPLE = 2
TEST_IMG_DIR = utils.TEST_ROOT
SYNTHETIC_DIR = utils.SYNTHETIC_ROOT
OUTPUT_ROOT = utils.OUTPUT_ROOT / "e0"


def main(*, force: bool = False):
    print("Hello PINN-Starlight-core!")
    print("E0")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device:{device.type}")
    test_images = _test_images(Path(TEST_IMG_DIR))
    _metrics_test()

    _fft_test(output_root, test_images, E0_DOWNSAMPLE, FFT_SIGMA)
    _fourier_ls_test(
        output_root,
        test_images,
        FOURIER_LS_DOWNSAMPLE,
        FOURIER_LS_LAMBDA,
    )
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



def _fourier_ls_test(output_root, test_images, downsample, lambda_ls):
    """Run a fast, non-formal Stage-10 Fourier-LS smoke path."""
    for img in test_images:
        img_name = _sample_name(img)
        observed = utils.load_gray_image(img, downsample=downsample)
        predicted, info = fourier_ls._estimate_background_with_settings(
            observed,
            float(lambda_ls),
            FOURIER_LS_SPARSE_WEIGHT,
            alpha=fourier_ls.ALPHA,
            max_components=fourier_ls.MAX_COMPONENTS,
            max_sweeps=FOURIER_LS_MAX_SWEEPS,
            component_maxiter=FOURIER_LS_COMPONENT_MAXITER,
            xtol=fourier_ls.POWELL_XTOL,
            ftol=fourier_ls.POWELL_FTOL,
            return_info=True,
            show_progress=True,
            progress_desc=f"E0 Fourier-LS {img_name}",
        )
        residual = utils.reconstruct_residual(observed, predicted)
        _save_images(output_root, "fourier_ls", img_name, observed, predicted, residual)
        utils.write_json(
            Path(output_root) / "fourier_ls" / f"estimated_parameters_{img_name}.json",
            {
                **info,
                "smoke_only": True,
                "note": (
                    "E0 runtime smoke budget only; not a formal hyperparameter selection "
                    "and not part of the Global Configuration Lock."
                ),
            },
        )

    print("Fourier-LS Done! (E0 smoke-only Stage-10 profile/BCD budget)")



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
            show_progress=True
        )
        predicted = result["background_pred"]
        residual = utils.reconstruct_residual(observed, predicted)
        _save_images(output_root, "coordinate_pinn", img_name, observed, predicted, residual)

    print("PINN Done!")


def _fft_test(output_root, test_images, downsample, sigma):
    for img in test_images:
        img_name = _sample_name(img)
        observed = utils.load_gray_image(img, downsample=downsample)
        predicted = fft.estimate_background(observed, sigma)
        residual = utils.reconstruct_residual(observed, predicted)
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
        learning_rate=1e-3,
        base_channels=base_channels,
        patience=1,
        min_delta_relative=0.002,
        use_amp=True,
        inference_mode="auto",
        inference_tile_size=patch_size,
        inference_overlap=min(64, patch_size // 4),
        seed=int(utils.SEEDS[0]),
    )

    for img in test_images:
        img_name = _sample_name(img)
        observed = utils.load_gray_image(img, downsample=downsample)
        predicted = unet.predict(
            checkpoint,
            observed,
            device=device,
            inference_mode="auto",
            tile_size=patch_size,
            overlap=min(64, patch_size // 4),
        )
        residual = utils.reconstruct_residual(observed, predicted)
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

    # Oracle background checks the evaluator without rebuilding the frozen star
    # catalogue.  Because the formal observation contains noise, the oracle
    # residual is clean_true + noise rather than exactly clean_true.
    oracle_prediction = {
        "background_pred": background_true.copy(),
        "residual_pred": utils.reconstruct_residual(observed, background_true),
    }
    scores = metrics.evaluate_synthetic(sample, oracle_prediction)

    if scores["bg_mae"] > 1e-6 or scores["bg_rmse"] > 1e-6:
        raise AssertionError(f"oracle background error is not zero: {scores}")
    # 普通数值指标必须是有限值。
    for key, value in scores.items():
        if key in {"bg_psnr", "residual_psnr"}:
            continue

        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(value):
                raise AssertionError(
                f"评估器关键输出包含 NaN 或 Inf: {key}={value}, scores={scores}"
            )

    # PSNR 的特殊情况：
    # 完美预测时 RMSE=0，因此 PSNR=+inf 是数学上正确的结果。
    for psnr_key, rmse_key in (
        ("bg_psnr", "bg_rmse"),
        ("residual_psnr", "residual_rmse"),
    ):
        psnr_value = float(scores[psnr_key])
        rmse_value = float(scores[rmse_key])

        if np.isnan(psnr_value) or np.isneginf(psnr_value):
            raise AssertionError(
                f"{psnr_key} 非法: {psnr_value}, scores={scores}"
            )

        if np.isposinf(psnr_value) and rmse_value > 1e-12:
            raise AssertionError(
                f"{psnr_key}=+inf，但 {rmse_key}={rmse_value} 并非 0"
            )

    if scores["matched_star_flux_count"] > 0 and not np.isfinite(
        scores["matched_star_mean_relative_flux_error"]
    ):
        raise AssertionError(f"存在成功光度匹配时通量指标必须有限：{scores}")

    print(
        "Metrics test passed: "
        f"shape={clean_true.shape}, "
        f"bg_mae={scores['bg_mae']:.2e}, "
        f"residual_mae={scores['residual_mae']:.2e}, "
        f"star_f1={scores['star_f1']:.3f}, "
        f"matched_stars={scores['matched_star_count']}"
    )


def _first_synthetic_sample():
    """Load one formal synthetic sample through the manifest and float loader."""
    rows = utils.load_manifest(utils.SYNTHETIC_MANIFEST)
    if not rows:
        raise ValueError("合成数据 manifest 为空")
    return utils.load_synthetic_sample(rows[0])


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
    arrays = {
        "observed": np.asarray(observed),
        "background_pred": np.asarray(predicted),
        "residual_pred": np.asarray(residual),
    }
    if any(array.ndim != 2 for array in arrays.values()):
        raise ValueError(
            f"{method} E0 输出必须都是二维数组："
            + ", ".join(f"{name}={array.shape}" for name, array in arrays.items())
        )
    if observed.shape != predicted.shape or observed.shape != residual.shape:
        raise ValueError(
            f"{method} 输出尺寸不一致：observed={observed.shape}, "
            f"background={predicted.shape}, residual={residual.shape}"
        )
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise FloatingPointError(f"{method} {name} 包含 NaN 或 Inf")

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



def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only the E0 smoke-test output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(force=args.force)
