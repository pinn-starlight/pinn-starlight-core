"""E0、E2、E4 共用的 FFT-Gaussian 背景估计 baseline。

约定：
- 输入和输出均为 [0, 1] 线性灰度浮点数组。
- 先做 reflection padding，频域高斯低通后裁回原尺寸。
- 截止尺度只能在验证集从 0.02/0.04/0.08/0.16 中选择一次。
- 测试集和真实图使用同一个锁定值。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from experiments.common.utils import experiment_utils as utils

SIGMA_CANDIDATES = (0.02, 0.04, 0.08, 0.16)


def estimate_background(observed, normalized_sigma: float):
    """返回 background_pred，不在这里裁剪 residual_pred。"""
    observed = np.asarray(observed, dtype=np.float32)
    if observed.ndim != 2:
        raise ValueError(f"observed 必须是二维灰度数组，实际形状为 {observed.shape}")
    if not np.isfinite(observed).all():
        raise ValueError("observed 包含 NaN 或 Inf")
    if not 0.0 < normalized_sigma <= 0.5:
        raise ValueError("normalized_sigma 必须位于 (0, 0.5] 范围内")

    spatial_sigma = 1.0 / (2.0 * np.pi * normalized_sigma)
    pad = max(4, int(np.ceil(3.0 * spatial_sigma)))
    pad_y = min(pad, observed.shape[0] - 1)
    pad_x = min(pad, observed.shape[1] - 1)
    padded = np.pad(observed, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")

    frequency_y = np.fft.fftfreq(padded.shape[0])[:, None]
    frequency_x = np.fft.rfftfreq(padded.shape[1])[None, :]
    radius_squared = frequency_x**2 + frequency_y**2
    transfer = np.exp(-0.5 * radius_squared / normalized_sigma**2)

    spectrum = np.fft.rfft2(padded)
    filtered = np.fft.irfft2(spectrum * transfer, s=padded.shape)
    background = filtered[
        pad_y : pad_y + observed.shape[0],
        pad_x : pad_x + observed.shape[1],
    ]
    return np.asarray(background, dtype=np.float32)


def single_estimate(input_path, normalized_sigma: float, downsample: int = 2):
    """读取单张图片并返回 observed、background_pred 和 residual_pred。"""
    # Keep E0 on the same preprocessing path as PINN and U-Net.
    observed = utils.load_gray_image(input_path, downsample=downsample)
    predicted = estimate_background(observed, normalized_sigma)
    residual = observed - predicted
    return observed, predicted, residual


def select_sigma(
    validation_samples: Iterable,
    candidates=SIGMA_CANDIDATES,
) -> float:
    """仅根据验证集背景指标选择截止尺度。"""
    samples = list(validation_samples)
    if not samples:
        raise ValueError("validation_samples 不能为空")
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("candidates 不能为空")

    scores = {}
    for candidate in candidates:
        sample_losses = []
        for sample in samples:
            if isinstance(sample, dict):
                observed = sample["observed"]
                background_true = sample["background_true"]
            else:
                observed, background_true = sample

            observed = np.asarray(observed, dtype=np.float32)
            background_true = np.asarray(background_true, dtype=np.float32)
            if observed.shape != background_true.shape:
                raise ValueError("observed 与 background_true 的尺寸不一致")

            prediction = estimate_background(observed, float(candidate))
            sample_losses.append(float(np.mean((prediction - background_true) ** 2)))

        scores[float(candidate)] = float(np.mean(sample_losses))

    return min(scores, key=lambda candidate: scores[candidate])
