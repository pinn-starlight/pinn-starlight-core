
"""Scientific image metrics. Inputs are never clipped here."""

import numpy as np
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)


def mae(prediction, target):
    prediction, target = _pair(prediction, target)
    return float(np.mean(np.abs(prediction - target)))


def rmse(prediction, target):
    prediction, target = _pair(prediction, target)
    mse = mean_squared_error(target, prediction)
    return float(np.sqrt(mse))


def psnr(prediction, target, data_range=1.0):
    """PSNR with fixed data_range=1.0 and no clipping."""
    prediction, target = _pair(prediction, target)
    return float(
        peak_signal_noise_ratio(
            target,
            prediction,
            data_range=float(data_range),
        )
    )


def ssim(prediction, target, data_range=1.0):
    """SSIM with data_range fixed to 1.0 and no clipping."""
    prediction, target = _pair(prediction, target)
    return float(
        structural_similarity(
            target,
            prediction,
            data_range=float(data_range),
        )
    )


def _pair(prediction, target):
    prediction = get_image(prediction, clip=False)
    target = get_image(target, clip=False)
    if prediction.shape != target.shape:
        raise ValueError(f"指标输入尺寸不一致：{prediction.shape} 与 {target.shape}")
    return prediction, target


def get_image(image, clip: bool = False):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"指标输入必须是二维灰度图，实际为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("指标输入包含 NaN 或 Inf")
    return np.clip(array, 0.0, 1.0) if clip else array
