"""专门存放评估函数的文件"""

import numpy as np
from skimage.metrics import mean_squared_error, structural_similarity


def mae(prediction, target):
    prediction, target = _pair(prediction, target)
    return float(np.mean(np.abs(prediction - target)))


def rmse(prediction, target):
    mse = mean_squared_error(prediction, target)
    return float(np.sqrt(mse))


def ssim(prediction, target, data_range= 1.0):
    """使用 11x11 高斯窗口计算灰度 SSIM，data_range 固定为 1。"""
    return float(
        structural_similarity(
            prediction,
            target,
            data_range=data_range,
        )
    )


def _pair(prediction, target):
    prediction = get_image(prediction)
    target = get_image(target)
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