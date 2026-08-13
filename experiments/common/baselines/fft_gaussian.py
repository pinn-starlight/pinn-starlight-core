"""E0、E2、E4 共用的 FFT-Gaussian 背景估计 baseline。

约定：
- 输入和输出均为 [0, 1] 线性灰度浮点数组。
- 先做 reflection padding，频域高斯低通后裁回原尺寸。
- 截止尺度只能在验证集从 0.02/0.04/0.08/0.16 中选择一次。
- 测试集和真实图使用同一个锁定值。
"""

from __future__ import annotations


def estimate_background(observed, normalized_sigma: float):
    """返回 background_pred，不在这里裁剪 residual_pred。"""
    # TODO: 实现 fft2/fftshift、高斯传递函数、ifft2 和裁剪。
    raise NotImplementedError


def select_sigma(validation_manifest) -> float:
    """仅根据验证集背景指标选择截止尺度。"""
    # TODO: 保存所有候选结果，不要只保存最优值。
    raise NotImplementedError
