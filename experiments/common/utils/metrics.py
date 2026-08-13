"""E1-E4 共用的评价指标。

合成数据：
- background_pred 对 background_true：MAE、RMSE、PSNR、SSIM
- residual_pred 对 clean_true：MAE、PSNR、SSIM
- 星点：Precision、Recall、F1、平均相对光通量误差
- 单源偏心子集：中心定位误差

真实图没有背景真值，不计算 PSNR/SSIM。星点阈值和匹配半径必须在
验证集锁定，不能针对测试图或单张真实图修改。
"""

from __future__ import annotations


def evaluate_synthetic(sample: dict, prediction: dict) -> dict[str, float]:
    """计算单个合成样本的统一指标。"""
    # TODO: 明确 data_range=1.0，并处理除零和空星点情况。
    raise NotImplementedError


def evaluate_real(observed, background_pred, residual_pred) -> dict[str, float]:
    """只返回无需真值即可直接计算、且含义明确的统计量。"""
    # TODO: 先决定真正需要报告的无参考统计，避免堆无意义指标。
    raise NotImplementedError
