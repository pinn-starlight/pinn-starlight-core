"""E0、E2、E4 共用的 U-Net-small 监督式 baseline。

TODO:
1. 定义单通道输入、单通道背景输出的小型 U-Net。
2. 输入 observed，监督目标为 background_true，训练损失使用 MSE。
3. 仅训练集做裁剪、翻转和旋转增强。
4. 使用验证集早停并保存最佳 checkpoint。
5. 记录网络深度、基础通道数、参数量、训练轮数和随机种子。
6. 测试集只在 checkpoint 锁定后评估一次。
"""

from __future__ import annotations


def build_model():
    """创建最终需要在论文中准确描述的 U-Net-small。"""
    raise NotImplementedError


def train(train_manifest, validation_manifest, output_dir):
    """训练并返回验证集最优 checkpoint 的路径。"""
    raise NotImplementedError


def predict(checkpoint, observed):
    """返回与 observed 同尺寸的 background_pred。"""
    raise NotImplementedError
