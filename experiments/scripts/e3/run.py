"""E3：PDE 与源点中心最小消融。运行入口：本文件。

四个版本：
- DataOnly：physics_weight=0
- CenterFixed：中心固定为 (0, 0)
- BrightInitFixed：亮区初始化后冻结中心
- BrightInitLearnable：相同初值，训练中学习中心

除上述变量外，全部使用 E1 锁定配置。只在单源偏心子集报告中心误差，
多光源样本不计算单一 E_center。
"""

from __future__ import annotations


VARIANTS = (
    "data_only",
    "center_fixed",
    "bright_init_fixed",
    "bright_init_learnable",
)


def build_variant_config(name: str, locked_pinn_config: dict) -> dict:
    """只修改该消融允许改变的中心/PDE 设置。"""
    # TODO: 加断言，确保其余配置与 E1 完全一致。
    raise NotImplementedError


def main() -> None:
    # TODO: 使用与 E2 相同的固定测试集和三个随机种子。
    # TODO: 保存中心轨迹、稳定性信息和完整消融表数据。
    raise NotImplementedError


if __name__ == "__main__":
    main()
