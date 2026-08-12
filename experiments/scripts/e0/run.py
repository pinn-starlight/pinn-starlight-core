"""E0：先试试三种方法能不能跑，不进入论文结果。

目标：用两张合成样本和 native_test.tif 固定裁剪，让 FFT-Gaussian、
U-Net-small 和 PINN 都完成一次最短运行。这里只检查整条流程是否可用，
不比较方法效果，也不把数值写入论文结果。

最简单的通过条件：
- 输入、background_pred、residual_pred 尺寸和数值范围正确；
- loss、指标和参数无 NaN/Inf；
- 至少能打印 loss，并保存一张背景图和一张残差图；
- 当前 PINN 加载器接口已修正，alpha 固定为 0.5。
"""

from __future__ import annotations


def main() -> None:
    # TODO: 每种方法只跑最少量步骤，能输出 loss 和图片就行。
    # TODO: E0 只回答“程序能不能跑”，不比较谁的效果好。
    raise NotImplementedError


if __name__ == "__main__":
    main()
