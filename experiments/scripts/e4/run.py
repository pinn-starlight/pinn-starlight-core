"""E4：固定真实图像对比。运行入口：本文件。

使用已锁定的 FFT 参数、U-Net checkpoint 和 PINN 配置。主图固定为
data/image/f7.5.tif；如需补充图，必须在运行前写入清单。

输出布局：
observed | FFT background | U-Net background | PINN background
observed | FFT residual   | U-Net residual   | PINN residual

真实图没有背景真值，不报告 PSNR/SSIM，也不宣称恢复不可见星光。
"""

from __future__ import annotations


REAL_IMAGES = ("data/image/f7.5.tif",)


def main() -> None:
    # TODO: 固定裁剪和显示范围，三种方法使用同一输入。
    # TODO: 保存全分辨率浮点数组、对照图、耗时和失败案例备注。
    # TODO: 检查星点吸收、结构误判、边缘伪影、负值和新增亮点。
    raise NotImplementedError


if __name__ == "__main__":
    main()
