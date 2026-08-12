"""E2：合成测试集主对比。运行入口：本文件。

前置条件：
- 数据 manifest、FFT 截止尺度、U-Net checkpoint、PINN 配置均已锁定。
- 测试集没有参与任何调参。

输出：逐图 metrics.csv、按方法和背景类型汇总的 summary.json、三种方法的
background_pred/residual_pred，以及论文主结果表所需数据。
"""

from __future__ import annotations


METHODS = ("fft_gaussian", "unet_small", "pinn")


def run_method(method: str, sample: dict, locked_config: dict) -> dict:
    """统一返回 background_pred、residual_pred 和耗时。"""
    # TODO: 三种方法必须复用同一预处理结果。
    raise NotImplementedError


def main() -> None:
    # TODO: 每个测试样本按三个种子运行需要随机性的部分。
    # TODO: 分开记录 U-Net 离线训练、推理和 PINN 逐图优化成本。
    raise NotImplementedError


if __name__ == "__main__":
    main()
