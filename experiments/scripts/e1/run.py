"""E1：只使用合成验证集锁定 PINN 配置。运行入口：本文件。

依次选择，避免完全网格搜索：
T1 网络结构：[512]、[256, 64]、[128, 128]
T2 physics_weight：0.01、0.1、0.3、0.4、0.5
T3 kernel_size：21、31
T4 训练步数：从 3000 继续到验证曲线稳定

优先比较 background MAE，再比较 residual PSNR/SSIM 和光通量误差。
E1 完成后输出一份只读配置，E2-E4 不再调 PINN 参数。
"""

from __future__ import annotations


def run_candidate(config: dict, validation_manifest) -> dict[str, float]:
    """运行一个候选配置并保存完整结果。"""
    # TODO: 从正式 PINN 训练函数调用，不要复制多份训练循环。
    raise NotImplementedError


def main() -> None:
    # TODO: 固定 alpha=0.5；先使用一个开发种子逐阶段选择。
    # TODO: 保存每个候选结果和最终 locked_pinn_config.json。
    raise NotImplementedError


if __name__ == "__main__":
    main()
