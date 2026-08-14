"""正式实验共用工具。

TODO:
1. 统一项目根目录、数据目录和输出目录。
2. 固定 Python/NumPy/PyTorch 随机种子。
3. 保存 config.json、metrics.csv、summary.json 和运行环境。
4. 保存浮点数组，再单独生成展示用 PNG。
5. 记录当前 Git commit；工作区非干净时给出明确标记。
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "experiments/output"
SEEDS = (20260728, 20260729, 20260730)


def set_seed(seed: int) -> None:
    """固定所有实际使用的随机数源。"""
    # TODO: 根据脚本实际依赖设置 random、NumPy、PyTorch 和 CUDA。
    raise NotImplementedError


def save_run_metadata(output_dir: Path, config: dict) -> None:
    """保存配置、环境、Git commit 和运行状态。"""
    # TODO: 写入 config.json 和环境信息；不要只保存人类可读的 txt。
    raise NotImplementedError


def save_prediction_arrays(
    output_dir: Path,
    observed,
    background_pred,
    residual_pred,
) -> None:
    """保存用于指标计算的浮点数组以及对应展示图。"""
    # TODO: 浮点数组保留原始数值；PNG 仅用于展示。
    raise NotImplementedError
