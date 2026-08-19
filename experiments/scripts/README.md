# 论文实验脚本

正式协议见 [`docs/论文准备资料/03_实验/论文实验计划.md`](../../docs/论文准备资料/03_实验/论文实验计划.md)。代码入口固定为 E0-E4，正式输出统一写入 `experiments/outputs/`。

## 运行顺序

在项目根目录运行：

```powershell
$env:PYTHONPATH="src;."

# 1. 生成 30 个正式合成样本和固定 60/20/20 manifest
uv run python -m experiments.common.data.generate_synthetic

# 2. E0 只检查流程，不进入论文结果
uv run python -m experiments.scripts.e0.run

# 3. 依次锁定网络、物理权重、初始化尺度和步数
uv run python -m experiments.scripts.e1.run

# 4. FFT-Gaussian、U-Net-small、PINN 三方法主对比
uv run python -m experiments.scripts.e2.run

# 5. PDE 与中心模式消融
uv run python -m experiments.scripts.e3.run

# 6. 固定真实图定性对比
uv run python -m experiments.scripts.e4.run
```

所有入口都支持 `--help`。正式默认值符合实验计划；只有 E0 或调试时才应使用较少步数、较少样本或单个种子。调试结果不能填入论文。

PINN 的正式默认源点模式为 `bright_init_fixed`：先用平滑亮区估计源点，再在训练中固定该物理位置。`bright_init_learnable` 仍可通过 E1 的 `--center-mode` 或 E3 消融显式运行，但在当前 PDE 辅助项较弱且没有源点标注的目标下，它只代表“允许漂移”的对照，不应默认解释为更优的定位方法。

实验入口默认拒绝非空输出目录，避免覆盖已有结果。确认要重跑时，可以显式加入 `--force`；它只清空 `experiments/outputs` 下指定的那个实验目录，不会清理其他目录。例如：

```powershell
uv run python -m experiments.scripts.e2.run --force --output-root experiments/outputs/e2_synthetic_fixed
uv run python -m experiments.scripts.e4.run `
  --e2-config experiments/outputs/e2_synthetic_fixed/locked_e2_config.json `
  --force `
  --output-root experiments/outputs/e4_real_fixed
```

## 数据协议

`generate_synthetic` 从 `data/collections/manifest.csv` 读取 5 张 `synthetic_base_candidate`，先按基础图固定划分，再为每张图生成：

- `single_radial`：单中心径向背景；
- `single_eccentric`：偏心椭圆背景；
- `multi_source`：双光源非均匀背景；
- `low/high` 两档强度、尺度和噪声。

最终得到 18 个训练样本、6 个验证样本和 6 个测试样本。同一基础图不会跨集合。后续实验只读取 `data/collections/synthetic/manifest.csv`，不会临时重新划分。

星点 Precision、Recall、F1 和光通量误差使用 `clean_true` 上预先固定的局部对比极大值作为参考，阈值为 `0.03`，匹配半径为 3 像素。它不是外部星表真值，论文中必须称为“由干净参考图提取的星点参考”。

星点检测最多保留 2000 个候选；当 `*_star_count_capped` 或 `detected_star_count_capped` 为 `1` 时，对应数量只是下界，不能按精确星数解读。

## 正式输出

每个实验根目录保存 `config.json`、`environment.json`、`metrics.csv` 和 `summary.json`。每张预测保存：

```text
observed.tif / observed.png
background_pred.tif / background_pred.png
residual_pred.tif / residual_pred.png
```

TIFF 是指标计算用的 32 位浮点数组，PNG 只用于查看和论文排版。

- E1：`candidate_results.csv`、`locked_pinn_config.json`、每个候选的 loss 与中心轨迹；
- E2：`main_result_table.csv`、`grouped_results.csv`、`failure_cases.csv`、U-Net checkpoint 与 `locked_e2_config.json`；
- E3：`ablation_table.csv`、四个版本的逐图指标与中心轨迹；
- E4：2×4 `comparison.png`、无参考统计和待人工填写的 `inspection_checklist.md`。

只有 `main_result_table.csv`、`grouped_results.csv`、`ablation_table.csv` 和 E4 对比图经过完整正式运行后，才可进入论文结果。
