# 机制诊断（非正式实验）

这里的脚本回答的是**机制问题**，不是论文主流程，也不能替代 Stage-A/B/CV5/E4：

| 脚本 | 问题 |
| --- | --- |
| `init_memory.py` | physics-first 初始化的投影优势能保留多少？锚定强度要多大？背景代价多少？ |
| `alpha_sweep.py` | 哪个 `alpha` 让 screened-Poisson 物理损失降得最快，发生在训练哪一段？ |
| `mode_eval.py` | `star` / `nebula` / `balanced` 三种工作点的背景、星点与投影权衡 |

共同边界：

- 只读 **validation split**，没有 `--split` 参数，test 永远不打开。
- 不修改任何正式 config、lock、seed 或预算；不写 `experiments/outputs/` 下的正式目录。
- 默认输出到 `experiments/outputs/diagnostics/<probe>/<timestamp>_<suite>/`（已被 git 忽略）。
- `smoke` 预算只验证管道，数字没有科学意义；`full` 是声明过的诊断预算。

## 数据

诊断只读 `collections/synthetic_formal/` 与 `collections/synthetic_stage_b/` 的 validation 行。
本机默认只有一两个样本，`smoke` 会自动挑磁盘上存在的样本；`full` 缺文件会直接报错。

```bash
bash tools/fetch_smoke_data.sh          # 解出最小集（约 382 MB）
```

## 初始化探针

```bash
# 管道自检（CPU，几十秒）
python -m experiments.scripts.diagnostics.init_memory --suite smoke

# 全量（GPU；8 个 Stage-B validation 样本）
python -m experiments.scripts.diagnostics.init_memory --suite full --device cuda:0
```

设计：`init ∈ {data_first, physics_first}` × `finetune ∈ {none, data_only, sgc, anchored(λ)}`，
λ 默认 `0.25/0.5/0.75/1.0`。归因：

```text
init_effect        = P(physics, init) - P(data, init)
refine_effect(arm) = P(init, arm, final) - P(init, init)
interaction(arm)   = refine_effect(physics, arm) - refine_effect(data, arm)
retention(arm)     = [P(physics, arm, final) - P(data, arm, final)] / init_effect
```

输出：

- `runs.csv`：每个 cell 的投影、背景、星点指标、耗时
- `attribution.csv`：上面的归因量
- `anchor_curve.csv`：锚定臂
- `summary.json` / `findings.md`：均值、标准差与中性文字结论

## alpha 扫描

```bash
python -m experiments.scripts.diagnostics.alpha_sweep --suite smoke

python -m experiments.scripts.diagnostics.alpha_sweep --suite full --device cuda:0 \
    --dataset stage_a \
    --samples clean_single_002_multi_source_high,clean_single_002_single_radial_high
```

- α 默认 `0.25/0.5/0.75/1.0/1.25/1.5`；`coupled` 让 α 同时进入 PDE 与 forcing 模板，`frozen` 把模板固定在 0.5。
- 训练按 chunk 调用正式 `_train_steps`，每个 chunk 后在固定子集上记录物理损失的六项分解，分解和与 `L_phys` 的误差应小于 1e-9（相对）。
- 指标：`max_log_decay_per_step`、`step_at_max_decay`、`t50/t90`、`final_relative`。
- `cosine_ls` 作为解析对照：它内部硬性要求 α=0.5，诊断只在这次调用周围临时改模块常量（文件本身不改，lock 哈希不变），调用后立刻恢复。

输出：`curves.csv`、`metrics.csv`、`summary.csv`、`cosine_ls_summary.csv`、`alpha_summary.json`；加 `--plot` 生成 `alpha_curves.png`。

## 模式评估

```bash
python -m experiments.scripts.diagnostics.mode_eval --suite smoke

python -m experiments.scripts.diagnostics.mode_eval --suite full --device cuda:0
```

- 默认 arms：`pinn_joint`（nebula）、`sgc`（star）、`balanced@λ`（λ 默认 0.25/0.5/0.75/1.0）。
- `balanced` 的接受规则预先写在 `experiments/configs/diagnostics.py`：投影 ≥ 0.15 且 bg MAE ≤ 1.25×SGC；只是诊断建议，不用于选正式方法。
- 输出：`runs.csv`、`paired_deltas.csv`、`summary.csv`、`summary.json`、`findings.md`。

## 读数注意

- `--max-side` 会做 stride 降采样：投影比和背景指标仍可用，星点指标会被标记 `star_metrics_skipped`，不要和正式数字直接比。
- 全量结论只认全分辨率、完整预算的 `--suite full` 输出；`smoke` 只看管道是否通。
- 归档里已有的冻结结果（`pinn_preservation_exploratory_frozen_v2`）可以作为先验证据，但不要和新探针的运行目录混在一起。
- 单图演示走 `experiments/cli.py`（`python -m experiments.cli estimate ...`），不要用诊断脚本代替 CLI。
