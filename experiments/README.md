# experiments

Stage-A / Stage-B 正式实验代码。这里使用命名空间包（子目录没有 `__init__.py`），所有命令都从仓库根目录运行。

## 结构

| 路径 | 作用 |
| --- | --- |
| `common/baselines/` | 五种方法实现：coordinate PINN、decoupled-gradient PINN、FFT-Gaussian、cosine-domain LS、U-Net |
| `common/data/` | 半合成数据生成（Stage-A/B）、native 背景估计 |
| `common/utils/` | 指标、选择产物、实验 IO、Stage-B 专用工具 |
| `configs/` | `formal_experiment.py`（Stage-A）、`stage_b_experiment.py`（Stage-B）、`split_protocols.py`（standard / CV5） |
| `scripts/e1`…`e4` | E1 选择、E2 冻结 benchmark、E3 消融、E4 真实图像 |
| `scripts/selection/` | Stage-A 的四种方法超参选择 |
| `scripts/stage_b/` | Stage-B 的选择、preflight、lock、formal run |
| `scripts/protocol/` | Stage-A/B 的阶段编排、CV5 准备与聚合 |
| `scripts/final_pinn/` | 最终 PINN 的 Stage-A/B 配对评估与 CV5 聚合 |

## 阶段顺序

Stage-A standard：

1. `experiments.scripts.preflight.run` — 数据与 manifest 检查
2. `selection.gaussian` / `selection.fourier_ls` / `selection.unet` / `e1` — 仅用 train/validation 选择
3. `config_lock.run` — Global Configuration Lock
4. `e2.run` — 四方法冻结 benchmark
5. `e3.run` — Original-PINN 消融（正式入口默认包含）
6. `final_pinn.stage_a` — 最终 PINN 的配对评估

Stage-B：

1. `stage_b.preflight`
2. `stage_b.select_gaussian` / `select_fourier_ls` / `select_unet` / `select_pinn`
3. `stage_b.lock`
4. `stage_b.formal_run`
5. `final_pinn.stage_b`

CV5：

1. `protocol.prepare_cv5` 生成五折视图
2. 每一折执行 `protocol.run --stage a --mode cv5 --fold N`
3. `protocol.aggregate_cv5` 聚合 baseline 结果
4. `final_pinn.aggregate_cv5` 聚合最终 PINN 结果

E4：

1. `experiments.scripts.real_images.run` — 五方法真实图像对比

## 说明

- 配置是单一来源：方法候选空间、随机种子、执行预算都在 `configs/` 中，不要在各脚本里另写一份。
- 选择阶段只接触 train/validation；test 数据在 Global Lock 之后才打开。
- 顶层 `run_stage11.py` 负责 GPU 隔离、数据生成与阶段调度；这里的模块可以被单独调用，但正式实验应通过入口运行。
