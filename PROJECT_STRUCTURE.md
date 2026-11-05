# 项目结构说明

本文档说明仓库的目录职责、模块边界、实验执行顺序与输出位置。  
入口和常用命令见 [README.md](README.md)；论文图与搬运工具分别见 [scripts/figures/README.md](scripts/figures/README.md) 和 [tools/README.md](tools/README.md)。

| 想了解 | 建议阅读 |
| --- | --- |
| 仓库有哪些目录、各自放什么 | 第 2 节「顶层目录」 |
| 核心包/实验模块的职责 | 第 3、4 节 |
| 数据与结果写到哪 | 第 8 节「数据与输出」 |
| 一次完整实验怎么跑 | 第 9 节「运行主线」 |
| 新增代码应该放哪 | 第 10 节「依赖方向与新增代码放哪里」 |

---

## 1. 一句话总览

`pinn-starlight-core` 是 PINN-Starlight 的 Stage-11 研究仓库：

- `src/pinn_starlight_core/` 提供可复用的核心 PINN 组件与一个独立小 demo；
- `experiments/` 实现五种方法、Stage-A/B 数据与全部正式实验；
- `run_stage11.py` 是正式实验的唯一调度入口；
- `scripts/` 放论文图流水线和一次性运维脚本；
- `tools/` 负责服务器结果搬运、离线重建与指标回填；
- `collections/`、`test/` 是数据根，`experiments/outputs/`、`paper_figures/`、`artifacts/` 是输出/归档。

---

## 2. 顶层目录

```text
pinn-starlight-core/
├── README.md                     # 项目入口：环境、运行、约定
├── PROJECT_STRUCTURE.md          # 本文档：逐目录/逐模块说明
├── pyproject.toml                # 项目元数据、依赖、pixi/uv 配置
├── pixi.lock / uv.lock           # 两套包管理器的锁文件
├── .python-version               # Python 3.11
├── .gitignore / .gitattributes
│
├── run_stage11.py                # 正式实验调度器（GPU 隔离、数据准备、阶段编排）
├── run_stage11.sh                # Linux 启动器（GPU/驱动/Pixi 环境检查）
├── run_stage11.ps1               # Windows PowerShell 启动器
│
├── src/pinn_starlight_core/      # 可安装的核心包
├── experiments/                  # Stage-A/B、E1-E4、CV5 的实验代码
├── scripts/                      # 图流水线与一次性运维脚本
├── tools/                        # 服务器归档/恢复/重建/回填工具
├── research_diagnostics/         # 冻结的探索性诊断
│
├── collections/                  # 数据清单与（运行后生成的）数据集
├── test/                         # 原始测试图像（native_test.tif）
├── paper_figures/                # 已生成的论文图 3/4/6
├── experiments/outputs/          # 正式实验输出（运行时生成，已忽略）
├── logs/                         # 调度/训练日志（运行时生成，已忽略）
└── artifacts/                    # 本地大文件归档（已忽略）
```

---

## 3. 核心包：`src/pinn_starlight_core/`

这是唯一被安装进 Python 环境的包（`pip install -e .`），也是 `experiments/` 的底层依赖。核心包不反向依赖 `experiments/`。

```text
src/pinn_starlight_core/
├── __init__.py                   # 包版本
├── __main__.py                   # python -m pinn_starlight_core
├── main.py                       # 独立小 demo：单图训练 + 诊断输出
├── data/
│   ├── __init__.py               # 导出 ImageLoader / coordinate_grid
│   └── image_loader.py           # RAW/TIFF/PNG 读取、降采样、灰度与坐标网格
├── nn/
│   ├── __init__.py
│   ├── pinn_layers.py            # SkyglowMLP（2 -> 128 -> 128 -> 1）
│   ├── pinn_loss.py              # 数据 MSE、物理 MSE、autograd Laplacian
│   └── physics_model.py          # Alpha、EffectiveForcingMixture
└── physics/
    ├── __init__.py
    └── forcing.py                # 与 cosine-domain LS 共用的 forcing 几何工具
```

### 3.1 模块职责

| 模块 | 关键接口 | 说明 |
| --- | --- | --- |
| `data/image_loader.py` | `ImageLoader`、`coordinate_grid` | 读取 `.nef/.cr2/.dng/.arw`、`.tif/.tiff`、`.png/.jpg`；统一到 `[0,1]` 浮点灰度；坐标网格为 `[-1,1]^2` |
| `nn/pinn_layers.py` | `SkyglowMLP` | Xavier 初始化的坐标 MLP，输出单通道背景 |
| `nn/pinn_loss.py` | `mse_data`、`mse_physics`、`laplacian` | screened-Poisson 残差 `ΔB - αB + S_city` 的均方误差 |
| `nn/physics_model.py` | `Alpha`、`EffectiveForcingMixture` | `Alpha` 用 sigmoid 把 α 限制在区间内；`EffectiveForcingMixture` 用观测图初始化最多 K 个 signed forcing 分量 |
| `physics/forcing.py` | `candidate_centers_from_observed`、`point_source_term_numpy`、`build_forcing_component_numpy` | PINN 与 cosine-domain LS 共享的“只看观测”的中心/模板规则 |
| `main.py` | `single_train`、`main` | 非论文主流程；读取 `test/` 下的图片跑一个小规模 demo |

### 3.2 核心数据流（demo）

```text
test/*.tif
  -> ImageLoader.get_gray_data()  -> coords, brightness
  -> SkyglowMLP(coords)          -> background_pred
  -> EffectiveForcingMixture     -> city_pred
  -> Alpha                       -> alpha
  -> mse_data + w * mse_physics  -> backward/step
```

正式实验不直接调用 `main.py`，而是复用其中的网络、损失和 forcing 组件。

---

## 4. 实验代码：`experiments/`

```text
experiments/
├── configs/                     # 配置唯一来源
├── common/
│   ├── baselines/               # 五种方法实现
│   ├── data/                    # 数据生成与背景建模
│   └── utils/                   # IO、指标、选择产物、Stage-B 专用工具
└── scripts/
    ├── preflight/               # 只读预检
    ├── selection/               # Stage-A 三个方法的超参选择
    ├── e1/                      # E1：PINN 选择
    ├── config_lock/             # Global Configuration Lock
    ├── e2/                      # E2：冻结 benchmark
    ├── e3/                      # E3：PINN 消融
    ├── e4/                      # E4：真实图像五方法对比
    ├── stage_b/                 # Stage-B：选择、lock、formal run
    ├── final_pinn/              # 最终 PINN 的配对评估与 CV5 聚合
    ├── protocol/                # Stage-A/B/CV5 的编排、准备与聚合
    └── verify_cosine_ls/        # DCT-II LS 实现的数值自检
```

`experiments/` 使用命名空间包组织（子目录没有 `__init__.py`），所有命令从仓库根目录运行。

### 4.1 `common/baselines/`：五种方法

| 文件 | 方法 | 角色 |
| --- | --- | --- |
| `coordinate_pinn.py` | Coordinate PINN | E0/E1/E2/E3/E4 共用的标准 PINN；包含 `PINNConfig`、`train_background` |
| `sgc_pinn.py` | SGC-PINN（最终方法） | Stage-11 最终方法：数据-only `B0` → 固定几何 forcing 拟合 → 解耦梯度微调 |
| `fft_gaussian.py` | FFT-Gaussian | 频域高斯低通基线 |
| `cosine_ls.py` | Cosine-LS（DCT-II 解析解） | 在 DCT-II 域解析求解 screened-Poisson；对 K 个 forcing 活跃集枚举 |

所有方法都遵守同一个边界：**只用观测值和 train/validation 做选择或训练，test 只在冻结之后打开**。

### 4.2 `common/data/`：数据与背景

| 文件 | 作用 |
| --- | --- |
| `synthetic_backgrounds.py` | 正式实验的三种合成光污染背景（单源/偏心/多源等） |
| `native_background.py` | 用固定 `photutils.Background2D` 规则估计并扣除相机原生低频背景 |
| `generate_synthetic.py` | 生成 Stage-A 半合成数据集 → `collections/synthetic_formal/` |
| `stage_b_backgrounds.py` | Stage-B 的频谱重叠压力测试场景（含保护结构） |
| `generate_stage_b.py` | 生成 Stage-B 数据集 → `collections/synthetic_stage_b/` |
| `elliptical_background.py` | 旧椭圆背景工具的兼容包装与可视化示例 |
| `exponential_background.py` | 旧指数背景工具的兼容包装与可视化示例 |

### 4.3 `common/utils/`：实验基础设施

| 文件 | 作用 |
| --- | --- |
| `experiment_utils.py` | 数据根解析、manifest 读取、TIFF/PNG IO、输出目录、随机种子、Global Lock 校验 |
| `metrics.py` | E1-E4 共用指标：背景 MAE/SSIM、残差、星点检测/测光、forced reference、真实图像评估 |
| `scores.py` | 基础图像指标（MAE/RMSE/PSNR/SSIM），输入不做裁剪 |
| `selection_artifacts.py` | 选择产物（`*_selected.json`）的 provenance 与依赖校验 |
| `stage_b_metrics.py` | Stage-B 扩展指标：暗星召回、保护区域 MAE、保护结构投影误差 |
| `stage_b_anchor.py` | Stage-A 与 Stage-B 的兼容性锚点检查 |
| `stage_b_provenance.py` | Stage-B 选择产物与 lock 的 provenance |
| `e4_visualization.py` | E4 论文图的显示策略（每张源图自己的色调映射） |

### 4.4 `configs/`：配置唯一来源

| 文件 | 作用 |
| --- | --- |
| `formal_experiment.py` | Stage-A 的 `FORMAL_METHODS`、随机种子、候选空间、执行预算、最终 PINN 配置 |
| `stage_b_experiment.py` | Stage-B 配置；继承 Stage-A 的候选空间与预算，只改变场景分布 |
| `split_protocols.py` | 五个相机基准的分组规则：`standard`（3/1/1）与 `cv5`（五折，fold 0 等于 standard） |

新增方法、指标或阶段时，优先在这里登记，而不是在各脚本里另写一份常量。

### 4.5 `scripts/`：各阶段入口

| 目录/文件 | 阶段 | 作用 |
| --- | --- | --- |
| `preflight/run.py` | 公共 | 只读检查数据、manifest、真实图像数量 |
| `selection/gaussian.py` | Stage-A 选择 | 选择 FFT-Gaussian 的 σ |
| `selection/fourier_ls.py` | Stage-A 选择 | 选择 cosine-domain LS 的 λ 与稀疏权重 |
| `selection/pinn.py` | Stage-A 选择 | 选择 PINN 配置 |
| `config_lock/run.py` | Stage-A 冻结 | 生成 Global Configuration Lock |
| `benchmark.py` | Stage-A 评估 | 三方法冻结 benchmark（只打开 test） |
| `ablation.py` | Stage-A 消融 | Original-PINN 消融 |
| `real_images/run.py` | E4 | 真实高污染图像四方法对比 |
| `real_images/render_for_paper.py` | E4 出图 | 按每张源图的显示风格渲染 E4 结果 |
| `stage_b/preflight.py` | Stage-B | Stage-B 预检 |
| `stage_b/select_gaussian.py` | Stage-B 选择 | Gaussian 选择 |
| `stage_b/select_fourier_ls.py` | Stage-B 选择 | cosine-domain LS 选择 |
| `stage_b/select_pinn.py` | Stage-B 选择 | PINN 选择 |
| `stage_b/lock.py` | Stage-B 冻结 | 生成 Stage-B Global Lock |
| `stage_b/formal_run.py` | Stage-B 评估 | Stage-B 正式 benchmark |
| `final_pinn/stage_a.py` | Stage-A 确认 | data-only vs 最终 PINN 的配对评估 |
| `final_pinn/stage_b.py` | Stage-B 确认 | Stage-B 上的最终 PINN 配对评估 |
| `final_pinn/common.py` | 共用 | forced-reference 指标、配对比较、CSV 写出 |
| `final_pinn/aggregate_cv5.py` | CV5 汇总 | 按相机基准聚合最终 PINN 五折结果 |
| `protocol/run.py` | 编排 | 用同一个代码路径跑 standard / cv5 的 A/B 阶段 |
| `protocol/prepare_cv5.py` | CV5 准备 | 生成五折 split view（只改 manifest，不复制 TIFF） |
| `protocol/aggregate_cv5.py` | CV5 汇总 | 按 fold × method 聚合并输出均值/标准差 |
| `verify_cosine_ls/run.py` | 自检 | 验证 DCT-II Neumann 实现 |

---

## 5. 图与运维：`scripts/`

```text
scripts/
├── figures/                      # 论文图 3/4/6
│   ├── collect_assets.py         # 自动匹配并收集冻结数组（推荐）
│   ├── pack_figure_assets.py     # 按固定样本打包三张图的 ZIP
│   ├── render_figures.py         # 参数化 --root 的渲染器
│   ├── render_figures_server_exact.py  # 固定 /workspace/stage11 的精确渲染器
│   └── legacy/                   # 历史版本（不再维护）
└── ops/
    ├── backfill_forced_reference.sh     # 只读回填 forced-reference 指标
    └── validate_pinn_preservation.sh    # 冻结 PINN 保持性验证
```

这些脚本只读冻结结果，不训练模型、不修改原始输出。

---

## 6. 搬运与重建：`tools/`

| 文件 | 作用 |
| --- | --- |
| `export_server_archives.py` | 服务器端增量导出 ZIP；按 P00-P07 优先级分卷，默认 4 GiB |
| `restore_server_archives.py` | 本地恢复 ZIP；支持 `--verify-only`、`--phases`、`--no-overwrite` |
| `rebuild_derived_outputs.py` | 从冻结的 `background_pred.tif` 重建 observed/residual TIFF 与 PNG |
| `verify_rebuild_ready.py` | 校验恢复后的 manifest、TIFF 与派生文件完整性 |
| `backfill_forced_reference_metrics.py` | 仅评估：对冻结预测回填 forced-reference 指标 |

典型链路：

```text
服务器 experiments/outputs/
  -> export_server_archives.py   -> stage11_transfer_archives/B*.zip
  -> （传输）
  -> restore_server_archives.py  -> 本地 experiments/outputs/
  -> rebuild_derived_outputs.py  -> observed/residual/PNG
  -> verify_rebuild_ready.py     -> 完整性检查
```

`export_server_archives.py` 会跳过 `artifacts/`、`.venv/`、`.idea/`、`.agents/`、缓存和原始 `collections/`/`test/` 数据，避免把本地归档再次打包。

---

## 7. 探索性诊断：`research_diagnostics/`

| 文件 | 作用 |
| --- | --- |
| `pinn/pinn_preservation_frozen.py` | 冻结的 PINN 保持性/分解探索性验证；不属于正式 A/B 主流程 |
| `pinn/pinn_preservation_frozen_manifest.json` | 上述冻结脚本的清单与校验信息 |

对应的启动脚本是 `scripts/ops/validate_pinn_preservation.sh`。

---

## 8. 数据与输出

### 8.1 数据根

默认数据根是仓库根目录；也可以用环境变量覆盖：

```bash
export PINN_STARLIGHT_DATA_ROOT=/path/to/data-root
```

该目录需要包含：

```text
<data-root>/
├── collections/
│   ├── manifest.csv                  # 采集清单（源数据）
│   ├── real_high_pollution/          # E4 真实图像（期望 16 张）
│   ├── synthetic_formal/             # Stage-A 生成数据（运行时）
│   └── synthetic_stage_b/            # Stage-B 生成数据（运行时）
└── test/
    └── native_test.tif               # 原始测试图 / demo 输入
```

### 8.2 输出目录

```text
experiments/outputs/
├── selection/
│   ├── pinn/                         # PINN 选择产物
│   ├── fft_gaussian/
│   └── fourier_ls/
├── configuration_lock/               # Stage-A Global Configuration Lock
├── e2_benchmark/                     # 三方法冻结 benchmark
├── e3_ablation/                      # Original-PINN 消融
├── final_pinn_validation/            # 最终 PINN 配对评估
├── e4_real/                          # 真实图像五方法对比
├── stage_b/
│   ├── selection/                    # 四个 Stage-B 选择器
│   ├── configuration_lock/           # Stage-B Global Lock
│   ├── benchmark/                    # Stage-B 正式 benchmark
│   └── final_pinn_validation/        # Stage-B 最终 PINN 评估
├── cv5/
│   └── fold_0 … fold_4/              # 每折独立的 selection/benchmark/final 输出
└── cv5_summary/
    ├── stage_a/                      # baseline 五折汇总
    ├── stage_b/
    └── stage_a_final_pinn/           # 最终 PINN 五折汇总
```

其他输出：

- `paper_figures/`：论文图 3/4/6（PNG + PDF）；
- `logs/`：调度器与 CV5 fold 日志；
- `artifacts/`：本地大文件归档（默认被 git 忽略）。

---

## 9. 运行主线

正式入口统一是 `run_stage11.py`（由 `.sh`/`.ps1` 包装）。

### 9.1 Stage-A（`a`）

```text
ensure_dataset(synthetic_formal)
  -> protocol.run --stage a --phase all
       -> preflight
       -> selection.gaussian / selection.fourier_ls / selection.pinn
       -> config_lock
       -> benchmark
       -> ablation
  -> final_pinn.stage_a
```

### 9.2 Stage-B（`b`）

```text
ensure_dataset(synthetic_stage_b)
  -> protocol.run --stage b --phase all
       -> stage_b.preflight
       -> stage_b.select_gaussian / select_fourier_ls / select_pinn
       -> stage_b.lock
       -> stage_b.formal_run
  -> final_pinn.stage_b
```

### 9.3 CV5（`cv5`）

```text
protocol.prepare_cv5
  -> 对 fold 0..4:
       protocol.run --stage a --mode cv5 --fold N
       final_pinn.stage_a
  -> protocol.aggregate_cv5       -> cv5_summary/stage_a
  -> final_pinn.aggregate_cv5    -> cv5_summary/stage_a_final_pinn
```

多个 GPU 时，不同 fold 可以作为独立 worker 并行；每个科学子进程仍然只看到一个 GPU。

### 9.4 E4（`e4`）

```text
experiments.scripts.real_images.run
  -> 五种方法在 collections/real_high_pollution/ 上逐图推理
  -> experiments/outputs/e4_real/
```

### 9.5 论文图

```text
服务器冻结输出
  -> scripts/figures/collect_assets.py  （或 pack_figure_assets.py）
  -> scripts/figures/render_figures_server_exact.py
  -> paper_figures/Fig3_*.png/.pdf, Fig4_*.png/.pdf, Fig6_*.png/.pdf
```

---

## 10. 依赖方向与新增代码放哪里

下图是主要依赖方向（`experiments/configs` 也会引用 `experiments/common` 中的场景定义）：

```text
run_stage11.py
  └── experiments/scripts/*            # 阶段编排、入口
        ├── experiments/configs/*      # 配置与协议
        ├── experiments/common/*       # 方法、数据、指标
        └── src/pinn_starlight_core/*  # 网络、损失、forcing、图像 IO
```

约定：

- 新方法：放 `experiments/common/baselines/`，并在 `experiments/configs/formal_experiment.py` 登记；
- 新指标：放 `experiments/common/utils/metrics.py` 或 `scores.py`；
- 新实验阶段：放 `experiments/scripts/<stage>/`，通过 `protocol/run.py` 或 `run_stage11.py` 编排；
- 一次性脚本：放 `scripts/figures/` 或 `scripts/ops/`，不要堆回根目录；
- 服务器数据工具：放 `tools/`；
- 不要手改 `experiments/outputs/`、`paper_figures/` 里的结果；重新生成请走脚本。

---

## 11. 常用命令速查

```bash
# 正式实验
./run_stage11.sh a
./run_stage11.sh b
./run_stage11.sh cv5
./run_stage11.sh e4
./run_stage11.sh all

# 单阶段（示例）
pixi run -e remote python -m experiments.scripts.protocol.run \
    --stage a --mode standard --phase all --device cuda:0 --include-ablation

# 数据预检
pixi run -e remote python -m experiments.scripts.preflight.run

# 论文图
pixi run -e remote python scripts/figures/collect_assets.py --root /workspace/stage11 --dry-run
pixi run -e remote python scripts/figures/collect_assets.py --root /workspace/stage11 --zip
pixi run -e remote python scripts/figures/render_figures_server_exact.py --out paper_figures --dpi 600

# 归档恢复/重建
pixi run -e remote python tools/restore_server_archives.py artifacts/transfers/stage11_transfer_archives .
pixi run -e remote python tools/rebuild_derived_outputs.py
pixi run -e remote python tools/verify_rebuild_ready.py --require-derived

# 指标回填与保持性验证
bash scripts/ops/backfill_forced_reference.sh
bash scripts/ops/validate_pinn_preservation.sh
```

---

## 12. 历史与归档

- `scripts/figures/legacy/`：旧版图脚本（`collect_figure_assets.py`、`collect_stage11_figure_assets.py`、`render_figures_exact_v1.py`），仅作历史参考。
- `artifacts/`：
  - `transfers/stage11_transfer_archives/`：服务器搬运 ZIP（约 9.6 GB）；
  - `transfers/transfer_forced_reference_20260912_225540/`：forced-reference 回填批次；
  - `review/e4_review_for_paper.zip`：E4 审阅图；
  - `logs/`：已结束的 backfill 日志/PID/退出码；
  - `patches/PATCH_MANIFEST.json`：历史补丁清单。
- `artifacts/` 除 `README.md` 外均被 `.gitignore` 忽略；确认备份后可释放空间。

---

## 13. 维护检查

```bash
# 目录与缓存
git status --short

# Python 语法（会生成 __pycache__，已由 .gitignore 忽略）
python -m compileall -q src experiments tools scripts

# 核心包与实验模块导入
PYTHONPATH=src python -c "import pinn_starlight_core; import experiments.scripts.protocol.run"

# 入口帮助
python run_stage11.py --help
python tools/restore_server_archives.py --help
python scripts/figures/collect_assets.py --help
```
