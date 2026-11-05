# PINN-Starlight Core

物理启发的夜空 / 光污染背景估计研究代码。仓库包含 Stage-A / Stage-B 正式实验、E1–E4 子实验、CV5 分组评估、五种方法的实现、论文图流水线，以及服务器结果搬运与离线重建工具。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `src/pinn_starlight_core/` | 可安装的核心包：坐标 MLP、Laplacian/损失、effective forcing、图像读取；`python -m pinn_starlight_core` 是独立小 demo，不属于论文主流程 |
| `experiments/common/baselines/` | 五种方法：coordinate PINN、decoupled-gradient PINN（最终方法）、FFT-Gaussian、cosine-domain LS、U-Net |
| `experiments/configs/` | 实验配置的唯一来源：Stage-A/Stage-B、split 与 CV5 协议 |
| `experiments/scripts/e1`…`e4` | 选择、冻结 benchmark、消融、真实图像实验 |
| `experiments/scripts/protocol/` | Stage-A / Stage-B / CV5 的阶段编排与聚合 |
| `experiments/scripts/final_pinn/` | 最终 PINN 的配对评估与 CV5 聚合 |
| `scripts/figures/` | 论文图 3/4/6 的资产收集、渲染与打包 |
| `scripts/ops/` | 一次性回填 / 保持性验证脚本 |
| `tools/` | 服务器归档导出/恢复、派生文件重建、指标回填、完整性检查 |
| `research_diagnostics/` | 冻结的探索性诊断脚本 |
| `collections/`、`test/` | 数据根：采集清单与原始测试图 |
| `paper_figures/` | 已生成的论文图（输出） |
| `artifacts/` | 本地大文件归档（含约 9.6 GB 服务器搬运包；已被 git 忽略） |

更细的目录/模块说明见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)；各子系统的说明见 [experiments/README.md](experiments/README.md)、[scripts/figures/README.md](scripts/figures/README.md)、[tools/README.md](tools/README.md) 和 [artifacts/README.md](artifacts/README.md)。

## 环境

- Python 3.11
- 推荐 Pixi：`pixi install`（CUDA profile 见 `pyproject.toml`）
- 或 uv/pip：`uv sync` / `pip install -e .`

`pyproject.toml` 同时保留了 pixi 与 uv 配置；`pixi.lock` 与 `uv.lock` 都在版本控制中。

## 运行正式实验

Linux：

```bash
./run_stage11.sh a       # Stage-A：数据检查 -> 选择 -> 配置锁 -> E2 -> E3 -> Final PINN
./run_stage11.sh b       # Stage-B 压力测试
./run_stage11.sh cv5     # 5-fold 分组评估
./run_stage11.sh e4      # 真实高污染图像五方法对比
./run_stage11.sh all     # A -> B -> CV5 -> E4
```

Windows PowerShell：

```powershell
.\run_stage11.ps1 a
```

`run_stage11.sh` 会先检查 GPU、驱动与 Pixi 环境。每个科学子进程只看到一个 GPU（内部仍使用 `cuda:0`），调度策略的完整说明在 `run_stage11.py` 顶部。也可以直接调用模块，例如：

```bash
pixi run -e remote python -m experiments.scripts.protocol.run \
    --stage a --mode standard --phase all --device cuda:0 --include-ablation
```

## 数据与输出

- 默认数据根是仓库根目录下的 `collections/` 与 `test/`；也可以设置 `PINN_STARLIGHT_DATA_ROOT` 指向包含这两个目录的位置。
- 正式输出写到 `experiments/outputs/`（已忽略）；CV5 写到 `experiments/outputs/cv5/fold_<n>/`。
- 数据生成、split 规则与候选空间都集中在 `experiments/configs/`。

## 论文图与工具

- 图 3/4/6 的收集与渲染：见 `scripts/figures/README.md`。
- 服务器归档导出/恢复与离线重建：见 `tools/README.md`。
- `paper_figures/` 中是已生成结果；重新渲染请用脚本，不要手工修改图片。

## 仓库约定

- `__pycache__`、`.pytest_cache`、`.ruff_cache`、`*.egg-info/` 等生成物不入库。
- `experiments/outputs/`、`logs/`、`artifacts/` 不入库。
- `scripts/figures/legacy/` 仅保留历史版本，当前流程不使用。
