# PINN-Starlight Core

物理启发的夜空 / 光污染背景估计研究代码。仓库包含 Stage-A / Stage-B 正式实验、E1–E4 子实验、CV5 分组评估、四种方法的实现、论文图流水线，以及服务器结果搬运与离线重建工具。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `src/pinn_starlight_core/` | 可安装的核心包：坐标 MLP、Laplacian/损失、effective forcing、图像读取；`python -m pinn_starlight_core` 是独立小 demo，不属于论文主流程 |
| `experiments/configs/` | 实验配置的唯一来源：Stage-A/Stage-B、split 与 CV5 协议 |
| `experiments/scripts/selection/` | Stage-A 三种方法（Gaussian / Fourier-LS / PINN）的超参选择 |
| `experiments/scripts/benchmark.py`、`ablation.py` | 冻结 benchmark 与 Original-PINN 消融 |
| `experiments/scripts/real_images/` | 真实高污染图像对比与论文出图 |
| `experiments/scripts/stage_b/` | Stage-B 的选择、preflight、lock 与正式运行 |
| `experiments/scripts/protocol/` | Stage-A / Stage-B / CV5 的阶段编排与聚合 |
| `experiments/scripts/final_pinn/` | 最终 PINN 的配对评估与 CV5 聚合 |
| `experiments/scripts/diagnostics/` | 非正式机制探针（初始化 / alpha，validation only） |
| `experiments/cli/` | `pinn-starlight` CLI：单图估计、doctor、样本列表 |
| `scripts/figures/` | 论文图 3/4/6 的资产收集、渲染与打包 |
| `tools/` | 服务器归档导出/恢复、派生文件重建、指标回填、完整性检查 |
| `research_diagnostics/` | 冻结的探索性诊断脚本 |
| `collections/`、`test/` | 数据根：采集清单与原始测试图 |
| `paper_figures/` | 已生成的论文图（输出） |
| `archived` | 本地大文件归档（含约 9.6 GB 服务器搬运包；已被 git 忽略） |

更细的目录/模块说明见 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)；各子系统的说明见 [experiments/README.md](experiments/README.md)、[tools/README.md](tools/README.md) 和 [artifacts/README.md](archived/README.md)。

## 环境

- Python 3.11
- 推荐 Pixi：`pixi install`（CUDA profile 见 `pyproject.toml`）
- 或 uv/pip：`uv sync` / `pip install -e .`

`pyproject.toml` 同时保留了 pixi 与 uv 配置；`pixi.lock` 与 `uv.lock` 都在版本控制中。

## CLI（Release / 演示）

`pinn-starlight` 是演示与发布用的单图命令行，不是正式实验入口。两种调用方式等价：

```bash
python -m experiments.cli estimate --image photo.jpg --mode star
pinn-starlight estimate --image photo.jpg --mode star   # pip install -e . 后可用
```

常用子命令：

```bash
python -m experiments.cli help
python -m experiments.cli version
python -m experiments.cli doctor --json
python -m experiments.cli samples --stage a
python -m experiments.cli estimate --sample clean_single_002_multi_source_high --methods all
```

- `--mode star|nebula|balanced`：星点（SGC）、星云（joint PINN）、平衡（physics-first + 锚定，实验性）。
- `--methods all` 跑四种论文方法；也可以只跑 `fft_gaussian,fourier_ls,pinn,final_pinn,balanced` 中的子集。
- 输出写到 `--out/<sample>/`：`observed.tif`、每方法 `background.tif`/`residual.tif`、`preview.png`、`summary.json`。
- 照片没有真值，只输出描述性指标；合成样本用冻结星表，但仍是 CPU/demo 预算，**不是论文数字**。
- 退出码：0 成功、2 用法、3 环境、4 输入、5 方法失败。

### 分发包（portable bundle）

推荐的分发方式是**自包含目录**：内置解释器与依赖，用户解压后直接运行，**不需要安装 Python**。

```bash
python tools/build_release.py --portable
```

产物：

```text
build/dist/pinn-starlight-0.2.0-linux-x86_64/
├── python/              # 内置 CPython（可重定位，拷贝而非符号链接）
├── site-packages/       # 依赖，CPU torch 来自 download.pytorch.org/whl/cpu
├── run.py
├── pinn-starlight.sh    # Linux 启动脚本
├── pinn-starlight.bat   # Windows 启动脚本
└── README.txt
```

要点：

- **不用 venv**。`python -m venv` 会把绝对路径写进控制台脚本的 shebang 和 `pyvenv.cfg`，
  目录一移动就 `bad interpreter`。这里用内置解释器 + `pip --target`。
- `build_release.py` 的最后一步会把整个目录**复制到临时目录再运行**，专门用来抓"搬不走"。
- 每个平台各自构建：同一脚本在 Linux 上产出 linux 目录，在 Windows 上产出 windows 目录
  （启动器与目录名自动切换）。**macOS 未验证，不提供。**
- `--skip-deps` 只搭结构、不下载依赖，只验证 `version` / `doctor`。

### wheel（给已有 Python 环境的人）

wheel 本身约 200 KB，`version` / `doctor` / `samples` 只依赖标准库。但
**`[project].dependencies` 目前仍列着完整研究依赖**（jupyter、lightning、astropy 等），
`pip install` 会把它们一起装上，而 `torch` 仍需单独安装。这一点在上 PyPI 前必须先改。

```bash
python tools/build_release.py    # 构建 wheel/sdist，并在干净 venv 里自检降级行为
```

自检会确认：无依赖环境下 `version` 正常、`doctor` 退出 3 而不是抛 traceback、
`estimate` 给出安装提示而不是崩溃。

## 运行正式实验

Linux：

```bash
./run_stage2.sh a       # Stage-A：数据检查 -> 选择 -> 配置锁 -> E2 -> E3 -> Final PINN
./run_stage2.sh b       # Stage-B 压力测试
./run_stage2.sh cv5     # 5-fold 分组评估
./run_stage2.sh e4      # 真实高污染图像四方法对比
./run_stage2.sh all     # A -> B -> CV5 -> E4
```

仓库只提供 `run_stage2.sh`；Windows 上直接调用 `python run_stage2.py a`（或 pixi 环境中的等价命令）。

`run_stage2.sh` 会先检查 GPU、驱动与 Pixi 环境。每个科学子进程只看到一个 GPU（内部仍使用 `cuda:0`），调度策略的完整说明在 `run_stage2.py` 顶部。也可以直接调用模块，例如：

```bash
pixi run -e remote python -m experiments.scripts.protocol.run \
    --stage a --mode standard --phase all --device cuda:0 --include-ablation
```

## 冒烟测试（无需 GPU）

正式实验需要 GPU 和约 11 GB 合成数据，但整套科学代码路径可以在 CPU 上用小预算跑通：

```bash
python -m tools.smoke_test              # 全部 17 项，约 25 秒
python -m tools.smoke_test --list       # 列出测试项
python -m tools.smoke_test --only pinn  # 只跑某几项
```

它覆盖 data → 背景估计 → PINN → SGC-PINN → metrics 的完整链路，输出只用于证明代码可执行、内部自洽、数组有限，**不是科学结果**。退出码为 0 才算通过。

缺少样本数据时，先用 `bash tools/fetch_smoke_data.sh` 从归档解出约 280 MB 的最小数据集。

## 机制诊断（非正式，validation only）

`experiments/scripts/diagnostics/` 下的脚本回答机制问题，不进入论文主流程，也不替代 Stage-A/B/CV5/E4：

```bash
python -m experiments.scripts.diagnostics.init_memory --suite smoke   # 初始化优势能否保留 / 锚定强度
python -m experiments.scripts.diagnostics.alpha_sweep --suite smoke   # 哪个 alpha 让物理损失降得最快
python -m experiments.scripts.diagnostics.mode_eval --suite smoke     # star / nebula / balanced 的权衡
```

- 只读 validation split；不修改正式 config / lock / seed / 预算。
- 输出写到 `experiments/outputs/diagnostics/`（已忽略）；`smoke` 只验证管道，`full` 才是声明过的诊断预算。
- 协议、预算与读数注意事项见 [experiments/scripts/diagnostics/README.md](experiments/scripts/diagnostics/README.md)。

## 数据与输出

- 默认数据根是仓库根目录下的 `collections/` 与 `test/`；也可以设置 `PINN_STARLIGHT_DATA_ROOT` 指向包含这两个目录的位置。
- 正式输出写到 `experiments/outputs/`（已忽略）；CV5 写到 `experiments/outputs/cv5/fold_<n>/`。
- 数据生成、split 规则与候选空间都集中在 `experiments/configs/`。

## 论文图与工具

- 图 3/4/6 的收集与渲染脚本在 `scripts/figures/`。
- 服务器归档导出/恢复与离线重建：见 `tools/README.md`。
- `paper_figures/` 中是已生成结果；重新渲染请用脚本，不要手工修改图片。

## 仓库约定

- `__pycache__`、`.pytest_cache`、`.ruff_cache`、`*.egg-info/` 等生成物不入库。
- `experiments/outputs/`、`logs/`、`archived` 不入库。
- `scripts/figures/legacy/` 仅保留历史版本，当前流程不使用。
