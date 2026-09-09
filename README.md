# PINN Starlight

基于物理约束神经网络（PINN）的单幅星空图像光污染背景估计研究原型。

项目使用坐标 MLP 表示连续的光污染背景，并通过 screened Poisson 方程残差引入平滑与空间结构先验。估计背景从观测图中扣除后得到残差图，用于分析星点保留和背景分离效果。

> [!IMPORTANT]
> 本项目仍处于研究与实验阶段。输出的残差不等同于严格恢复的真实天体信号；当前 PDE 是物理启发正则，而不是完整的大气辐射传输模型。

## 方法概览

对于归一化灰度观测图像，采用加性观测模型：

$$
I_{obs}=I_{star}+I_{bg}.
$$

坐标网络以 $(x,y)\in[-1,1]^2$ 为输入，预测单通道背景 $\widehat I_{bg}$。训练时使用数据项与物理项的联合损失：

$$
\mathcal L=\mathcal L_{data}+\lambda\mathcal L_{physics},
$$

$$
\mathcal L_{data}=\operatorname{MSE}(\widehat I_{bg},I_{obs}),
$$

$$
\mathcal L_{physics}=\operatorname{MSE}
\left(\nabla^2\widehat I_{bg}-\alpha\widehat I_{bg}+I_{city},0\right).
$$

其中，$I_{city}$ 是带中心、横纵尺度和方向的单源椭圆包络，二阶导数由 PyTorch `autograd` 计算。正式实验固定 $\alpha=0.5$，背景扣除结果定义为：

$$
\widehat I_{star}=I_{obs}-\widehat I_{bg}.
$$

## 主要内容

- 坐标 MLP 光污染背景估计
- screened Poisson 物理残差与结构化 $I_{city}$ 源项
- RAW、TIFF、PNG、JPEG 等图像读取及线性灰度预处理
- FFT-Gaussian、U-Net-small 与 PINN 三种方法的统一比较
- 合成数据定量评估、PINN 消融和真实图定性评估
- 配置、指标、预测数组、可视化和运行环境的统一保存

## 环境要求

- Python 3.11 或更高版本
- [uv](https://docs.astral.sh/uv/)
- 推荐使用支持 CUDA 的 NVIDIA GPU；没有 CUDA 时会自动使用 CPU，但 PINN 训练可能较慢

项目的 PyTorch 软件源当前指向 CUDA 12.6。首次安装依赖：

```powershell
uv sync
```

当前项目采用 `src` 布局，但尚未配置为可安装包。运行命令前需要在项目根目录设置 `PYTHONPATH`。

PowerShell：

```powershell
$env:PYTHONPATH="src;."
```

Bash：

```bash
export PYTHONPATH="src:."
```

## 快速开始

先运行 E0 冒烟测试，检查图像加载、指标计算以及三种方法的最小流程：

```powershell
uv run python -m experiments.scripts.e0.run
```

E0 需要以下数据：

- `data/test/` 中至少一张真实测试图
- `data/collections/synthetic/` 中至少两个合成样本

如尚未生成合成数据，先准备 `data/collections/manifest.csv` 及其中登记的干净候选图，再执行：

```powershell
uv run python -m experiments.common.data.generate_synthetic
```

冒烟测试结果写入 `experiments/outputs/e0/`。

## 正式实验

正式实验必须按顺序运行。E2–E4 会读取前序阶段锁定的配置或模型：

| 阶段 | 命令 | 作用 | 主要产物 |
| --- | --- | --- | --- |
| E0 | `uv run python -m experiments.scripts.e0.run` | 全流程冒烟测试 | 三种方法的示例背景图与残差图 |
| E1 | `uv run python -m experiments.scripts.e1.run` | 在验证集选择 PINN 配置 | `locked_pinn_config.json` |
| E2 | `uv run python -m experiments.scripts.e2.run` | 合成数据主对比 | 主结果表、U-Net checkpoint、`locked_e2_config.json` |
| E3 | `uv run python -m experiments.scripts.e3.run` | PDE 与源点中心消融 | 消融表和逐样本结果 |
| E4 | `uv run python -m experiments.scripts.e4.run` | 真实图对比 | 背景/残差对照图和人工检查清单 |

所有实验输出位于 `experiments/outputs/`。脚本会保存浮点数组、展示图、CSV 指标、JSON 配置及运行元数据。部分阶段默认拒绝覆盖已有非空输出；需要重跑时，请先确认结果已经备份，再修改对应脚本顶部的 `FORCE_OVERWRITE_OUTPUT` 或输出目录配置。

`src/pinn_starlight_core/main.py` 仍保留早期单图训练入口，其中包含可学习 `alpha` 的原型实现：

```powershell
uv run python -m pinn_starlight_core.main
```

该入口读取 `data/test/`，只打印最终损失和源项参数，不属于 E0–E4 正式评估流程。

## 实验协议

正式实验包含三类方法：

| 方法 | 计算方式 |
| --- | --- |
| FFT-Gaussian | 频域高斯低通背景估计 |
| U-Net-small | 在合成配对数据上监督训练，随后分块推理 |
| PINN | 对每张图像独立优化坐标网络与结构化源项 |

合成样本由 `clean_true`、`background_true` 和 `observed` 组成：

```text
observed = clip(clean_true + background_true, 0, 1)
```

合成实验评估背景 MAE/RMSE/PSNR/SSIM、残差 MAE/PSNR/SSIM、星点 Precision/Recall/F1 和光通量误差。真实图没有可靠的背景真值，因此只进行定性对照和无参考统计，不报告全参考 PSNR/SSIM。

## 项目结构

```text
pinn-starlight-core/
├── src/pinn_starlight_core/
│   ├── data/image_loader.py          # 图像读取、灰度转换与坐标网格
│   ├── nn/pinn_layers.py             # 坐标 MLP
│   ├── nn/physics_model.py           # alpha 与结构化 I_city
│   ├── nn/pinn_loss.py               # 数据损失、PDE 残差与拉普拉斯算子
│   └── main.py                       # 早期单图训练入口
├── experiments/
│   ├── common/baselines/             # FFT、U-Net-small、PINN 共用实现
│   ├── common/data/                  # 合成数据生成
│   ├── common/utils/                 # 指标、评分与实验工具
│   └── scripts/e0...e4/              # 分阶段实验入口
├── docs/论文准备资料/                # 方法、实现和实验协议
├── data/                             # 本地数据，默认不纳入 Git
├── pyproject.toml
└── README.md
```

## 已知限制

- 单幅图中的气辉、银河、薄云、暗角等平滑结构可能与光污染混淆。
- 数据损失直接拟合混合观测，网络仍可能吸收部分星光。
- 当前没有独立边界条件或边界损失。
- 单个椭圆源项难以完整表达多光源、复杂天际线和薄云。
- 当前仅处理灰度亮度；尚未建立逐通道 RGB 物理模型。
- 合成背景是简化的解析平滑背景，不代表完整的大气散射过程。

## 文档

- [阅读入口](docs/论文准备资料/00_从这里开始.md)
- [数学与方法](docs/论文准备资料/01_数学与方法/README.md)
- [技术事实交接单](docs/论文准备资料/02_实现与交接/技术事实交接单.md)
- [论文实验计划](docs/论文准备资料/03_实验/论文实验计划.md)

## 参考文献

- Garstang, R. H. (1986). *Model for artificial night-sky illumination*.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics, 378*, 686–707.

## License

当前仓库未包含独立的许可证文件。若计划公开发布或复用代码，请先补充并确认许可证。
