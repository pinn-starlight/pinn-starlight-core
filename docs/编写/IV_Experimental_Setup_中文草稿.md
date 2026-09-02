# IV. Experimental Setup（中文草稿）

> 本章根据当前 `experiments/outputs` 中的配置、manifest 和运行环境整理。正式论文中应将标题、图表编号和引用改为 IEEE 格式，并在提交前再次核对数据划分。

## A. Research Questions

本文围绕以下三个问题开展实验：

**RQ1：** 与 FFT-Gaussian 和 U-Net-small 相比，PINN 的光污染背景估计和星点保留效果如何？

**RQ2：** PDE 约束、亮区初始化以及可学习的光污染源点中心是否能够改善估计结果？

**RQ3：** 三种方法在真实星空图像上会产生怎样的背景、残差和典型伪影？

实验顺序固定为 E1--E4。E1 只用于选择 PINN 配置；配置锁定后，再进行 E2 主对比、E3 消融和 E4 真实图像分析。真实图像不参与超参数选择。

## B. Synthetic and Real Data

### 1) Synthetic data generation

合成数据由干净基础星图、模拟光污染背景和随机噪声组成：

$$
I_{obs}=I_{clean}+I_{bg}^{true}+n,
\tag{19}
$$

其中 $I_{clean}$ 是干净星图，$I_{bg}^{true}$ 是已知的背景真值，$n$ 是零均值高斯噪声。每个样本保存 `clean_true`、`background_true`、`observed` 以及背景源项参数和随机种子。

当前 manifest 共包含 5 张基础星图，每张基础图生成 3 种背景形态和 2 个污染强度等级，共 30 个样本变体：

| 项目 | 设置 |
| --- | --- |
| 背景形态 | `single_radial`、`single_eccentric`、`multi_source` |
| 强度等级 | `low`：强度 0.10，噪声标准差 0.005；`high`：强度 0.20，噪声标准差 0.015 |
| 背景尺度 | `low` 为 0.55，`high` 为 0.85 |
| 图像范围 | 线性灰度、归一化到 `[0,1]` |
| 预处理 | RAW 图像下采样 2 倍；当前未使用中心裁剪 |

`single_radial` 使用单个近似圆对称光源，`single_eccentric` 使用偏心的椭圆光源，`multi_source` 使用两个光源叠加。背景生成函数参考 Kocifaj 2025 的散射建模思路，但只保留二维有限光源、透射、相函数和几何衰减等因素，不构成完整三维辐射传输模拟。

### 2) Data split

样本首先按基础星图划分，再生成污染变体，以避免同一基础星图的不同污染版本跨集合泄漏：

| 划分 | 基础星图 | 样本数 |
| --- | --- | ---: |
| train | `clean_single_003`、`clean_stack_002`、`clean_stack_001` | 18 |
| validation | `clean_single_002` | 6 |
| test | `clean_single_001` | 6 |

需要记录一个当前结果中的协议差异：E2 的 `locked_e2_config.json` 列出的六个固定评估样本是 `clean_stack_002` 的三个背景形态和两个强度等级，而 manifest 将 `clean_stack_002` 标记为 `validation`。因此，本文在未重新整理划分前应将 E2 结果称为“固定评估样本上的比较”，不应直接称为独立测试集结果。

合成数据中的星点参考来自 `clean_true` 上一次性提取的局部对比极大值，阈值为 `0.03`，匹配半径为 3 像素。该参考用于统一比较，不等同于外部星表标注。

### 3) Real data

E4 使用 16 张真实图像，包括 `data/test/native_test.tif` 和 `real_high_001` 至 `real_high_015` 共 15 张高污染候选图。真实图像只用于背景、残差和运行成本的无参考统计及定性观察，因为没有可靠的背景真值，不计算真实图像的背景 MAE、PSNR 或 SSIM。

## C. Compared Methods

本文比较三个方法：

1. **FFT-Gaussian**：对归一化灰度图进行二维高斯低通，使用反射填充处理边界；E2 和 E4 使用验证集锁定的归一化尺度 $\sigma=0.02$。
2. **U-Net-small**：输入单通道观测图，输出背景估计图；使用合成配对数据进行监督训练，背景损失为 MSE，验证集早停。网络基础通道数为 16，训练 20 个 epoch，每个 epoch 100 个 step，batch size 为 4，patch size 为 $256\times256$，patch overlap 为 32。
3. **PINN**：每张输入图像独立优化坐标 MLP 和结构化源项，使用 E1 锁定配置。其数据项和 PDE 残差项均采用 MSE，不使用跨图像训练。

三种方法的计算范式不同，因此分别记录 FFT 的单图处理时间、U-Net 的离线训练时间和单图推理时间，以及 PINN 的单图优化时间。不能把三者的训练时间简单视为同一种成本。

## D. PINN Configuration

E1 选择并锁定的 PINN 配置如下：

| 参数 | 值 |
| --- | --- |
| 隐藏层 | `[512]`，`Tanh` 激活，线性输出 |
| `physics_weight` | 0.5 |
| `kernel_size` | 31（用于亮区模糊和中心初始化） |
| 优化步数 | 3000 |
| batch size | 8192 |
| MLP 学习率 | $1\times10^{-3}$ |
| 源项学习率 | $1\times10^{-3}$ |
| $\alpha$ | 固定为 0.5 |
| 中心模式 | `bright_init_fixed` |
| 预测 batch size | 65536 |

所有正式 PINN 运行使用随机种子 `20260728`、`20260729` 和 `20260730`，并记录 CPU/GPU、PyTorch、CUDA 版本和代码 commit。当前正式输出在 NVIDIA GeForce RTX 3090、PyTorch 2.12.1 和 CUDA 12.6 环境下生成。

## E. Experimental Protocol

### 1) E1: Configuration selection

E1 在固定验证数据上采用单变量顺序选择：先比较隐藏层结构，再比较物理损失权重、中心初始化模糊核大小和训练步数。所有候选使用相同的 $\alpha=0.5$、预处理和数据划分。选择优先级为背景 MAE，其次为残差质量、星点指标和运行稳定性。

### 2) E2: Main comparison

E2 在固定的六个合成评估样本上比较 FFT-Gaussian、U-Net-small 和 PINN。每个方法输出 `background_pred` 和 `residual_pred`，并从浮点数组计算背景重建、残差和星点指标。FFT 为确定性方法，PINN 和 U-Net 使用三个随机种子；结果报告均值和标准差，同时保留按背景形态和强度等级的分组结果。

### 3) E3: Ablation

E3 固定 E1 的网络、优化器、步数和 $\alpha$，只改变 PDE、中心初始化和中心是否可学习：

| 变体 | 设置 | 目的 |
| --- | --- | --- |
| `data_only` | `physics_weight=0` | 测试 PDE 项的作用 |
| `center_fixed` | 中心固定在坐标原点 | 测试无亮区定位先验时的表现 |
| `bright_init_fixed` | 由模糊亮区初始化并冻结中心 | 测试亮区初始化的作用 |
| `bright_init_learnable` | 相同初值，但训练中学习中心 | 测试中心自由调整是否有额外收益 |

中心误差只在单源偏心子集上计算：

$$
E_{center}=\sqrt{(\widehat x_c-x_c)^2+(\widehat y_c-y_c)^2}.
\tag{20}
$$

多光源样本不报告单一中心误差。

### 4) E4: Real-image analysis

E4 使用 E2 锁定的 FFT 参数、U-Net checkpoint 和 PINN 配置。所有图像统一下采样 2 倍，默认保留整幅图。每张图生成观测图、三种方法的背景图和残差图，并检查以下现象：大尺度梯度残留、星点被背景吸收、银河或薄云被误判、边缘伪影、负值以及原图不存在的新增亮点。

## F. Evaluation Metrics

### 1) Synthetic metrics

对于合成数据，背景预测与 `background_true` 比较，残差预测与 `clean_true` 比较：

- 背景：MAE、RMSE、PSNR、SSIM；
- 残差：MAE、PSNR、SSIM；
- 星点：Precision、Recall、F1；
- 光度：平均相对通量误差；
- 资源：训练时间、单图时间、峰值显存和参数量。

PSNR 和 SSIM 在 `[0,1]` 范围内计算。星点阈值和匹配半径在实验前固定，不根据测试结果调整。

### 2) Real-image proxy statistics

真实图像没有背景真值，因此只报告：背景均值、背景总变差、残差均值、残差标准差、负残差比例、检测到的星点数量和运行时间。这些统计描述输出特征和计算成本，不能单独证明某一种方法的真实背景更准确。

## G. Reproducibility and Reporting

每次正式实验保存配置、随机种子、代码 commit、运行环境、逐图指标、汇总统计、浮点数组和展示图。论文结果应以 `summary.json`、`metrics.csv`、`main_result_table.csv` 和 `ablation_table.csv` 为数字来源；PNG 只用于可视化，不用于重新计算指标。

由于三种方法的训练范式、样本数量和随机性不同，结果主要用于趋势比较。报告中应同时给出样本数量、均值和标准差，并明确 E2 当前固定评估样本与 manifest 划分之间的差异。
