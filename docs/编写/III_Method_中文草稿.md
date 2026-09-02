# III. Method（中文草稿）

> 这是一份根据当前代码、数学推导和实验配置整理的中文方法章节草稿。正式投稿前仍需补充参考文献、图号、章节编号，并由代码结果再次核对。

## A. Overview

本章介绍一种基于物理启发约束的单幅星空图像光污染背景估计方法。给定一幅归一化灰度观测图像，方法以像素坐标作为输入，通过坐标多层感知机（multilayer perceptron, MLP）建立从图像位置到背景亮度的连续映射，并输出光污染背景估计。为了减少网络对局部星点等高频结构的拟合，训练目标同时包含观测数据项和 screened Poisson 方程残差项。

需要强调的是，本文研究对象是光污染背景估计，而不是一般性的天体信号分离。背景扣除后的图像仅作为残差信号进行分析，其星点保留能力通过合成数据实验评价。

## B. Observation Model

一幅观测到的星空图像可近似表示为星光信号与光污染背景之和：

$$
I_{obs}(x,y)=I_{star}(x,y)+I_{bg}(x,y),
\tag{1}
$$

其中，$I_{obs}$ 表示输入观测图像，$I_{star}$ 表示星点及其他天体信号，$I_{bg}$ 表示由人工照明经大气散射形成的空间变化背景。由于单幅图像中气辉、银河、薄云和暗角等结构也可能呈现缓慢变化，该分解只是一种建模近似，并不保证各成分在真实图像中严格可辨识。

本文的网络不直接预测总观测亮度，而是拟合背景函数：

$$
\widehat I_{bg}(x,y)=f_\theta(x,y),
\tag{2}
$$

其中 $f_\theta$ 为参数为 $\theta$ 的坐标网络。完成背景估计后，背景扣除残差定义为

$$
\widehat I_{star}(x,y)=I_{obs}(x,y)-\widehat I_{bg}(x,y).
\tag{3}
$$

式（3）表示背景扣除结果，不代表方法已经从单幅图像中完全恢复了不可见的星光，也不保证残差中不包含噪声或被误扣除的背景结构。

## C. Physics-Inspired Background Constraint

### 1) Screened Poisson model

光污染背景通常比星点具有更大的空间尺度和更缓慢的变化趋势。为表达这种先验，本文采用以下 screened Poisson 型约束：

$$
\nabla^2 I_{bg}(x,y)-\alpha I_{bg}(x,y)+I_{city}(x,y)=0,
\qquad \alpha>0,
\tag{4}
$$

其中，$\nabla^2$ 是二维拉普拉斯算子，$I_{city}$ 表示结构化的光污染源项，$\alpha$ 是在大尺度近似下引入的正系数。本文借鉴 Garstang 夜天光模型中与光源方向和距离相关的函数形式，但式（4）并不是对完整三维大气辐射传输过程的严格等价求解。

在归一化图像坐标下，正式实验固定 $\alpha=0.5$。因此，$\alpha$ 在本文中是用于保持简化模型数值稳定性的有效模型系数，不将其解释为从图像中准确反演得到的真实大气常数，也不直接把它称为背景的空间尺度。

### 2) Structured source term

为描述主要光污染源的位置、范围和方向，本文使用一个带椭圆包络的结构化源项。首先定义光源中心 $(x_c,y_c)$ 到像素坐标 $(x,y)$ 的距离：

$$
r=\sqrt{(x-x_c)^2+(y-y_c)^2+\varepsilon},
\tag{5}
$$

其中 $\varepsilon$ 是用于提高数值稳定性的微小常数。源项的径向变化采用物理启发的距离相关函数：

$$
b(r;\alpha)=\cos(\sqrt{\alpha}r)+\alpha^2r^4.
\tag{6}
$$

对式（6）代入二维径向拉普拉斯算子，可构造相应的径向源项：

$$
q(r;\alpha)=\alpha b(r;\alpha)-\nabla^2 b(r;\alpha).
\tag{7}
$$

为了允许光污染区域具有非圆形范围和方向，定义旋转椭圆高斯包络：

$$
\begin{aligned}
x'&=(x-x_c)\cos\theta+(y-y_c)\sin\theta,\\
y'&=-(x-x_c)\sin\theta+(y-y_c)\cos\theta,
\end{aligned}
\tag{8}
$$

$$
G_\phi(x,y)=\exp\left[-\frac12\left(\frac{x'^2}{\sigma_x^2}+\frac{y'^2}{\sigma_y^2}\right)\right],
\tag{9}
$$

其中，$\phi=\{x_c,y_c,\sigma_x,\sigma_y,\theta\}$ 表示源项参数，$\sigma_x$ 和 $\sigma_y$ 控制椭圆范围，$\theta$ 控制旋转方向。当前实现没有额外的独立幅度参数，最终的结构化光污染源项写为

$$
I_{city}(x,y)=q(r;\alpha)G_\phi(x,y).
\tag{10}
$$

该构造只借用了 Garstang 模型的部分函数结构，并通过椭圆包络提供有限的空间适应能力。加入包络后，$I_{city}$ 是可学习的结构化源项，而不是某个解析背景函数的严格闭式解。单个源项对多光源、复杂天际线和非均匀薄云的表达能力有限。

## D. Coordinate MLP

### 1) Input and output

对输入图像进行线性灰度化和统一归一化后，将每个像素的位置映射到归一化坐标：

$$
(x,y)\in[-1,1]^2.
\tag{11}
$$

网络输入为坐标向量 $\mathbf a_0=[x,y]^\mathsf T$，输出为该位置的单通道背景亮度 $\widehat I_{bg}(x,y)$。因此，网络参数量不随图像像素数量线性增加，输出可以视为定义在连续坐标域上的背景函数。

### 2) Network architecture

隐藏层使用 `Tanh` 激活函数，输出层使用线性映射。对于第 $l$ 个隐藏层，有

$$
\mathbf z_l=W_l\mathbf a_{l-1}+\mathbf b_l,
\qquad
\mathbf a_l=\tanh(\mathbf z_l).
\tag{12}
$$

最终输出为

$$
\widehat I_{bg}(x,y)=W_L\mathbf a_{L-1}+\mathbf b_L.
\tag{13}
$$

由于 `Tanh` 关于输入坐标是光滑可导的，网络输出可以通过自动微分计算二阶偏导数。E1 调参后，正式实验采用隐藏层结构 `[512]`、`kernel_size=31`、`steps=3000` 和 `center_mode=bright_init_fixed`。候选结构和其他超参数只在验证集上进行选择。

## E. Joint Optimization Objective

### 1) Data term

从图像中采样 $N$ 个坐标点及对应观测亮度，数据一致性损失定义为

$$
\mathcal L_{data}=\frac1N\sum_{i=1}^{N}
\left(\widehat I_{bg}^{(i)}-I_{obs}^{(i)}\right)^2.
\tag{14}
$$

该项使背景估计贴近输入观测，但由于输入中同时含有星点，它也可能促使网络吸收部分局部星光。因此，数据项本身不能保证星点完全保留。

### 2) PDE residual term

将网络输出代入式（4），得到 PDE 残差

$$
R_\theta(x,y)=\nabla^2\widehat I_{bg}(x,y)
-\alpha\widehat I_{bg}(x,y)+I_{city}(x,y).
\tag{15}
$$

物理约束损失定义为残差的均方误差：

$$
\mathcal L_{physics}=\frac1N\sum_{i=1}^{N}R_\theta(x_i,y_i)^2.
\tag{16}
$$

本文未额外施加独立的边界条件或边界损失，因此式（16）应理解为内部点上的物理启发正则，而不是完整边值问题的求解。

### 3) Total loss

总训练目标为

$$
\mathcal L=\mathcal L_{data}+\lambda\mathcal L_{physics},
\tag{17}
$$

其中 $\lambda$ 控制数据拟合与物理先验之间的权衡。正式配置使用 $\lambda=0.5$，即 `physics_weight=0.5`。较大的物理权重并不必然带来更好的背景或星点指标，其影响需要通过验证集和消融实验单独报告。

## F. Differentiation and Optimization

网络关于坐标的一阶和二阶导数均由 PyTorch 自动微分计算。首先对 $\widehat I_{bg}$ 关于 $(x,y)$ 求一阶导数，再对一阶导数继续求导，得到

$$
\nabla^2\widehat I_{bg}
=\frac{\partial^2\widehat I_{bg}}{\partial x^2}
+\frac{\partial^2\widehat I_{bg}}{\partial y^2}.
\tag{18}
$$

随后根据式（15）--（17）计算总损失，并通过 Adam 更新网络参数 $\theta$ 以及结构化源项参数 $\phi$。一次训练迭代包括以下步骤：

1. 从输入图像读取坐标和观测亮度；
2. 通过坐标 MLP 计算背景预测；
3. 通过自动微分计算背景预测的拉普拉斯项；
4. 计算结构化源项、PDE 残差和总损失；
5. 反向传播并更新 $\theta$ 和 $\phi$。

这里的自动微分只负责自动计算导数，不会取消反向传播本身。

## G. Inference Procedure

训练完成后，在整幅归一化坐标网格上逐点计算 $\widehat I_{bg}$，再根据式（3）得到背景扣除残差。输出包括观测图、背景估计和残差图，同时保存浮点数组以支持后续指标计算。对于合成数据，背景估计可与已知的 `background_true` 比较；对于真实图像，由于没有可靠的背景真值，只报告背景均值、总变差、残差统计和运行时间，并结合可视化进行定性分析。

## H. Scope and Limitations

本文方法的目标是验证物理启发正则在单幅光污染背景建模中的可行性，而不是构建完整的大气辐射传输模拟器。其主要限制包括：

1. 单幅图像的背景与天体结构存在不可辨识性，平滑的气辉、银河、薄云和暗角可能被混入背景；
2. 单个椭圆源项难以表示多个光源或复杂地平线；
3. 数据项直接拟合混合观测，PDE 只能抑制而不能理论上消除星点吸收；
4. 当前没有显式边界条件；
5. PINN 需要逐图优化，计算成本高于无需训练的 FFT 基线和已训练的 U-Net 推理；
6. 正式实验中的 $\alpha=0.5$ 是归一化坐标下的有效模型系数，不是可直接测量的大气物理常数。

因此，后续实验应重点比较背景重建、残差信号保留、通量误差和计算成本，并避免将残差直接表述为已经完成的一般性天体信号分离。
