# PINN 数学原理

## 1. 物理基础

### 1.1 光污染约束方程

光污染背景采用以下物理启发约束：

$$
\nabla^2 I_{bg}-\alpha I_{bg}+I_{city}=0.
$$

其中：

- $I_{bg}$ 表示平滑的光污染背景；
- $\alpha$ 表示屏蔽系数；
- $I_{city}$ 表示结构化光污染源项。

该方程的详细来源与适用范围见[光污染方程简化步骤](./光污染方程简化步骤.md)。

### 1.2 观测图像分解

观测图像近似分解为星光与光污染背景之和：

$$
I_{obs}=I_{star}+I_{bg}.
$$

网络不预测总观测亮度，而是直接输出光污染背景估计

$$
\widehat I_{bg}(x,y)=f_\theta(x,y)\approx I_{bg}(x,y).
$$

训练完成后可用

$$
\widehat I_{star}=\max\left(I_{obs}-\widehat I_{bg},0\right)
$$

得到残差信号。

### 1.3 损失函数

定义 PDE 残差

$$
R_\theta(x,y)
=\nabla^2 \widehat I_{bg}(x,y)
-\alpha \widehat I_{bg}(x,y)
+I_{city}(x,y).
$$

本文使用均方误差约束背景估计与观测亮度的一致性：

$$
\mathcal{L}_{data}
=\frac{1}{N}\sum_{i=1}^{N}
\left(\widehat I_{bg}^{(i)}-I_{obs}^{(i)}\right)^2,
$$

物理损失同样对 PDE 残差使用均方误差：

$$
\mathcal{L}_{physics}
=\frac{1}{N}\sum_{i=1}^{N}R_\theta(x_i,y_i)^2.
$$

总损失为

$$
\mathcal{L}
=\mathcal{L}_{data}
+\lambda\mathcal{L}_{physics},
$$

其中 $\lambda$ 控制物理约束的权重。

Raissi 等人的经典 PINN 将数据误差与 PDE 残差都写为均方误差。本文同样采用这一联合损失形式：数据项使背景估计贴近观测亮度，PDE 残差则约束其满足平滑且具有空间结构的背景模型。由于 $I_{obs}=I_{star}+I_{bg}$，该联合优化不能理论上保证完全不拟合星点；其实际分离效果需要通过合成数据和真实图像实验评估。

## 2. 网络正向计算

输入坐标为

$$
\mathbf{a}_0=
\begin{bmatrix}
x_i\\
y_i
\end{bmatrix}.
$$

对第 $l$ 个隐藏层，有

$$
\mathbf{z}_l=W_l\mathbf{a}_{l-1}+\mathbf{b}_l,
\qquad
\mathbf{a}_l=\tanh(\mathbf{z}_l),
\qquad l=1,\ldots,L-1.
$$

输出层采用线性变换：

$$
\widehat I_{bg}(x_i,y_i)
=W_L\mathbf{a}_{L-1}+\mathbf{b}_L.
$$

使用平滑的 `Tanh` 激活函数，可以保证网络输出关于输入坐标二阶可导，从而计算拉普拉斯算子。

## 3. 自动微分

本项目不需要手工编写逐层反向传播公式。PyTorch 的 `autograd` 在训练过程中承担两类不同但相关的求导工作。

### 3.1 对输入坐标求导

物理损失需要计算

$$
\nabla^2 \widehat I_{bg}
=\frac{\partial^2 \widehat I_{bg}}{\partial x^2}
+\frac{\partial^2 \widehat I_{bg}}{\partial y^2}.
$$

程序先对 $\widehat I_{bg}$ 关于 $(x,y)$ 求一阶导数，再对一阶导数继续求导。计算一阶导数时保留计算图，才能继续得到二阶导数。

### 3.2 对可学习参数求导

构造总损失后，调用 `loss.backward()`，PyTorch 会自动计算

$$
\frac{\partial\mathcal{L}}{\partial\theta},
\qquad
\frac{\partial\mathcal{L}}{\partial\alpha},
\qquad
\frac{\partial\mathcal{L}}{\partial\phi_{city}},
$$

其中 $\theta$ 是 MLP 参数，$\phi_{city}$ 是 $I_{city}$ 模块中的可学习参数。随后优化器根据这些梯度更新参数。

因此，**反向传播仍然发生，但由自动微分框架完成，无需在文档或代码中手工展开每一层的链式法则**。

## 4. 单次训练迭代

一次训练迭代可概括为：

1. 从图像中采样坐标 $(x_i,y_i)$ 与观测亮度 $I_{obs}^{(i)}$；
2. 通过 MLP 正向计算 $\widehat I_{bg}^{(i)}$；
3. 使用自动微分计算 $\nabla^2 \widehat I_{bg}^{(i)}$；
4. 计算 $I_{city}$、PDE 残差和总损失；
5. 调用 `loss.backward()` 自动计算参数梯度；
6. 调用优化器更新 MLP、$\alpha$ 和 $I_{city}$ 的可学习参数。

自动微分避免了手工推导高阶复合导数，但不会取消反向传播本身。
