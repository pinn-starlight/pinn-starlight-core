# PINN Starlight

> 基于物理约束的低噪声星空图像光污染背景建模与天体信号分离

PINN Starlight 是一个面向低噪声星空图像的研究型原型项目。它使用物理信息神经网络（PINN）对光污染背景进行建模，并尝试将其与真实天体信号分离。

当前版本聚焦于三件事：

1. 基础 PINN 光污染分离主链路
2. 可学习的 `I_city` 光污染源项
3. 有界可学习的 `alpha` PDE 系数

---

## 当前版本：V3

V3 的重点是把原先固定的 PDE 系数 `alpha` 改成**受约束的可学习参数**，并让它在训练中同时作用于：

- 物理损失 `∇²I - αI + I_city`
- `I_city` 的解析公式

当前代码已经具备：

- 基于坐标输入的 MLP 背景建模
- 数据损失 + 物理损失联合训练
- `autograd` 二阶微分拉普拉斯
- 基于模糊亮区初始化的可学习 `I_city`
- 有界可学习 `alpha` 模块
- 真实图像加载（RAW / PNG / JPG / TIFF）
- 训练后输出观测图、预测图、残差图

---

## 方法概览

输入为低噪声星空图像，输出为每个像素位置上的光污染背景估计值。

```text
输入图像
  ↓
坐标网格 (x, y) + 像素亮度 I_obs
  ↓
MLP 预测背景 I_pred(x, y)
  ↓
物理约束：∇²I_pred - αI_pred + I_city = 0
  ↓
残差：I_star = I_obs - I_pred
```

其中：

- `I_pred`：网络预测的光污染背景
- `I_city`：结构化光污染源项
- `alpha`：受约束可学习 PDE 系数
- `I_star`：从原图中分离出的残差信号

---

## 项目结构

```text
pinn-starlight-core/
├── src/pinn_starlight_core/
│   ├── data/
│   │   ├── FakeRAW.py
│   │   └── PhotoLoader.py
│   ├── nn/
│   │   ├── Alpha.py
│   │   ├── Icity.py
│   │   ├── Layers.py
│   │   └── Losses.py
│   ├── utils/
│   │   └── PINLaplacian.py
│   └── main.py
├── scripts/
│   ├── PhotoTraining.py
│   └── Test.py
├── docs/
├── pyproject.toml
└── README.md
```

---

## 快速开始

```bash
uv sync
uv run python scripts/PhotoTraining.py
```

训练脚本默认会：

- 读取输入目录中的图像
- 训练 PINN 背景模型
- 输出：
  - `*_observed.png`
  - `*_predicted.png`
  - `*_residual.png`

---

## 数学形式

当前采用的物理约束形式为：

$$
\nabla^2 I - \alpha I + I_{city} = 0
$$

训练损失为：

$$
\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{physics}
$$

其中：

$$
\mathcal{L}_{data} = \frac{1}{N}\sum (I_{pred} - I_{obs})^2
$$

$$
\mathcal{L}_{physics} = \frac{1}{N}\sum (\nabla^2 I_{pred} - \alpha I_{pred} + I_{city})^2
$$

---

## 当前版本的研究定位

本项目当前并不把“极限星等提升”或“星表输出”作为核心任务，而把它们视为后续评估指标。

V3 的核心目标是：

- 让 `alpha` 成为可学习、但不完全放飞的 PDE 系数
- 让 `I_city` 成为可学习的结构化源项
- 建立一个适用于低噪声星空图像的物理约束分离原型

---

## 下一步

后续版本将继续推进：

1. `I_city` 范围学习
2. RGB-aware 训练与损失
3. 天空区域 mask 与真实场景适配
4. 预训练与 checkpoint 保存
5. 系统实验与消融

详见：

- `docs/版本路线图.md`
- `docs/项目题目.md`

---

## 参考文献

- Garstang, R. H. (1986). Model for artificial night-sky illumination.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks.

---

## 开源协议

MIT License
