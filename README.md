# PINN Starlight

> 基于物理信息神经网络的城市光穹反演与深空信号恢复

一款面向业余天文摄影师的光污染去除工具。输入城市近郊拍摄的 RAW 格式星空照片，通过物理信息神经网络（PINN）估算光污染的空间分布并从原图中分离，"挖出"被光污染淹没的暗弱天体信号。

**信号分离，不是信号生成**——不脑补、不伪造，所有输出可溯源到原始像素。

---

## ✨ 核心特性

- 🔬 **物理可解释**：基于光传播扩散方程约束，不是黑盒
- 📸 **RAW 输入**：保留传感器线性响应，物理模型成立
- 🎯 **反问题建模**：显式反演光污染本身，而非端到端去噪
- 🔍 **可验证输出**：附带光污染分布图，肉眼可检验
- 🚫 **诚实边界**：明确标注物理恢复上限，不夸大效果

---

## 📖 工作原理

```
输入：城市近郊拍摄的 RAW 星空照片
  │
  ├─ rawpy 解析 → 线性像素值
  │
  ▼
PINN 模型
  输入：像素坐标 (x, y)
  输出：该点光污染强度 light_pollution_intensity(x, y)
  物理约束：∇²I - κI + S = 0（简化扩散方程）
  │
  ▼
信号分离：starlight_intensity = observed_intensity - light_pollution_intensity
  │
  ▼
输出：
  - 恢复的星空图（主产品）
  - 光污染分布图（副产品，用于验证）
```

### 为什么 PINN 适合这个问题

| 方法       | 如何区分光污染与星光                                          |
|----------|-----------------------------------------------------|
| 普通 CNN   | 凭训练数据经验（可能误判暗星）                                     |
| 传统滤波     | 按频域特征（易把暗星当背景）                                      |
| **PINN** | **光污染必须符合物理传播规律（平滑、径向衰减），星光不符合（点状、尖锐）——物理方程是天然过滤器** |

---

## 🎯 典型使用场景

```
天文爱好者 → 城市近郊（如成都三圣乡）→ 微单拍摄
  → 导出 RAW → 载入本工具 → 30 秒处理
  → 导入 PixInsight / DeepSkyStacker 继续后期
```

---

## ⚠️ 能力边界

| ✅ 能做          | ❌ 不能做           |
|---------------|-----------------|
| 恢复被中等光污染淹没的暗星 | 在市中心重建郊外级别星空    |
| 比传统滤波保留更多真实细节 | 从严重过曝区域恢复信息     |
| 输出可解释的光污染分布   | 凭空生成原始数据中不存在的天体 |

受**香农信息论**约束：光污染远超星光强度时，恢复原则上不可能。本工具诚实承认这个上限。

---

## 🛠 技术栈

- **语言**：Python 3.12
- **核心库**：PyTorch、rawpy、NumPy、*astropy、Matplotlib（待定）*
- **测试**：Jupyter、pytest
- **包管理**：uv
- **版本控制**：Git

---

## 📊 评估指标

### 待定
项目论文将报告以下指标：

- PSNR / SSIM（通用图像质量）
- 暗星检测率（天文专用）
- 光度测量误差（天文专用）
- 消融实验（验证物理约束作用）
- 与传统方法（中值滤波、AstroPy Background2D）对比

---

## 🗂 项目结构

### 待定

```

pinn-starlight-core/
├── src/
│   └── pinn_starlight/
│       ├── __init__.py
│       ├── model.py           # PINN 网络定义
│       ├── physics_loss.py    # 物理损失函数
│       ├── data_pipeline.py   # RAW 读取与预处理
│       └── train.py           # 训练脚本
├── tests/                     # pytest 测试
├── notebooks/                 # Jupyter 实验
├── data/                      # 数据集（.gitignore）
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 🚀 快速开始

### 待定

```bash
# 克隆仓库
git clone https://github.com/pinn-starlight/core.git
cd core

# 安装依赖（需先安装 uv）
uv sync

# 运行测试
uv run pytest

# 启动 Jupyter
uv run jupyter notebook
```

---

## 👥 团队



| 成员       | 职责                             |
|----------|--------------------------------|
| **主力开发** | 代码实现、数据管道、论文主笔、可视化             |
| **算法审核** | PINN 架构 review、数学公式校验、传统方法对比实现 |
| **学术顾问** | 物理方程把关、论文终审、方向纠偏               |

---

## 🏆 项目背景

本项目为 **2026 年丘成桐中学科学奖（计算机领域）** 参赛作品。

---

## 📄 引用的关键先前工作

### 待定

- Raissi et al., "Physics-informed neural networks", *Journal of Computational Physics*, 2019
- Falchi et al., "The new world atlas of artificial night sky brightness", *Science Advances*, 2016
- （更多文献随项目进展补充）

---

## 📝 开源协议

### 待定

*MIT License*

---

## 🌠 致谢

> 感谢每一位仍抬头看星空的人。

PR Test