# PINN Starlight — Stage10-A

> **当前正式实验版本：Stage10-A fixed split。**  
> 本 README 描述当前正式链路；历史演化见 `PINN_Starlight_Stage10_完整项目说明_合并中文版.md`。论文事实以正式源码、selection artifacts 与 Global Configuration Lock 为最高优先级。

## 1. 任务

PINN Starlight 研究星空图像中的**非均匀光污染/天空背景估计与抑制**。四种正式方法统一输出：

```text
background_pred
```

公共实验层统一重建：

```text
residual_pred = observed - background_pred
```

主 benchmark 不是通用去噪。semi-synthetic observation model 为：

```text
observed = clean_true + background_true
```

其中 `clean_true` 是 native-background-flattened camera reference，并非 noiseless ground truth。正式主 benchmark 不额外注入未经标定的 Gaussian/Poisson noise，科学 float 数组不 clip。

## 2. 权威顺序

1. `experiments/configs/formal_experiment.py` 与 E0–E4 实际源码；
2. selection artifacts + Global Configuration Lock；
3. `docs/编写/III_Method_中文草稿.md` 与 `IV_Experimental_Setup_中文草稿.md`；
4. Stage 10 总交接文档；
5. Stage 9 文档只作为历史审计快照。

`src/pinn_starlight_core/main.py` 是历史 prototype，不定义正式论文 PINN。

## 3. 四种正式方法

| internal id | paper-facing name | role |
| --- | --- | --- |
| `fft_gaussian` | Gaussian | classical low-frequency filtering baseline |
| `fourier_ls` | Fourier-LS | physics-matched classical optimization baseline |
| `unet_small` | **U-Net** | supervised background regression baseline |
| `pinn` | PINN | per-image physics-informed coordinate optimization |

### PINN

- coordinate MLP；
- `alpha=0.5`；
- signed effective forcing；
- `K_max=3`；
- E1 选择冻结 configuration；
- E2/E3/E4 逐图重新优化；
- 正式 Laplacian backend = nested autograd。

### Fourier-LS

- 与 PINN 共享 forcing family / observed-only initialization；
- full-resolution screened-Poisson Fourier solver；
- bounded block-coordinate Powell；
- exact nonnegative active-set weight profile；
- CUDA `torch.fft`；
- float64/complex128；
- Parseval/caching 仅为数学等价的实现优化，不改变正式优化问题。

### U-Net

代码 identifier 仍为 `unet_small`，论文统一显示为 **U-Net**：

```text
64 → 128 → 256 → 512 → 1024
Conv + ReLU
L1 background regression
patch 256
batch 4
12 × 100 training steps
Adam
AMP
fixed tiled inference: 256 / overlap 64 / tile batch 1
```

## 4. 数据协议

- 5 个 camera-reference bases；
- base-level split：3 train / 1 validation / 1 test；
- 3 background types × 2 levels；
- 30 formal synthetic samples = 18 train + 6 validation + 6 test；
- `multi_source` 实际注入 2 个源；
- forcing capacity 最多 3 个 effective components；
- native-background audit p95 relative flux error threshold = 5%。

## 5. Formal execution policy

Stage10-A 正式 E2/E3/E4：

```text
benchmark GPU = single visible GPU
logical device = cuda:0
PINN Laplacian = autograd
Fourier-LS backend = cuda
Fourier-LS dtype = float64/complex128
U-Net memory format = NCHW contiguous
U-Net optimizer = ordinary Adam
U-Net tiled batch size = 1
```

多 GPU 不用于正式 benchmark。若后续使用多 GPU 缩短 selection wall time，应采用 independent candidate/seed-level parallelism，并单独记录。

## 6. 正式运行顺序

PowerShell（Pixi）：

```powershell
$PY = ".\.pixi\envs\default\python.exe"

& $PY -m experiments.scripts.preflight.run
& $PY -m experiments.scripts.e0.run --force

& $PY -m experiments.scripts.e1.run --force --device cuda:0 --laplacian-backend autograd
& $PY -m experiments.scripts.selection.gaussian.run --force
& $PY -m experiments.scripts.selection.fourier_ls.run --force --backend cuda
& $PY -m experiments.scripts.selection.unet.run --force

& $PY -m experiments.scripts.config_lock.run --force

& $PY -m experiments.scripts.e2.run --force
& $PY -m experiments.scripts.e3.run --force
& $PY -m experiments.scripts.e4.run --force
```

已有正式 synthetic 数据时不要无理由重建。只有明确需要重建时才运行：

```powershell
& $PY -m experiments.common.data.generate_synthetic --force
```

## 7. `run_pipeline.py`

当前 runner 已修复并支持选择起止阶段，不再包含不存在的 `selection.pinn.run`。

例如：

```powershell
& $PY run_pipeline.py --start-at e1 --stop-after lock --force
& $PY run_pipeline.py --start-at e2 --stop-after e4 --force
```

## 8. E0–E4

- **E0**：smoke，不进入论文正式结果；
- **E1**：PINN validation selector；
- **Global Lock**：纯冻结闸门；
- **E2**：held-out synthetic 主比较；
- **E3**：PINN held-out ablation；
- **E4**：16 张真实高污染图的无真值 descriptive/qualitative evaluation。

## 9. 当前范围

当前完成的是 **Stage10-A fixed split**。Stage10-B grouped cross-validation 等 Stage10-A 完整冻结后再决定是否实现；当前不能把 cross-validation 写成已完成结果。
