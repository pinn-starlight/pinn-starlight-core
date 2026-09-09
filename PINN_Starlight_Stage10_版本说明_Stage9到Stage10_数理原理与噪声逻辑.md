# PINN Starlight Stage 10 版本说明

## Stage 9 → Stage 10：数据语义、物理强迫、四方法冻结与正式运行协议

**版本日期：2026-09-07**  
**基线：Stage 9 历史快照**  
**当前事实源：本 Stage 10 包中的正式源码与 `experiments/configs/formal_experiment.py`**

> 本文是 **Stage 9 → Stage 10 的增量说明**。Stage 9 两份历史文档保持原文件、原哈希不变，作为 `latest.zip → Stage 9` 的研发审计快照；完整连续交接请读根目录 `PINN_Starlight_Stage10_完整项目说明_合并中文版.md` 与 `PINN_Starlight_Stage10.docx`。
>
> 若历史段落、原型入口或旧草稿与当前正式源码冲突，**当前正式实验以源码事实为准**。正式 PINN 的实现入口是 `experiments/common/baselines/coordinate_pinn.py`；`src/pinn_starlight_core/main.py` 仅是旧的单图原型/演示入口，不能替代 E0–E4 协议。

---

## 1. 一页结论

Stage 10 不是 Stage 9 的简单调参，而是一次**基准语义、有效强迫模型和实验冻结边界的升级**。

1. 主任务是**光污染背景估计与移除**，不是去噪。正式半合成观测式固定为 `observed = clean_true + background_true`，不再额外注入 Gaussian、Poisson 或 Poisson-Gaussian 噪声。
2. `clean_true` 是历史变量名；Stage 10 中它表示**真实相机参考图经过固定 native low-frequency background subtraction 后的参考数组**，不是 noiseless ground truth。
3. 正式科学数组不做数值 clip，也不在 native flattening 之后做乘法重标定；PNG 的裁剪只服务显示。
4. 物理源项统一解释为 `S_city`：**signed effective forcing**。它不是逐点非负的“城市辐亮度图”。
5. PINN 和 Fourier-LS 共用 observed-only 候选中心规则和同一类 signed forcing family，但求解器不同；Gaussian 与 U-Net 仍是独立基线。
6. forcing 容量为 `K_max=3`；当前 `multi_source` 数据生成器实际注入**恰好 2 个**源。因此只能声称“模型最多支持 3 个有效分量；multi-source benchmark 使用 2 个注入分量”。
7. E1 就是 **PINN selector**。不存在 `experiments.scripts.selection.pinn.run`；Global Lock 也明确把 `pinn` 映射到 `python -m experiments.scripts.e1.run`。
8. 当前 `run_pipeline.py` 仍含一个过期的 `experiments.scripts.selection.pinn.run` 调用。由于本次任务只维护文档、**不改 Python**，所以 Stage 10 正式运行请按本文手工命令执行，不要把 `run_pipeline.py` 当作协议事实源。
9. 当前代码实现的是 **Stage10-A：固定 base-level train/validation/test 划分**。所谓 Stage10-B grouped/cross 版本目前没有实现，不得写成已完成实验。

---

## 2. 数据语义：为什么 Stage 10 要重建 benchmark

### 2.1 Stage 9 的 reference/label 语义风险

半合成观测可以写成：

```text
I_obs = I_ref + B_added
```

真实相机参考图 `I_ref` 仍可能带有 native low-frequency sky/background。若评估标签只取 `B_added`，单图背景估计方法可能自然把 native background 与新增 skyglow 一起解释为背景，而监督 U-Net 却只被训练去回归 `B_added`。这会造成方法之间的目标语义不完全一致。

### 2.2 Stage 10 的统一定义

Stage 10 先执行一个与四种被测方法无关的固定预处理：

```text
B_native_hat = N(I_ref)
clean_true   = I_ref - B_native_hat
observed     = clean_true + background_true
```

其中 `background_true` 只表示后来注入的 synthetic skyglow。这里的 `clean_true` 仍保留历史变量名，但论文应解释其真实含义：**native-background-flattened camera reference**。

正式代码明确冻结：

- `additional_synthetic_noise = False`；
- `skyglow_induced_shot_noise_simulated = False`；
- `scientific_clipping = False`；
- `multiplicative_rescaling_after_flattening = False`。

这意味着 Stage 10 主 benchmark 不把“增加 skyglow 后额外产生的 shot noise”假装成已物理标定的 Poisson 过程。若未来有 gain、曝光、电子数/ADU、read noise 等可靠标定，应作为独立 robustness/stress-test 协议处理，而不是悄悄混入主实验。

### 2.3 low/high 的含义

`low` / `high` 是**skyglow 强度和空间尺度等级**，不是噪声等级：

| level | intensity | scale |
| --- | ---: | ---: |
| low | 0.10 | 0.55 |
| high | 0.20 | 0.85 |

同一 base、同一 background type 的 low/high 共用 geometry seed，可用于 E2 的 high-low consistency 诊断。

---

## 3. Native-background flattening 与严格审计

固定预处理配置来自 `SYNTHETIC_DATA_CONFIG["native_background"]`：

| 项目 | Stage 10 冻结值 |
| --- | --- |
| 方法 | `photutils.Background2D_sigma_clipped_median` |
| mesh | `min(max(min(H,W)/16, 64), 256)` 对应的冻结策略 |
| sigma clip | 3.0，最多 5 次 |
| filter size | 3 |
| star audit threshold | 0.03 |
| star audit FWHM | 3.0 |
| audit max stars | 300 |
| `abs(flattened sky median)` | `< 0.005` |
| median centroid shift | `< 0.10 px` |
| median relative flux error | `< 0.01` |
| **p95 relative flux error** | **`< 0.05`（5%）** |
| additive identity max abs error | `< 1e-6` |
| strict audit | `True` |

特别注意：Stage 10 正式代码的 p95 阈值是 **5%**，不是 3%。生成器会先写入每个 base 的 `audit.json`，随后才在 strict audit 失败时抛异常，因此失败现场仍可追溯。

---

## 4. 合成数据协议

固定 base-level split：

```text
train:
  clean_single_003
  clean_stack_002
  clean_stack_001
validation:
  clean_single_002
test:
  clean_single_001
```

每个 base 生成 `3 background types × 2 levels = 6` 个 variant：

- `single_radial`
- `single_eccentric`
- `multi_source`（当前代码恰好生成两个源）

因此总计：

```text
5 bases × 3 types × 2 levels = 30 samples
train / validation / test = 18 / 6 / 6
```

所有 spatial 参数位于统一的 `[-1,1]^2` 归一化坐标系；RAW loader 下采样为 2；正式 synthetic 默认不做中心裁剪。

每个 base 的星点参考目录在 flatten 后的 camera reference 上一次性冻结：threshold `0.03`、FWHM `3.0`、matching radius `3 px`。它是项目内部的 frozen reference catalogue，不是外部 catalogue ground truth。

---

## 5. 屏蔽泊松方程与 signed effective forcing

正式物理约束写作：

```text
∇²B(x,y) - α B(x,y) + S_city(x,y) = 0
```

Stage 10 正式实验固定 `α = 0.5`。

对项目采用的径向 ansatz，代码中的基础有效强迫项为：

```text
q(r; α) = 2α cos(√α r)
        + (√α / r) sin(√α r)
        - 16 r² α²
        + r⁴ α³
```

实现用 `sqrt(r² + 1e-8)` 避免 `r=0` 的数值奇异。

第 k 个 forcing template 为 `q(r_k;α)` 与旋转椭圆 Gaussian envelope 的乘积，并在固定 `64×64` canonical grid 上做 unit-RMS normalization。总强迫写成：

```text
S_city(x,y) = Σ_k w_k * T_k(x,y),    w_k >= 0
```

`w_k` 通过 softplus 保证非负，但 `T_k` 本身可以有负瓣，所以 `S_city` 是 **signed effective forcing**，不能把它解释成逐像素非负的城市发光强度。

---

## 6. 正式 PINN：`coordinate_pinn.py` 才是 E0–E4 事实源

### 6.1 每张图独立优化

正式 PINN 只接收 `observed`，每张图重新初始化并优化；E1 产生的是**配置选择 artifact**，不是可复用 PINN checkpoint。E2/E3/E4 按冻结配置对各图重新求解。

### 6.2 网络与损失

背景网络输出经过：

```text
B_hat = sigmoid(MLP(x,y))
```

因此正式 PINN 的 `B_hat` 被约束到 `(0,1)`。总损失是：

```text
L_data   = MSE(B_hat, I_obs)
L_phys   = mean((∇²B_hat - α B_hat + S_city)^2)
L_sparse = mean(w_k)
L_total  = L_data + λ_phys L_phys + λ_sparse L_sparse
```

当 `physics_weight=0` 时，forcing 参数不会进入 optimizer，同时 `L_phys`、`L_sparse` 在该分支记为 0；这使 E3 `data_only` 成为真正的数据项基线。

### 6.3 Stage 10 默认配置与 forcing 参数化

| 参数 | 默认值 / 规则 |
| --- | --- |
| hidden dims | `[128,128]`（E1 还会选择） |
| physics weight | 0.5（E1 还会选择） |
| sparse weight | `1e-3`（E1 还会选择） |
| nominal steps | 10000（E1 continuation 决定最终冻结值） |
| batch | 8192 |
| model / forcing LR | `1e-3 / 1e-3` |
| alpha | **0.5 固定** |
| center warm-up | 1000 个**绝对 global steps** |
| K max | 3 |
| sigma bounds | `[0.05, 0.55]` |
| theta bounds | `[-π/2, π/2]` |
| P99.5 winsor | 99.5 |
| low-pass sigma fraction | 0.04 × min(H,W) |
| min center distance | 0.30（`[-1,1]^2`） |
| bounded center refinement | `0.08*tanh(offset)`，再 clamp 到 `[-1,1]` |
| initial weights | `[1.0, 0.05, 0.05]` |
| initial sigma | 0.425 |
| canonical RMS grid | 64 |
| active logging threshold | relative weight ≥ 0.05 |
| prediction batch | 65536 |

候选中心完全由 `observed` 初始化：P99.5 winsorization → strong Gaussian low-pass → greedy NMS。`active` 只是诊断标签，不进行 hard pruning。

### 6.4 不要把 prototype PINN 混进论文

`src/pinn_starlight_core/main.py` 是历史原型：它有 learnable `alpha`（初值 0.55、范围 0.4–0.6）、1000 steps、batch 1024、physics weight 0.4，并直接联合优化 model/forcing/alpha；它不等于 Stage 10 正式实验。

论文与正式复现实验必须引用：

```text
experiments/common/baselines/coordinate_pinn.py
experiments/configs/formal_experiment.py
```

---

## 7. Fourier-LS：共享物理 family，不共享优化器

Fourier-LS 与 PINN 共享：

- `α=0.5`；
- `K_max=3`；
- observed-only center initialization；
- sigma/theta/center bounds；
- signed effective forcing family；
- unit-RMS component normalization；
- nonnegative forcing amplitudes。

但它不是“另一个 PINN”。它在周期边界条件下对固定 forcing geometry 使用 screened-Poisson Fourier 解，并在 `K=3` 时枚举全部 `2^K = 8` 个 active sets，精确 profile 非负权重。几何参数由 bounded profile block-coordinate Powell 优化：最多 2 sweeps，relative-objective early stop `1e-4`，每分量 `maxiter=25`，`xtol=1e-3`，`ftol=1e-4`。

选择阶段：

- `lambda_LS`: `0.01, 0.1, 0.3, 0.4, 0.5`；
- 若最优点落边界，额外尝试 `0.001` 或 `1.0`；
- 先用 sparse=`1e-3` 选 lambda，再在所选 lambda 上比较 sparse `1e-4,1e-3,1e-2`；
- 只使用 validation。

---

## 8. Gaussian 与 U-Net-small

### 8.1 FFT-Gaussian

Gaussian selector 只用 validation，从归一化 sigma：

```text
0.02, 0.04, 0.08, 0.16
```

中选择 validation background MSE 最小者。方法本身是确定性的。

### 8.2 U-Net-small

U-Net 是**单通道 background regression**，不是语义分割：

- 4-level encoder/decoder；
- LeakyReLU；
- 最后一层 `1×1 Conv`，无输出激活；
- training/validation loss 为 mean absolute error（L1）；
- base channels = 16；
- patch = 256；batch = 4；
- 12 epochs × 100 steps/epoch；
- LR = `1e-3`；patience = 3；relative min delta = 0.002；
- 三个 formal seeds：`20260728/29/30`；
- AMP requested；
- inference 优先 full image，只有 CUDA OOM 才固定 tiled fallback：tile 256、overlap 64。

每个 seed 生成自己的 checkpoint。E2 会评估三个已冻结 checkpoint；E4 只使用 validation loss 最小的那个 checkpoint。

---

## 9. 四个 pre-test selector 与 Global Configuration Lock

Stage 10 把四种方法的 pre-test 选择都放在 test gate 之前：

| 方法 | selector / trainer | 可见 split |
| --- | --- | --- |
| PINN | `python -m experiments.scripts.e1.run` | validation |
| FFT-Gaussian | `python -m experiments.scripts.selection.gaussian.run` | validation |
| Fourier-LS | `python -m experiments.scripts.selection.fourier_ls.run` | validation |
| U-Net-small | `python -m experiments.scripts.selection.unet.run` | train + validation |

E1 顺序：

```text
T1 network: [256,128], [256,64], [128,128], [128,64]
T2 physics_weight: 0.4, 0.5, 0.8, 1.0, 1.5
T3 sparse_weight: 1e-4, 1e-3, 1e-2
T4 steps: 每次续训 1000，最多 50000；stability tolerance 0.005，patience 3
```

PINN 的排序 key 为：validation background MAE → residual SSIM（越高越好）→ matched-star mean relative flux error → parameter count。

Global Lock (`schema_version = 5`) 是纯 freeze gate：它不训练、不调参、也不加载 synthetic image arrays。它验证四份选择 artifact、U-Net checkpoint hash、formal config、dataset protocol、manifest/source provenance，然后才写出 `formal_test_ready=True`。

---

## 10. E0–E4 的职责

### E0：smoke，不是论文结果

E0 只检查 I/O、shape、有限值和小预算运行；不会用于选模型，也不进入正式结果。

### E1：PINN validation selector

见上节。E1 **不是**一个额外于 PINN selector 的实验，也没有 `selection/pinn/run.py`。

### E2：held-out synthetic test 主比较

E2 只加载 `split=test`，不允许重新选择 checkpoint 或读取 train/validation。四方法统一返回 `background_pred`，公共层统一计算：

```text
residual_pred = observed - background_pred
```

主要产物包含 `metrics.csv`、`training.csv`、`grouped_results.csv`、`main_result_table.csv`、`failure_cases.csv`、`low_high_consistency.csv` 等。

### E3：PINN held-out test 消融

冻结变体：

| variant | 只允许改变的含义 |
| --- | --- |
| `data_only` | `physics_weight=0`, `sparse_weight=0` |
| `single_component` | `K=1` |
| `multi_fixed_centers` | K=3，关闭 center refinement |
| `multi_bounded_centers` | 正式 K=3 bounded refinement |

中心误差只在 `single_eccentric` 子集、仅 evaluation 侧使用 injected true center，并匹配最近的 active effective component；不能让 generation truth 进入 Method。

### E4：真实高污染图描述性比较

E4 要求 `collections/real_high_pollution` **恰好 16 张图**。没有可靠背景真值，因此只做 descriptive statistics 和定性观察，不报告“真实背景 MAE/PSNR/SSIM”之类不存在的 truth-based 指标。

---

## 11. 指标与残差语义

科学残差始终是 raw float：

```text
residual_pred = observed - background_pred
```

正式 star detection 使用 raw residual；`nonnegative` / `legacy_01` 只是 validation/debug 模式。

synthetic 指标包括：

- background MAE/RMSE/PSNR/SSIM；
- residual MAE/RMSE/PSNR/SSIM；
- star precision/recall/F1；
- matched-star mean relative flux error。

星点检测上限是 2000，并有 `*_count_capped` 标志。因此若 cap 触发，应写“检测数达到 2000 的预设上限”，不能写“预测出 2000 颗星”。

flux metric 使用 local-background-subtracted raw-aperture photometry；正式 aperture radius 4.5 px、annulus 6–8 px。

真实 E4 指标是 descriptive-only，例如 background mean/TV、residual mean/std/negative fraction、detected-star count、runtime；不能把它们当 ground-truth accuracy。

---

## 12. Stage10-A 与 Stage10-B 状态

当前包里的**已实现正式协议**是 Stage10-A：固定 5 个 camera-reference bases 的 3/1/1 base-level split，对应 18/6/6 synthetic samples。

当前源码中没有 grouped cross-validation / cross-fold selector / grouped-cross lock 的实现。因此：

- 可以把 Stage10-B 作为未来版本设计；
- 不得在 Method、Experimental Setup 或论文中声称“已完成 grouped/cross 实验”；
- 若以后实现 Stage10-B，它应产生新的协议版本与新的 selection/lock/results，而不是覆盖 Stage10-A 的历史输出。

---

## 13. 正式手工运行顺序

Python 版本要求：`>=3.11,<3.12`，仓库 `.python-version` 为 `3.11`。依赖可由项目的 `uv` 或 `pixi` 配置安装。

若数据不在仓库旁，可设置：

```powershell
$env:PINN_STARLIGHT_DATA_ROOT = "D:\path\to\data-root"
```

**推荐逐步运行，不使用当前 `run_pipeline.py`：**

```powershell
# 0. 第一次只读预检（若 formal synthetic 尚未生成，会提示下一步）
python -m experiments.scripts.preflight.run

# 1. 重建 Stage 10 synthetic dataset
python -m experiments.common.data.generate_synthetic --force

# 2. 重建后再次只读预检
python -m experiments.scripts.preflight.run

# 3. 可选 smoke
python -m experiments.scripts.e0.run --force

# 4. 四个 pre-test selector / trainer
python -m experiments.scripts.e1.run --force
python -m experiments.scripts.selection.gaussian.run --force
python -m experiments.scripts.selection.fourier_ls.run --force
python -m experiments.scripts.selection.unet.run --force

# 5. 纯冻结闸门
python -m experiments.scripts.config_lock.run --force

# 6. 正式 test / ablation / real branch
python -m experiments.scripts.e2.run --force
python -m experiments.scripts.e3.run --force
python -m experiments.scripts.e4.run --force
```

### 当前 launcher 已知问题

`run_pipeline.py` 目前在 E1 之后又调用：

```text
python -m experiments.scripts.selection.pinn.run
```

但该模块不存在。`config_lock/run.py` 的正式映射已经明确表明 PINN selector 是 `experiments.scripts.e1.run`。本次文档维护遵守“不改 Python”要求，所以**没有修 launcher**；正式复现请按上面的模块命令执行。

---

## 14. 论文中允许与禁止的表述

### 推荐表述

- “The task is light-pollution background estimation/removal rather than generic denoising.”
- “The camera reference is flattened with a fixed method-independent native-background estimator before synthetic skyglow injection.”
- “No additional Gaussian or Poisson noise is injected in the main benchmark.”
- “The physics term uses a sparse mixture of signed effective-forcing templates.”
- “The formal PINN fixes α=0.5 and optimizes each image independently.”
- “The model supports up to three effective forcing components; the current multi-source synthetic benchmark injects two sources.”
- “Real-image results are descriptive/qualitative because no reliable background ground truth is available.”

### 禁止或需要改写

- ❌ “clean_true 是无噪声 ground truth。”
- ❌ “low/high 是低噪声/高噪声。”
- ❌ “S_city 是逐点非负的城市辐亮度。”
- ❌ “正式 PINN 学习 alpha。”
- ❌ “U-Net 是 segmentation baseline。”
- ❌ “E1 之后还要运行 selection.pinn。”
- ❌ “当前 Stage 10 已完成 grouped/cross-validation。”
- ❌ “E4 的无参考统计证明某方法的真实背景更准确。”

---

## 15. 当前源码权威映射

| 事实 | 当前权威文件 |
| --- | --- |
| 所有 formal 配置 | `experiments/configs/formal_experiment.py` |
| synthetic protocol | `experiments/common/data/generate_synthetic.py` |
| background generator | `experiments/common/data/synthetic_backgrounds.py` |
| native background/audit | `experiments/common/data/native_background.py` |
| 正式 PINN | `experiments/common/baselines/coordinate_pinn.py` |
| shared forcing geometry | `src/pinn_starlight_core/physics/forcing.py` + `src/pinn_starlight_core/nn/physics_model.py` |
| Gaussian | `experiments/common/baselines/fft_gaussian.py` |
| Fourier-LS | `experiments/common/baselines/fft_screened_poisson.py` |
| U-Net | `experiments/common/baselines/unet_small.py` |
| metrics | `experiments/common/utils/metrics.py` |
| PINN selector | `experiments/scripts/e1/run.py` |
| Gaussian/Fourier/U-Net selectors | `experiments/scripts/selection/*/run.py` |
| Global Lock | `experiments/scripts/config_lock/run.py` |
| E2/E3/E4 | `experiments/scripts/e2/run.py`, `e3/run.py`, `e4/run.py` |
| historical prototype | `src/pinn_starlight_core/main.py`（非正式 E0–E4 入口） |

---

## 16. 本次文档维护的边界

这次维护只做**文档同步与交接固化**：

- 不修改任何 `.py`；
- 不修改 Stage 9 两份历史快照；
- 不执行 E0–E4 数值实验；
- 不把历史结果当作当前 Stage 10 的新结果；
- 不虚构当前包中不存在的 Stage 9 源码二进制 diff。

当前包能证明的是：Stage 9 文档中已经记录的 `latest → Stage 9` 审计链被原样保留；Stage 10 的“当前事实”由本包正式源码重新核对并在新总文档中接续。若未来需要**机械的 Stage9 source tree → Stage10 source tree 逐文件代码 diff**，必须同时提供那一版 Stage9 源码树本身。
