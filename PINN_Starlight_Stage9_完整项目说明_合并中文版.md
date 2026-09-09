# PINN Starlight Stage 9：完整项目说明与正式实验操作手册（合并中文版）

> **文档定位**
>
> 这是一份把 Stage 2 → Stage 9 的研发记录、接口冻结、评价协议、数据接入审计、职责重构和正式实验运行手册合并到一起的中文主文档。
>
> 本文保留各阶段“当时为什么这样改”的历史信息，同时明确标记后来已经被新阶段取代的旧口径。这样既能作为小组交接文档，也能作为研发审计记录使用。
>
> - 文件编码：UTF-8
> - 面向环境：Windows / PowerShell / WPS / VS Code / Typora
> - 代码变量、路径、命令、类名、函数名、配置键和已冻结的英文技术名词保持原样
> - 面向人的解释、标题和操作说明统一为中文

---

## 0. 阅读前先看：哪些旧口径已经被后续阶段取代

把所有 Stage 文档合并后，最容易出现的误解是：某个早期阶段的描述和 Stage 9 最终代码事实并不完全一致。这不是文档冲突，而是项目迭代留下的历史轨迹。

请优先遵守下面四条“最终口径”：

1. **Stage 2 / Stage 2.1 里的额外 Gaussian noise 是历史口径。**  
   从 Stage 7 起，正式半合成观测模型冻结为：

   \[
   I_{\mathrm{obs}} = I_{\mathrm{ref}} + B_{\mathrm{true}}.
   \]

   `I_ref` 本身就是相机实拍参考图，因此正式 benchmark 不再额外注入 Gaussian、Poisson 或 Poisson-Gaussian noise。

2. **Stage 6 里 detection 使用 `max(raw_residual, 0)` 是历史口径。**  
   Stage 7 之后，正式 detection 使用 **raw residual**；`nonnegative` 和 `legacy_01` 只保留给 validation/debug 对照。

3. **Stage 3 / Stage 4 / Stage 5 的 Global Lock schema 是历史演化。**  
   Stage 9 最终冻结为 **Global Lock schema 4**，并把四个方法的 selection/training 从 Global Lock 中彻底拆出。

4. **Stage 8 中“U-Net 参数集中在 `config_lock/run.py`”已经被 Stage 9 重构取代。**  
   Stage 9 的唯一正式配置源是：

   ```text
   experiments/configs/formal_experiment.py
   ```

如果只想知道“现在最终应该怎么跑”，可以直接跳到本文 **第 11 节：Stage 9 正式实验运行手册**。

---

# 第一部分：项目演化与职责冻结

## 1. Stage 2：合成数据协议与 Method 接口封口

### 1.1 这一阶段解决什么问题

Stage 2 的目标不是增加更多模型，而是先把数据协议和 Method 能看到的信息边界封死，避免后面做 E1/E2/E3/E4 时出现：

- 同一张图在不同 runner 中被不同方式加载；
- U-Net 或 PINN 意外拿到 `background_true`、`clean_true`、metadata 等 truth；
- 不同实验各自重建 `star_reference`；
- residual 的定义在不同方法里各写一套；
- test split 与方法内部逻辑耦合。

这一阶段以重新上传的 `latest.zip` 为基线，并重新落实了此前讨论的 Synthetic Stage 1 协议，然后继续完成接口封口。

### 1.2 当时记录的源版本指纹

- Source ZIP SHA-256：  
  `9f412bf6897c802de92c7b07333c6f128a730d93fb6ecaf9d6b43adc2bc9551b`
- Stage 2 修改后 Python tree SHA-256：  
  `dc7bd72e6615bf60e6710eb5b32619f17205b993fdddf34d42c6f198fcd18b1c`
- 当时共计 22 个 `.py` 文件进入该树指纹。

当时上传的源码虽然已经包含 `fft_screened_poisson.py`，但 synthetic generator 仍是旧版：仍使用旧 `exponential/elliptical` 口径、会 clip `observed`，而且 `star_reference` 还是 TODO，因此 Stage 2 先重新实施冻结后的 synthetic protocol。

### 1.3 Stage 2 当时的 synthetic protocol

先按 clean base 切分，再生成 variants，防止同一个 clean base 的不同变体跨 train/validation/test：

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

每个 base 生成：

```text
3 个 background type × 2 个 intensity level = 6 个 variant
```

三个 background type：

```text
single_radial
single_eccentric
multi_source
```

Stage 2 **当时**的 low/high 设置为：

```text
low:
  intensity = 0.10
  scale = 0.55
  Gaussian noise std = 0.005

high:
  intensity = 0.20
  scale = 0.85
  Gaussian noise std = 0.015
```

> 注意：上面的额外 Gaussian noise 后来被 Stage 7 正式取消。这里保留只是为了说明 Stage 2 当时的开发历史。

`scale` 定义在统一的归一化坐标 `[-1,1]^2` 中。

### 1.4 避免 inverse crime

`single_eccentric` 和 `multi_source` 的 generator 不直接复用 PINN 方法里的 `Icity` 表达式。

原因是：如果 synthetic generator 和被测方法使用完全相同的解析生成形式，就会产生明显的 inverse-crime 风险——方法只是在反演自己生成数据时用的同一族函数，不能公平代表泛化能力。

### 1.5 star_reference 的冻结

对每个 `clean_true` base：

- 使用 threshold `0.03`；
- 只生成一次 frozen star catalogue；
- 同一个 clean base 的六个 variants 共用这一份 catalogue。

E1/E2/E3 不再各自从 `clean_true` 重建星表。

### 1.6 scientific float path 不 clip

Stage 2 把正式科学数据路径明确为 float TIFF。

当时观测模型写作：

```text
observed = clean_true + background_true + gaussian_noise
```

并且：

- scientific TIFF 不 clip；
- PNG 只作为显示用途，允许 display-only clipping；
- 稳定 seed 由 protocol identifiers 派生，不依赖循环顺序；
- generation truth 独立保存在 metadata，只允许 evaluation / explanatory analysis 使用。

> Stage 7 以后，正式观测式更新为 `I_obs = I_ref + B_true`，不再包含额外 synthetic noise。

### 1.7 Stage 2 的接口封口

这一阶段最关键的不是生成数据，而是封住 truth 旁路。

#### 1.7.1 synthetic TIFF 走 float loader

synthetic TIFF 不再经过可能 clip 到 `[0,1]` 的 `ImageLoader`，而是直接读 float TIFF。

U-Net 的 supervised training pairs 也使用这条 unclipped float path。

#### 1.7.2 E1/E2/E3 不再重建 star_reference

评价器统一使用 frozen metadata catalogue。

#### 1.7.3 E2 分组统一

E2 的统计分组统一成：

```text
background_type × intensity_level
```

#### 1.7.4 E2 Method dispatcher 不再收到完整 sample

旧设计：

```text
run_method(sample, ...)
```

会让 dispatcher 天然能够访问：

```text
clean_true
background_true
metadata
star_reference
generation truth
```

Stage 2 改成只传：

```text
observed ndarray
```

这是信息权限边界上非常重要的一步。

#### 1.7.5 PINN 不再自己定义 residual

PINN 正式 Method 只返回：

```text
background_pred
diagnostics/state
```

公共 reconstruction 层统一计算：

```text
R_hat = observed - B_hat
```

并放在：

```text
experiment_utils.reconstruct_residual
```

E0–E4 统一走这条 residual reconstruction path。

### 1.8 Stage 2 明确没有做完的内容

Stage 2 还是中间阶段，当时明确没有完成：

1. Global Configuration Lock 重构；
2. Fourier-LS 完整方法；
3. U-Net U-01/U-02 的最终训练与推理规则；
4. PSNR/SSIM 与星点 flux 指标清理；
5. 最终论文与文档同步。

### 1.9 当时完成的检查

- `python -m compileall -q .` 通过；
- 三种 background × 两种 level 的 deterministic/shape/finite/peak/source-centre 检查通过；
- high-level observation 可以自然超过 1.0，确认 scientific path 未 clip；
- float TIFF save/load 与 `load_observed(ndarray)` 能保留 `<0` 和 `>1` 数值；
- 静态检查确认 E2 `run_method` 没有 truth/sample access；
- E1/E2/E3 中没有重复的 star-reference constructor。

当时的检查环境缺少 `rawpy`、`astropy`、`photutils`，所以没有在该沙箱里做完整数据构建和 E1–E4 全流程运行。

---

## 2. Stage 2.1：冻结前的四项审计修正

### 2.1 基线

- Base ZIP：`pinn-starlight-synthetic-stage2.zip`
- SHA-256：`e07ee22fd48bd45c3c157319ad80fa73f56b1d63cb149a8d49ee73cde5f2045e`

这一阶段不新增 Fourier-LS、Global Lock 或后续 photometry 清理，只对 Stage 2 的冻结边界做补丁。

### 2.2 修正一：禁止 stale synthetic 数据混入

generator 默认拒绝写入非空的正式 synthetic 输出目录。

如果确实要重建，必须显式：

```text
--force
```

并且 destructive cleanup 被限制在 `data/collections` 的子目录内，避免误删项目其它路径。

这解决的问题是：旧样本目录如果残留，而新 manifest 只覆盖其中一部分，就可能产生“新旧数据混在一起”的 silent contamination。

### 2.3 修正二：low/high 共用 geometry seed

对同一个：

```text
base_id × background_type
```

low/high 共用同一个 `geometry_seed`。

因此以下几何属性保持配对：

- source center；
- eccentricity / orientation；
- two-source weights；
- random local-scale factors。

而 low/high 只改变冻结协议允许改变的 intensity/scale 等条件。

metadata/manifest 记录 `geometry_seed`；`background_seed` 作为 compatibility alias 保留。

> Stage 2.1 原文还提到 level-specific Gaussian noise；这部分后来由 Stage 7 取消。

### 2.4 修正三：正式 Method 边界彻底 array-only

`experiment_utils.load_observed()` 强制要求输入为：

```python
2-D np.ndarray
```

不再接受路径。

如果传 path，应直接报错，而不是暗中：

```text
path
  -> load_gray_image()
  -> clip [0,1]
```

U-Net 的：

```text
predict()
predict_with_model()
```

也统一走 array-only boundary。

文件读取属于 data/experiment layer 的职责，不属于 Method。

### 2.5 修正四：E3 正式覆盖完整 test variants

`MAX_TEST_SAMPLES` 设为：

```text
0
```

含义是正式默认跑完 held-out test base 的全部 6 个 variants，不再只取 manifest 前 3 行。

smoke/debug 可以保留显式子集开关，但正式默认必须是 full test。

### 2.6 Stage 2.1 冻结结论

到 Stage 2.1，以下内容被认为可以冻结：

- synthetic 输出目录的干净性；
- low/high geometry pairing；
- Method 的 array-only 输入边界；
- E3 的 full-test 默认行为；
- scientific float path 不 clip。

---

## 3. Stage 3：Global Configuration Lock 的第一次完整结构化

### 3.1 基线

- Base ZIP：`pinn-starlight-synthetic-stage2_1.zip`
- SHA-256：`3f285e378aa726e4817be564184655340cbc9123a6e5ff61ebe409b71f0d9b1d`

Stage 2.1 的 synthetic protocol 和 array-only Method boundary 保持不变。

### 3.2 为什么需要 Global Configuration Lock

实验公平性不能依赖“大家自觉不看 test”。

真正可靠的方式是让代码结构形成：

```text
train/validation selection
        ↓
Global Configuration Lock
        ↓
held-out test
```

一旦 test 被打开，方法配置就不应该再根据 test 表现改变。

### 3.3 Stage 3 的 Global Lock 做了什么

新增：

```text
experiments/scripts/config_lock/run.py
```

当时运行顺序是先 E1，再执行 Global Lock。

Global Lock 只读取 train/validation synthetic arrays，不加载 test sample arrays，并冻结：

- Gaussian sigma；
- 三个 validation-approved U-Net checkpoints；
- 由 validation loss 选出的 E4 U-Net checkpoint；
- 已由 E1 选择的 PINN configuration。

同时记录并下游验证：

- manifest SHA-256；
- U-Net checkpoint SHA-256。

### 3.4 E2 被改成 test-only

Stage 3 之后，E2 不再负责：

- 加载 train/validation；
- Gaussian sigma selection；
- U-Net training；
- test 后再选 checkpoint。

E2 只做：

```text
read global_configuration_lock.json
        ↓
open held-out test split
        ↓
evaluate frozen methods/checkpoints
```

三个 frozen U-Net seed checkpoints 都可以在 test 上被评价，但 test 结果不能反过来改变“哪个 checkpoint 被视为冻结选择”。

### 3.5 E3/E4 共用同一把锁

- E3 从 Global Lock 读取 PINN config/seeds；
- E4 直接使用 frozen Gaussian sigma；
- E4 使用 validation-selected U-Net checkpoint；
- E4 使用 frozen inference policy 和 PINN config；
- E4 不再依赖旧的 post-test `locked_e2_config.json`。

### 3.6 U-01 / U-02 正式实施

U-Net 训练：

```text
random 256×256 crops
base_channels = 16
batch_size = 4
learning_rate = 1e-3
max = 12 × 100 steps / seed
patience = 3
3 seeds
CUDA AMP
min_delta_relative = 0.002
```

`min_delta_relative=0.002` 的作用是：

- 极小的 validation 改善不再无限重置 early stopping；
- 但数值上最优 checkpoint 仍保存。

正式 inference policy：

```text
mode = auto
```

先尝试：

```text
full-image forward
```

图像先 pad 到 16 的倍数。

只有发生 CUDA OOM 时，才退回：

```text
tile = 256
overlap = 64
mean stitching
```

### 3.7 Stage 3 故意不开放 test

Stage 3 当时还没有完成 Fourier-LS，所以 Global Lock 中保留：

```text
fourier_ls_pending = true
formal_test_ready = false
```

E2/E3/E4 故意拒绝正式运行。

这是非常重要的工程思想：**缺一个正式方法时，不是假装三方法版本已经“可以先跑”，而是让 test gate 直接关闭。**

当时执行逻辑：

```text
synthetic data frozen
        ↓
E1
        ↓
config_lock/run.py
        ↓
Global Lock infrastructure
        ↓
formal_test_ready = FALSE
        ↓
等待 Fourier-LS 完整冻结
        ↓
E2 / E3 / E4
```

---

## 4. Stage 4：Fourier-LS 方法闭环与四方法正式锁

### 4.1 基线

- Base ZIP：`pinn-starlight-global-lock-stage3.zip`
- SHA-256：`89cc147dcae69690f609abebb5abdd6989409d80e7d186259c0d387e44d98ff4`

Stage 2.1 的 array-only boundary 和 Stage 3 的 Global Lock architecture 保持不变。

### 4.2 Stage 4 的核心目标

Stage 3 中 Fourier-LS 还只是 pending。

Stage 4 把它真正变成第四个正式 Method，并要求：

- `lambda_LS` 只能在 validation 上选择；
- E2/E4 正式运行四方法；
- Global Lock 只有在四方法都冻结后才能写出；
- 当时 schema 升到 v2；
- `formal_test_ready=true` 之后才允许打开 test。

### 4.3 Fourier-LS 不允许外部注入 source

旧的通用接口存在类似：

```text
source=None
```

这意味着 runner 可能向 Method 注入：

- synthetic generation truth；
- PINN 参数；
- 任意 forcing。

Stage 4 删除这种注入口。

Fourier-LS 的正式 Method 只接受：

```text
observed ndarray + frozen configuration
```

### 4.4 每张图的正式 Fourier-LS 流程

```text
observed ndarray only
        ↓
deterministic bright-region center estimate
        ↓
固定 center (x_c, y_c)
        ↓
Differential Evolution 搜索
phi = (sigma_x, sigma_y, theta)
        ↓
对每个候选 phi：
    构造 I_city(phi) = q(r; alpha) G_phi
        ↓
    inner periodic Fourier least-squares
    求 B*(phi)
        ↓
    计算 J(B*(phi), phi)
        ↓
得到 phi_hat, B_hat
```

### 4.5 Stage 4 冻结的 Fourier-LS 设置

```text
alpha = 0.5

sigma_x, sigma_y ∈ [0.05, 0.8]

theta ∈ [-pi/2, pi/2]

DE:
  popsize = 8
  maxiter = 30
  seed = 20260728
  polish = True
  tol = 0.01

center initializer:
  Gaussian blur
  + 95th-percentile bright-region centroid

boundary:
  periodic

scientific output:
  raw float background
  no [0,1] clipping
```

### 4.6 Laplacian 坐标尺度

离散 Laplacian 与 PINN 使用同一个归一化坐标系：

```text
h_x = 2/(W-1)
h_y = 2/(H-1)
```

Fourier normal equation 明确使用 adjoint 形式。

在当前 periodic symmetric screened-Laplacian 下，Fourier symbol 为实数，因此可化成此前推导的 closed form。

### 4.7 `lambda_LS` 与 PINN physics weight 是两件事

Fourier-LS 的全局权重：

```text
lambda_LS
```

与 PINN 的：

```text
physics_weight
```

没有任何“同一个参数”的含义，不能因为都叫 weight 就混在一起。

Stage 4 的初始 validation candidates：

```text
0.01
0.1
0.3
0.4
0.5
```

每个候选都必须跑完整的：

```text
observed
  -> phi_hat
  -> I_city
  -> B_hat
```

然后用 validation background MAE 选择。

如果最优值出现在初始区间边界，只额外扩一个候选：

```text
下边界 -> 0.001
上边界 -> 1.0
```

而不是临时打开一大片新 grid。

### 4.8 Stage 4 Global Lock 当时记录的内容

- Gaussian sigma；
- Fourier-LS weight；
- alpha；
- center rule；
- phi bounds；
- DE settings；
- periodic boundary；
- validation candidate table；
- 三个 validation-approved U-Net checkpoints；
- validation-selected E4 checkpoint；
- E1 PINN configuration；
- test seeds；
- manifest/checkpoint fingerprints；
- selection 阶段不能使用 test 的边界声明。

只有四个方法全部冻结后才写 lock。

### 4.9 E2 / E4 四方法化

正式 benchmark 方法：

```text
FFT-Gaussian
Fourier-LS
U-Net-small
PINN
```

Fourier-LS 的 runtime 计入：

- center estimation；
- 全部 DE evaluations；
- 全部 FFT solves。

E2 保存每图：

- Fourier-LS estimated parameters；
- optimizer diagnostics。

E4 也运行同样四方法，并保存 Fourier-LS diagnostics。

所有 background predictions 在科学指标计算时不 clip。

另外记录：

```text
below-zero background fraction
above-one background fraction
```

作为诊断，而不是把它们强行裁掉。

### 4.10 Stage 4 验证

新增 `tests/test_fourier_ls.py`，检查：

- 正式 Method 没有 external `source`/`I_city` argument；
- normalized periodic Fourier symbol 与 spatial screened-Laplacian 一致；
- inner Fourier solution 满足 least-squares normal equation；
- DE deterministic；
- 返回 phi diagnostics。

Stage 4 的四个 Fourier-LS tests 通过。

但当时源码包没有正式 synthetic dataset、E1 lock 或训练后的 U-Net checkpoints，所以实际选中的 `lambda_LS`、最终 Global Lock JSON 和论文数值必须通过正式实验运行产生。

---

## 5. Stage 5：Fourier-LS 接口、命名与调用链收口

### 5.1 基线

- Base ZIP：`pinn-starlight-fourier-ls-stage4.zip`
- SHA-256：`e8da7f302af96e7e0cff151778e3895091c40bb9f84c6374672fb2e025dc547d`

Stage 4 的数学、periodic FFT operator、逐图 DE search 和四方法 benchmark 设计保持不变。

### 5.2 参数命名统一成 `lambda_ls`

正式调用链统一为：

```text
validation
  -> select_lambda_ls(...)
  -> selected_lambda_ls
  -> Global Lock methods.fourier_ls.lambda_ls
  -> E2 / E4
  -> estimate_background(observed, lambda_ls)
```

从 Fourier-LS 正式链中删除旧命名：

```text
weight
selected_weight
initial_weight_candidates
physics_weight   # 仅指旧 Fourier-LS 内部误名
```

PINN 自己的 `physics_weight` 保持不变，因为它是另一个独立超参数。

### 5.3 正式 Method 接口缩窄

公共接口固定为：

```python
estimate_background(observed, lambda_ls, *, return_info=False)
```

E2/E4 不能传：

- source / I_city；
- phi；
- center；
- truth metadata；
- alpha；
- center settings；
- DE settings。

所有逐图 source quantities 必须由当前 `observed` 在 Method 内部重新估计。

validation 也调用同一个 public Method。

`background_true` 只能在 Method 输出以后，由 evaluator 用来给候选 `lambda_ls` 打分。

### 5.4 Stage 5 的 Global Lock schema v3

当时写：

```text
schema_version = 3
```

canonical fields：

```text
methods.fourier_ls.lambda_ls
methods.fourier_ls.initial_lambda_ls_candidates
```

formal test 要求 schema >= 3。

Stage 4 的 schema-v2 lock 不能 silently translate，必须重建。

E2/E4 inference 前还要检查 lock 中记录的：

- alpha；
- center rule；
- phi bounds；
- DE settings；
- periodic boundary；
- coordinate system；
- no-clipping policy；
- deterministic flag；

是否与正式 implementation 一致。

### 5.5 Stage 5 验证范围

测试覆盖：

- public signature；
- legacy-key rejection；
- schema-v3 gating；
- periodic operator equivalence；
- normal-equation correctness；
- deterministic DE；
- canonical validation-selection names；
- E2/E4/config-lock call chain。

Stage 5 仍然没有在打包环境里产生真实 `lambda_ls*`、最终 lock 或论文数字，因为缺少正式 dataset / E1 artifact / U-Net checkpoints。

---

## 6. Stage 6：Evaluation 与星点指标闭环

### 6.1 这一阶段不是改 Method，而是改评价链

Stage 6 的重点是回答一个很关键的问题：

> “方法输出的 background/residual 到底以什么数据形式进入 MAE、PSNR、SSIM、star detection 和 flux photometry？”

### 6.2 raw residual 是唯一科学 residual

每个 Method 只收到：

```text
observed + pre-test frozen configuration
```

公共 reconstruction：

```text
raw_residual = observed - background_pred
```

并且永远不 clip。

### 6.3 Background / residual 指标全部走 raw float

以下指标都在 raw float arrays 上计算：

```text
MAE
RMSE
PSNR
SSIM
```

PSNR/SSIM 使用：

```text
data_range = 1.0
```

这里 `data_range=1.0` 是统一尺度定义，不等价于“先 clip 到 [0,1]”。

如果 perfect prediction 导致：

```text
MSE = 0
PSNR = +inf
```

这是正常数学结果，不应人为改成有限值。

### 6.4 Detection 与 photometry 解耦

Stage 6 当时 formal detector view 定义为：

```text
detection_image = max(raw_residual, 0)
```

没有 upper clipping。

DAOStarFinder 只用于：

- detection；
- centroid。

它内部的 detector flux **不作为论文 flux metric**。

> Stage 7 后，formal detection 改为直接使用 raw residual；Stage 6 的 nonnegative detector view 只保留为历史记录。

### 6.5 matched-star flux 的正式定义

参考 flux：

- 在 raw `clean_true` 上；
- 做 local-background-subtracted circular aperture photometry。

预测 flux：

- 在 raw `residual_pred` 上；
- 使用同样的 aperture photometry 定义。

论文主 flux 指标统一为：

```text
matched_star_mean_relative_flux_error
```

中文：

**成功匹配星点的平均相对通量误差**

这个数只对“成功匹配”的星点计算。

没有匹配上的星点由：

```text
Precision / Recall / F1
```

负责表达。

如果某张图没有有效 matched flux：

```text
flux metric = NaN / blank
```

而不是伪造一个 `1.0` 或其它惩罚值。

### 6.6 star_reference schema

`star_reference` 只从 reference / clean base 生成一次，并由所有 Methods 共用。

Stage 6 metadata 记录：

- detection rule；
- photometry definition；
- exact cap status；
- `evaluation_schema_version=2`。

### 6.7 “检测到 2000 颗星”不等于“真的有 2000 颗星”

如果算法设定的最大检测数量是 2000，那么输出正好等于 2000 只能说明“触及上限的可能性很高”，不能直接说真实检测数就是 2000。

因此 generator 会额外 probe 一个 detection，显式记录：

```text
count_capped
```

只有它为真，才能说 catalogue 因上限被截断。

### 6.8 数据兼容性

Stage 5 synthetic metadata 没有 Stage 6 photometry schema。

因此正式运行 Stage 6 以后，旧 synthetic dataset 必须重新生成。

`evaluate_synthetic()` 会主动拒绝旧 `star_reference` metadata，避免两套评价协议混在一起。

### 6.9 Truth boundary 不变

以下 truth：

```text
background_true
clean_true
generation metadata
```

仍只存在于：

```text
data / evaluator layer
```

Method 只能在输出完成以后被 truth 打分。

Stage 6 修改 evaluator，不修改四个 Method 的数学定义，因此当时 Method-side Global Lock schema 仍为 v3。

---

## 7. Stage 7：观测模型最终冻结与 detection debug

### 7.1 新事实带来的科学协议修正

项目确认 reference 图像本身来自相机实拍，而且可能来自不同相机。

因此旧变量名 `clean_true` 容易让人误以为它是“无噪声 ground truth”。

Stage 7 明确：

- legacy 变量 `clean_true` 实际代表 camera-acquired reference sky image；
- 它已经包含自己的 acquisition characteristics；
- `background_true` 才是我们精确知道的、人工叠加的 skyglow / light-pollution component。

### 7.2 最终半合成观测模型

正式冻结：

\[
I_{\mathrm{obs}} = I_{\mathrm{ref}} + B_{\mathrm{true}}.
\]

不再额外注入：

```text
Gaussian noise
Poisson noise
Poisson-Gaussian noise
```

原因不是“真实相机没有噪声”，而是：

1. reference 本身已经含有采集特性；
2. 缺少逐图 radiometric calibration / sensor calibration；
3. 如果继续人为模拟额外 skyglow-induced shot noise，会产生伪精确。

因此：

```text
noise_std
noise_seed
```

从正式 generator / manifest protocol 中移除。

项目研究目标被明确为：

```text
background estimation / light-pollution removal
```

而不是 denoising benchmark。

low/high 只表示 skyglow intensity 与 spatial scale 的差异，不再暗含不同 synthetic noise level。

### 7.3 Formal detection 改成 raw

scientific residual 和 aperture photometry 始终保持 raw float。

Stage 7 的正式 detection：

```text
raw
```

即直接使用 raw residual，保留负值。

只在 validation/debug 中提供：

```text
nonnegative:
  max(raw, 0)

legacy_01:
  clip(raw, 0, 1)
```

### 7.4 `compare_star_detection_modes()`

这个工具对：

```text
raw
nonnegative
legacy_01
```

使用完全相同的：

- frozen reference catalogue；
- threshold；
- FWHM；
- matching radius；
- count cap。

但 matched-star flux 始终回到同一张 **raw residual** 上重新测量。

也就是说 detection preprocessing 不能偷偷改变 photometry 数据。

### 7.5 Validation-only debug CLI

提供：

```text
experiments/scripts/debug_detection_modes/run.py
```

它只允许 validation samples，并主动拒绝 test split。

### 7.6 schema 升级

Stage 7 重新生成的数据使用：

```text
evaluation_schema_version = 3
```

旧 schema-v2 metadata 被明确拒绝。

### 7.7 Stage 7 后的推荐顺序

```text
1. regenerate synthetic dataset
2. 如有需要，在 validation residual 上比较
   raw / nonnegative / legacy_01 detection
3. run E1
4. build Global Configuration Lock
5. 只有 pre-test choices 冻结后才 run E2/E3/E4
```

---

## 8. Stage 8：真实数据目录接入与冻结审计

### 8.1 基线

Stage 8 基于 Stage 7 Observation Model Closure。

记录的基线 SHA-256：

`505f6988a7c021fe260b25945c110abdf9233ced9b45f0de32dd56f17c029beb`

### 8.2 数据根目录适配

支持三种布局：

```text
1. 项目根目录直接有 collections/ + test/
2. legacy data/ 布局
3. 外置数据根目录：
   PINN_STARLIGHT_DATA_ROOT
```

### 8.3 legacy synthetic 隔离

旧：

```text
collections/synthetic/
```

被明确标为 legacy。

Stage 7 正式 30-sample 数据默认生成到：

```text
collections/synthetic_formal/
```

不能覆盖旧目录，也不能把旧样本算进正式数据。

### 8.4 E4 的真实图职责冻结

E4 只使用：

```text
collections/real_high_pollution/
```

中的 16 张真实图。

而：

```text
test/native_test.tif
```

只保留给 E0 / smoke test。

这两个用途不能混。

### 8.5 flux NaN 聚合修复

只有 flux metric 在 group mean/std 聚合时忽略 non-finite values，并额外记录 valid count。

missed detections 仍然由 Star F1 表达，不能因为 flux NaN 被“抹掉”。

### 8.6 preflight runner

新增 read-only preflight，用来检查：

- 5 个 clean bases；
- 30-sample plan；
- 18/6/6 split；
- 16 张 E4 images；
- native test image；
- legacy/formal synthetic separation。

### 8.7 Stage 8 冻结审计：Method truth isolation

审计结论：PASS。

正式 Method 调用只收到：

```text
observed + frozen method configuration
```

truth arrays 只在 data/evaluator layer。

U-Net 是唯一 supervised baseline，所以 `background_true` 只在 pre-test train/validation loader 中可见。

### 8.8 U-Net train/test separation

审计结论：PASS。

Stage 8 当时：

- `config_lock/run.py` 负责 U-Net training + validation checkpoint selection；
- E2 不训练；
- E2 只在 formal-ready lock 后打开 test；
- E4 加载 validation-selected checkpoint。

> Stage 9 后，这个职责再次被拆分：U-Net training/selection 移出 Global Lock，Global Lock 只做验证、聚合和哈希。

### 8.9 Checkpoint/config freeze

审计结论：PASS。

lock 记录：

- 三个 U-Net checkpoint SHA-256；
- synthetic manifest SHA-256。

formal runners 使用前验证 fingerprints。

### 8.10 “冻结 U-Net”不是冻结参数梯度

这一点很容易被误解。

正式 supervised U-Net training 时：

```text
所有网络参数都正常训练
```

没有把 encoder/layer 设成：

```text
requires_grad = False
```

所谓“frozen U-Net”是指：

```text
validation 选定 checkpoint
        ↓
checkpoint 被 hash/freeze
        ↓
test 阶段只能加载，不再根据 test 改
```

不是训练期间冻结某些 layer。

### 8.11 Source snapshot 的边界

Global Lock 能保护：

- config；
- artifact；
- checkpoint fingerprints。

但它不是 DRM。

如果 lock 之后改了正式源码，再把新代码结果与旧 lock/test run 混在一起，就破坏了可复现性。

因此 Stage 8 建议保留：

```text
Stage-8 ZIP
Git commit
```

作为 source snapshot。

---

## 9. Stage 9：四方法职责重构与最终冻结架构

### 9.1 Stage 9 的性质

Stage 9 不是重新发明科学设置，而是对 Stage 8 已审计版本做 architecture refactor。

目标是让代码结构真正对应此前确定的四层设计：

1. experiment config；
2. information boundary；
3. module responsibility；
4. narrow method interface。

### 9.2 唯一正式配置源

最终统一到：

```text
experiments/configs/formal_experiment.py
```

PINN、FFT-Gaussian、Fourier-LS、U-Net 的正式设置都从这里来。

不能再把真正的“实验定义”散落在：

- runner；
- selector；
- Method；
- Global Lock；
- checkpoint loader。

### 9.3 四个方法的职责模式统一

Stage 9 最终结构：

```text
formal_experiment.py
        ↓
method-local selection / training
(train + validation only)
        ↓
selected artifact
        ↓
Global Configuration Lock
(verify + aggregate + hash only)
        ↓
E2 / E3 / E4
```

### 9.4 PINN

PINN selection 仍由 E1 完成，但输出 method-local selected artifact。

### 9.5 Gaussian

Gaussian sigma selection 从 Gaussian Method implementation 中移出。

Method 本身只负责给定 frozen sigma 时执行 inference/math。

### 9.6 Fourier-LS

`lambda_ls` selection 从 Fourier-LS Method implementation 中移出。

Method 只做：

```text
observed + frozen lambda_ls
  -> per-image center/phi estimation
  -> Fourier-LS background
```

### 9.7 U-Net

U-Net train/validation/checkpoint selection 从 Global Lock 中移出。

Global Lock 不再是 trainer。

### 9.8 Global Lock 变成纯 freeze gate

Stage 9 的：

```text
config_lock/run.py
```

只负责：

```text
verify
aggregate
hash
freeze
```

不再：

```text
train U-Net
select Gaussian
select Fourier-LS
```

### 9.9 method-local config hash 与局部重建

每个方法都有自己的 config/source dependency。

因此如果只改 U-Net：

```text
U-Net selected artifact invalid
```

但只要 PINN/Gaussian/Fourier-LS 的：

- method-local config；
- manifest/split IDs；
- source dependencies；

都没变，它们的 selected artifacts 可以继续复用。

这就是 Stage 9 “修改一个方法，不必把所有 pre-test selection 全重跑”的基础。

### 9.10 Global Lock schema 4

最终 lock 冻结：

- formal config snapshot；
- 四个 selection artifact hashes；
- manifest hash；
- U-Net checkpoint hashes；
- formal source snapshot。

### 9.11 E2 / E4 的最终职责

E2：

```text
test-only
```

不能训练 U-Net，不能在 test 上重新 selection。

E4：

```text
只使用 collections/real_high_pollution/
```

### 9.12 E4 runtime 修复

Stage 9 修复了一个早已存在的 E4 runtime 问题：

```text
METHODS 被引用但未定义
```

最终统一从 central formal method order 获取。

### 9.13 正式命令统一成 module execution

不再依赖从 raw ZIP checkout 中直接运行嵌套脚本路径。

统一使用：

```powershell
uv run python -m ...
```

这样：

- `experiments/`；
- 已安装的 `src/` package；

都能在统一 import 语义下解析。

### 9.14 Stage 8 → Stage 9 科学数值保持不变

Stage 9 明确保持：

#### U-Net

```text
patch_size = 256
epochs = 12
steps_per_epoch = 100
batch_size = 4
base_channels = 16
learning_rate = 1e-3
patience = 3
min_delta_relative = 0.002
amp_requested = True
mode = auto
tile_size = 256
overlap = 64
3 formal seeds
```

#### PINN

E1 candidates 与 training budget 保持 Stage 8 数值。

#### Gaussian

sigma candidates 保持 Stage 8 数值。

#### Fourier-LS

保持：

- lambda candidates；
- boundary extension rule；
- DE budget；
- `alpha=0.5`；
- periodic boundary。

#### Observation / Evaluation

继续使用 Stage 7 冻结的：

```text
I_obs = I_ref + B_true
no additional synthetic noise
raw scientific residual
raw formal detection
raw aperture photometry
evaluation schema v3
```

### 9.15 Stage 9 自动验证结果

当时记录：

```text
compileall: PASS

pytest:
  33 passed
  1 existing warning
```

并且 Stage 8 → Stage 9 的 formal numeric parity 检查通过，覆盖：

- PINN candidates/budget；
- Gaussian candidates；
- Fourier-LS method/selection settings；
- U-Net training/inference settings。

打包时没有执行完整 formal training/test，因为上传内容只有源码包和目录信息，不含完整实际图像数据。

---

# 第二部分：最终职责、信息边界与实验规则

## 10. Stage 9 最终四层规则

### 10.1 第一层：实验配置

唯一正式配置源：

```text
experiments/configs/formal_experiment.py
```

这里定义的是“这一次 formal experiment 到底是什么”。

### 10.2 第二层：信息权限

Method 正式运行时：

```text
只允许 observed + frozen config/checkpoint/seed
```

不允许看到：

```text
background_true
clean/reference truth
generation metadata
star_reference construction internals
test-driven selection information
```

唯一 supervised baseline U-Net 可以在 **train/validation 阶段**读取监督 target，但不能把 test truth 反向用于 selection。

### 10.3 第三层：模块职责

```text
Method:
  数学 / inference

Selector:
  train/validation selection

Global Lock:
  verify + aggregate + hash + freeze

Evaluator:
  truth-based scoring after Method output

Test runner:
  held-out test execution only
```

### 10.4 第四层：最小接口

以 Fourier-LS 为典型：

```python
estimate_background(observed, lambda_ls, *, return_info=False)
```

接口越窄，就越难从 runner “偷偷塞 truth”。

同样的思想适用于其它 Method：runner 负责组织实验，不应该控制 Method 内部本应自行估计的物理/source 参数。

---

# 第三部分：Stage 9 正式实验运行手册

## 11. 正式运行：Windows / PowerShell 从零开始

### 11.1 不需要先改代码

对于已经冻结的这一版 formal experiment：

> **不要为了“先跑起来”临时改任何方法代码或参数。**

从项目根目录使用 module execution。

### 11.2 第一次同步环境

```powershell
uv sync
```

### 11.3 数据在项目外时设置根目录

只有当：

```text
collections/
test/
```

不在项目根目录时才需要：

```powershell
$env:PINN_STARLIGHT_DATA_ROOT = "D:\path\to\data-root"
```

### 11.4 第一次 preflight：检查原始数据接入

```powershell
uv run python -m experiments.scripts.preflight.run
```

它应该检查：

- clean reference bases；
- E4 real images；
- native smoke image；
- data root；
- legacy/formal synthetic separation。

### 11.5 生成正式 synthetic dataset

```powershell
uv run python -m experiments.common.data.generate_synthetic
```

Stage 7 以后正式生成模型是：

```text
I_obs = I_ref + B_true
```

没有额外 synthetic noise。

formal synthetic 默认目录：

```text
collections/synthetic_formal/
```

旧：

```text
collections/synthetic/
```

只作为 legacy，不参与正式 30 样本 benchmark。

### 11.6 第二次 preflight：检查正式 synthetic 数据

```powershell
uv run python -m experiments.scripts.preflight.run
```

此时还会验证：

```text
30 samples
18 / 6 / 6 split
```

也就是：

```text
train = 18
validation = 6
test = 6
```

### 11.7 生成四个独立的 pre-test selected artifacts

#### PINN

```powershell
uv run python -m experiments.scripts.e1.run
```

#### FFT-Gaussian

```powershell
uv run python -m experiments.scripts.selection.gaussian.run
```

#### Fourier-LS

```powershell
uv run python -m experiments.scripts.selection.fourier_ls.run
```

#### U-Net

```powershell
uv run python -m experiments.scripts.selection.unet.run
```

这四步只能使用：

```text
train + validation
```

不能用 held-out test 表现来决定参数。

### 11.8 建立纯 Global Configuration Lock

```powershell
uv run python -m experiments.scripts.config_lock.run
```

Stage 9 的 Global Lock 不训练、不 selection。

它只做：

```text
verify selected artifacts
verify central config
verify manifest/splits
verify source dependencies
verify checkpoint hashes
aggregate
hash
freeze
```

### 11.9 Lock 验证通过后才能打开正式 test

```powershell
uv run python -m experiments.scripts.e2.run
uv run python -m experiments.scripts.e3.run
uv run python -m experiments.scripts.e4.run
```

### 11.10 总顺序

```text
uv sync
        ↓
preflight
        ↓
generate synthetic
        ↓
preflight
        ↓
E1 / PINN selection
Gaussian selection
Fourier-LS selection
U-Net selection
        ↓
Global Configuration Lock
        ↓
E2
E3
E4
```

---

## 12. 正式参数到底改哪里

只改：

```text
experiments/configs/formal_experiment.py
```

不要为了改变 formal experiment，把参数散落修改到：

- Method source；
- E2 runner；
- E4 runner；
- Global Lock；
- checkpoint loader；
- evaluator。

### 12.1 U-Net 冻结值

```python
patch_size = 256
epochs = 12
steps_per_epoch = 100
batch_size = 4
base_channels = 16
learning_rate = 1e-3
patience = 3
min_delta_relative = 0.002
amp_requested = True
mode = "auto"
tile_size = 256
overlap = 64
```

### 12.2 Fourier-LS 关键固定设置

```text
alpha = 0.5

sigma_x, sigma_y in [0.05, 0.8]
theta in [-pi/2, pi/2]

DE:
  popsize = 8
  maxiter = 30
  seed = 20260728
  polish = True
  tol = 0.01

boundary = periodic
```

lambda 的 initial validation candidates：

```text
0.01, 0.1, 0.3, 0.4, 0.5
```

边界扩展：

```text
lower -> 0.001
upper -> 1.0
```

只在最优值落在初始搜索边界时额外扩一个点。

---

## 13. 只改一个方法，应该重跑什么

这个规则只适用于：

> **还没有使用 E2 test 反馈来决定新参数的情况。**

### 13.1 例：只改 U-Net

```powershell
uv run python -m experiments.scripts.selection.unet.run --force
uv run python -m experiments.scripts.config_lock.run --force
```

PINN / Gaussian / Fourier-LS 可以继续复用，但前提是它们自己的：

- config；
- manifest；
- split IDs；
- method-local source dependencies；

仍然验证一致。

### 13.2 四个 selector 对应关系

```text
PINN
  -> experiments.scripts.e1.run

Gaussian
  -> experiments.scripts.selection.gaussian.run

Fourier-LS
  -> experiments.scripts.selection.fourier_ls.run

U-Net
  -> experiments.scripts.selection.unet.run
```

### 13.3 修改 Gaussian

```powershell
uv run python -m experiments.scripts.selection.gaussian.run --force
uv run python -m experiments.scripts.config_lock.run --force
```

### 13.4 修改 Fourier-LS

```powershell
uv run python -m experiments.scripts.selection.fourier_ls.run --force
uv run python -m experiments.scripts.config_lock.run --force
```

### 13.5 修改 PINN

PINN selection 由 E1 负责，因此需要重新生成 PINN selected artifact，再重建 Global Lock。

---

## 14. Test-feedback rule：最不能破坏的一条规则

重新生成一把 Global Lock **不代表**可以把 test 当 validation 用。

如果你已经看到 E2 test 结果，然后因为 test 表现不满意而修改：

- lambda；
- sigma；
- U-Net budget；
- PINN hyperparameter；
- detection choice；
- checkpoint choice；

那么这已经是 test feedback。

正确处理只能是：

1. 参数变更由 train/validation 或新 protocol 证明；
2. 或者明确把整个实验定义成一个新的 protocol；
3. 不能把同一 held-out test 继续称作“完全未参与 selection 的 test”。

---

# 第四部分：评价与结果解释

## 15. Background / residual 指标

正式科学 arrays：

```text
background_pred: raw float
raw_residual = observed - background_pred
```

不做 `[0,1]` clipping。

在 raw arrays 上计算：

```text
MAE
RMSE
PSNR
SSIM
```

`data_range=1.0` 只是统一尺度定义。

### 15.1 PSNR = +inf

当：

```text
MSE = 0
```

时：

```text
PSNR = +inf
```

是数学上正确的，不是代码 bug。

---

## 16. 星点 detection 与 flux

### 16.1 正式 detection

Stage 7 最终：

```text
raw residual
```

### 16.2 validation/debug 对照

```text
nonnegative = max(raw, 0)
legacy_01 = clip(raw, 0, 1)
```

只用于 validation/debug，不应在 formal test 上临时“看哪个好”。

### 16.3 flux 不使用 DAOStarFinder 的 detector flux

DAOStarFinder 只负责：

- detection；
- centroid。

论文 flux metric 必须在 raw arrays 上重新做 local-background-subtracted aperture photometry。

### 16.4 matched-star flux metric

正式名称：

```text
matched_star_mean_relative_flux_error
```

中文：

**成功匹配星点的平均相对通量误差**

它必须与：

```text
Star Precision
Star Recall
Star F1
```

联合解释。

只看 matched flux error 会漏掉“很多星根本没匹配到”的情况。

### 16.5 没有有效 matched flux

写：

```text
NaN / blank
```

同时记录 valid count。

不要伪造：

```text
1.0
```

### 16.6 2000 颗星的问题

如果 catalogue 上限是 2000：

```text
count == 2000
```

不能直接描述为“检测到正好 2000 颗星”。

更准确的写法是：

```text
检测数量达到预设上限 2000
```

并结合：

```text
count_capped
```

判断是否真的发生截断。

---

# 第五部分：数据职责与正式目录

## 17. 正式 synthetic 数据

默认：

```text
collections/synthetic_formal/
```

旧：

```text
collections/synthetic/
```

是 legacy，不参与正式 benchmark。

### 17.1 30 样本结构

5 个 reference bases × 6 variants：

```text
5 × 3 background types × 2 levels = 30
```

split：

```text
18 train
6 validation
6 test
```

---

## 18. E4 真实数据

E4 只读取：

```text
collections/real_high_pollution/
```

正式记录为 16 张图。

`test/native_test.tif` 不是 E4 paper data，它只用于 smoke / E0。

---

## 19. 外置数据根目录

如果项目目录外单独保存：

```text
collections/
test/
```

设置：

```powershell
$env:PINN_STARLIGHT_DATA_ROOT = "D:\path\to\data-root"
```

否则不需要设置。

---

# 第六部分：四方法最终边界

## 20. FFT-Gaussian

职责：

```text
observed + frozen sigma
  -> background_pred
```

sigma 由独立 validation selector 选择。

Method 本身不负责 selection。

---

## 21. Fourier-LS

职责：

```text
observed + frozen lambda_ls
        ↓
internal center estimate
        ↓
internal phi search
        ↓
internal I_city(phi)
        ↓
periodic Fourier-LS
        ↓
background_pred
```

禁止 runner 注入：

```text
source
I_city
phi
center
truth metadata
alpha
DE settings
```

### 21.1 逐图 phi 与全局 lambda 的区别

```text
lambda_ls
```

是 validation 选出的全局 formal hyperparameter。

而：

```text
phi_hat = (sigma_x, sigma_y, theta)
```

是 Method 对每张当前 observed 图自行估计的内部量。

因此 `phi_hat` 不属于 Global Lock 的“逐图固定答案”，它是算法推理的一部分。

---

## 22. U-Net-small

训练：

```text
supervised
train/validation only
all parameters trainable
```

selection：

```text
method-local selector
```

test：

```text
load frozen checkpoint
no retraining
no test-driven checkpoint choice
```

inference：

```text
full image first
OOM -> tiled fallback
```

tile fallback：

```text
tile = 256
overlap = 64
mean stitching
```

---

## 23. PINN

E1 负责 pre-test selection，并写 method-local artifact。

PINN 的 `physics_weight` 是 PINN 自己的超参数，与 Fourier-LS `lambda_ls` 完全分离。

PINN Method 不再返回一套自己定义的 residual。

统一：

```text
background_pred
  -> common reconstruct_residual()
```

---

# 第七部分：常见错误与排查

## 24. 错误：在项目根目录外直接跑嵌套 `.py`

优先使用：

```powershell
uv run python -m experiments....
```

不要依赖：

```powershell
python some/deep/path/run.py
```

否则容易出现 `experiments/` 与 `src/` import path 不一致。

---

## 25. 错误：重新生成 synthetic 时目录不空

formal generator 默认应该拒绝把新 manifest 写进非空旧目录。

如果确实确认要重建：

```text
--force
```

使用前先确认目标确实是 formal synthetic 子目录，不要对项目其它路径做 destructive cleanup。

---

## 26. 错误：把 legacy synthetic 当正式数据

检查：

```text
collections/synthetic/
```

与：

```text
collections/synthetic_formal/
```

不要混用。

正式 Stage 7+ benchmark 用后者。

---

## 27. 错误：因为 raw residual 有负值就先 clip

不要。

scientific residual 必须保留 raw float。

只有 display PNG 可以 clip。

detection preprocessing 如果要比较，只能在 validation/debug 中通过固定工具比较，不能临时改 formal test。

---

## 28. 错误：把 U-Net “freeze”理解成不训练网络层

正式训练时所有参数正常优化。

freeze 指的是：

```text
validation-selected checkpoint
+ hash verification
+ test-stage immutability
```

---

## 29. 错误：看到 test 结果后再调参数，然后只重建 lock

这是 test leakage。

lock 只能证明“当前 artifact/config 的一致性”，不能把已经发生的 test feedback 洗掉。

---

## 30. 错误：把 Fourier-LS 的 `lambda_ls` 和 PINN `physics_weight` 混用

两者是独立方法的独立超参数。

正式代码命名已经刻意分开，文档也应保持这一点。

---

## 31. 错误：把 `phi_hat` 写成 Global Lock 选择出来的全局参数

不是。

`lambda_ls` 是全局 validation-selected hyperparameter。

`phi_hat` 是 Fourier-LS 对每张 observed 图在 inference 内部估计的逐图参数。

---

## 32. 错误：把 count=2000 写成“检测到 2000 颗星”

应该优先写：

```text
检测数量达到预设上限 2000
```

并查看：

```text
count_capped
```

---

# 第八部分：交接与审计摘要

## 33. 从 Stage 2 到 Stage 9，真正完成了什么

### Stage 2

封住 synthetic truth 旁路，统一 float path、star reference、residual reconstruction。

### Stage 2.1

解决 stale dataset、low/high geometry pairing、array-only Method boundary、E3 full-test 默认行为。

### Stage 3

建立 pre-test Global Configuration Lock 思想，把 E2 改成 test-only，并冻结 U-Net checkpoint policy。

### Stage 4

真正完成 Fourier-LS，使正式 benchmark 变成四方法，并在 validation 上选择 `lambda_LS`。

### Stage 5

收紧 Fourier-LS 接口，统一 `lambda_ls` 命名，消除调用链中的参数含义混乱。

### Stage 6

把 raw residual、PSNR/SSIM、star detection、matched-star photometry 和 NaN/count 语义统一。

### Stage 7

根据“reference 是相机实拍”的事实修正观测模型，取消额外 synthetic noise，并把 raw detection 冻结为正式路径。

### Stage 8

接入真实目录结构、隔离 legacy synthetic、冻结 E4 16 张真实图职责，并完成执行前审计。

### Stage 9

把四方法最终统一成：

```text
central config
  -> method-local selection
  -> selected artifact
  -> pure Global Lock
  -> formal test
```

并引入 schema 4、method-local hashes、formal source snapshot 与局部重建规则。

---

## 34. 最终实验架构一图看懂

```text
                    ┌─────────────────────────────────────┐
                    │ experiments/configs/formal_experiment.py │
                    │         唯一正式实验配置源          │
                    └──────────────────┬──────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
       PINN E1                  Gaussian selector              Fourier-LS selector
          │                            │                            │
          │                            │                            │
          └──────────────┬─────────────┴──────────────┬─────────────┘
                         │                            │
                         ▼                            ▼
                  selected artifacts           U-Net selector/training
                         │                            │
                         └──────────────┬─────────────┘
                                        ▼
                           Global Configuration Lock
                         verify + aggregate + hash only
                                        │
                          ┌─────────────┼─────────────┐
                          ▼             ▼             ▼
                         E2            E3            E4
                    held-out test   explanation   real images
```

---

## 35. 最终必须记住的六句话

1. **配置只在 `formal_experiment.py` 定义。**
2. **Method 只看 `observed + frozen config/checkpoint`。**
3. **selection 只能用 train/validation。**
4. **Global Lock 不训练、不 selection，只验证与冻结。**
5. **scientific background/residual/photometry 都保留 raw float，不为方便而 clip。**
6. **test 结果不能反过来决定同一实验的下一组参数。**

---

# 附录 A：本合并文档覆盖的原始 Stage 文件

本文件按内容合并并中文化了以下 Stage 文档：

```text
SYNTHETIC_STAGE2_NOTES.md
SYNTHETIC_STAGE2_1_NOTES.md
GLOBAL_CONFIGURATION_LOCK_STAGE3_NOTES.md
FOURIER_LS_STAGE4_NOTES.md
FOURIER_LS_STAGE5_INTERFACE_NOTES.md
STAGE6_EVALUATION_CLOSURE_NOTES.md
STAGE7_OBSERVATION_MODEL_NOTES.md
STAGE8_DATA_INTEGRATION_AUDIT.md
STAGE9_RESPONSIBILITY_REFACTOR_AUDIT.md
FORMAL_EXPERIMENT_RUN_GUIDE.md
```

中文环境包中的：

```text
README_中文环境说明.md
```

所规定的编码和中文化原则也已并入本文开头。

---

# 附录 B：各历史阶段的 superseded 关系

```text
Stage 2 / 2.1:
  additional Gaussian noise
        ↓ superseded by Stage 7
Stage 7:
  I_obs = I_ref + B_true
  no additional synthetic noise

Stage 6:
  detection_image = max(raw, 0)
        ↓ superseded by Stage 7
Stage 7:
  formal detection = raw residual
  nonnegative / legacy_01 = validation/debug only

Stage 3:
  Global Lock infrastructure, formal_test_ready=false
        ↓
Stage 4:
  schema v2, four-method lock
        ↓
Stage 5:
  schema v3, Fourier-LS canonical interface
        ↓
Stage 9:
  schema 4, method-local selection + pure freeze gate

Stage 8:
  U-Net training/selection still owned by config_lock
        ↓
Stage 9:
  U-Net selection moved out
  config_lock = verify + aggregate + hash only
```

---

# 附录 C：正式运行命令速查

```powershell
uv sync

# 可选：只有数据在项目外时设置
$env:PINN_STARLIGHT_DATA_ROOT = "D:\path\to\data-root"

uv run python -m experiments.scripts.preflight.run
uv run python -m experiments.common.data.generate_synthetic
uv run python -m experiments.scripts.preflight.run

uv run python -m experiments.scripts.e1.run
uv run python -m experiments.scripts.selection.gaussian.run
uv run python -m experiments.scripts.selection.fourier_ls.run
uv run python -m experiments.scripts.selection.unet.run

uv run python -m experiments.scripts.config_lock.run

uv run python -m experiments.scripts.e2.run
uv run python -m experiments.scripts.e3.run
uv run python -m experiments.scripts.e4.run
```

---

**文档结束。**
