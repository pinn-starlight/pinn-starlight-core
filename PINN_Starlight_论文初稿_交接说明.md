# PINN Starlight 论文初稿交接说明

## 1. 当前任务目标

这是一个面向星空摄影/星空图像的光污染背景抑制课题。核心目标不是“把天空变黑”，而是：

> 在尽量保留恒星等真实天体信号、尤其是星点位置与光度信息的前提下，估计并抑制非均匀光污染/天空背景。

当前项目已经经过多轮代码、实验协议和论文口径校正。下一步希望直接写一篇**中文论文初稿（可按 IEEE/竞赛论文结构组织）**。

最重要的原则：

1. **最终 Stage 10 源码和 Stage 10 总文档是事实源。**
2. 不要从早期 Stage 9、prototype、旧 README 或聊天中的临时设想反推最终实现。
3. 如果正式 E2–E4 结果尚未完整跑完，结果部分必须保留占位符或写成“待正式实验填入”，不能虚构数字。
4. 论文初稿可以先把 Introduction / Related Work / Problem Formulation / Method / Experimental Setup / Evaluation Protocol / Discussion 框架和正文写完整，再等正式结果补表格与结论。

---

## 2. 项目演化关系

项目不是从 Stage 10 凭空开始，完整逻辑链是：

**原始 latest → 四层职责/实验共识 → Stage 1 … Stage 9 → Stage 10**

Stage 9 文档属于历史审计快照，不能事后篡改；Stage 10 总文档继承 Stage 9 的审计主干，并继续说明 Stage 9 → Stage 10 的变化。

如果写论文，只需要写“最终科学方案”，不必把所有软件工程历史都塞进正文；但理解方法设计时必须知道 Stage 10 是对 Stage 9 的数据语义、物理源项和实验公平性的进一步修正。

---

## 3. Stage 10 最关键的数据语义

Stage 9 的一个核心问题是：原始 reference 图本身可能带有 native low-frequency background，而人工注入标签只对应额外加入的污染背景，导致监督目标和真实输入语义不完全一致。

Stage 10 改为：

\[
\hat B_{\rm native}=N(I_{\rm ref}),
\]

\[
I_{\rm clean}^{*}=I_{\rm ref}-\hat B_{\rm native},
\]

\[
I_{\rm obs}=I_{\rm clean}^{*}+B_{\rm added},
\]

\[
B_{\rm true}=B_{\rm added}.
\]

这里：

- \(N(\cdot)\) 是 native-background estimator（当前正式实现使用 Background2D 一类稳健二维背景估计思路）；
- \(I_{\rm clean}^{*}\) 只是经过 native-background correction 的参考图；
- **不能把它宣称为“真实无噪声 ground truth”。**

正式 benchmark **不再额外人为加入独立 Gaussian/Poisson noise**。原因是当前没有足够的相机 gain、exposure、electron/ADU、read-noise 等标定信息，无法把一个简单的噪声模型包装成真实相机噪声。

因此：

- 原始相机图自身已有真实采集噪声；
- `low/high` 指的是 skyglow/光污染注入强度或空间尺度等级；
- **不是 low-noise / high-noise。**

科学计算数组与显示图必须区分，不要为了显示而把正式数据随意 clip 或 min-max rescale。

---

## 4. 物理建模主线

观测图像可近似写为：

\[
I_{\rm obs}(x,y)=I_{\rm star}(x,y)+B(x,y),
\]

其中 \(B\) 是待估计的光污染/非均匀背景。

Stage 10 的核心物理约束采用 screened-Poisson 型方程：

\[
\nabla^2 B-\alpha B+S_{\rm city}=0.
\]

正式实验中：

\[
\alpha=0.5
\]

固定，不把它解释成从单张图像精确反演得到的真实大气物理常数，而是物理启发简化模型中的固定尺度参数。

非常重要：

- Stage 10 不再把源项简单解释为“非负城市辐亮度 \(I_{\rm city}\)”；
- 更准确的叫法是 **signed effective forcing \(S_{\rm city}\)**；
- 它是为了让 screened-Poisson 约束与所选背景基函数保持一致的“有效强迫项”，不等价于真实、可直接测量的城市光强。

源项/forcing 的构造带有 Garstang 型距离结构启发，并与椭圆 Gaussian envelope 结合。核心逻辑可概括为：

1. 先定义径向结构 \(b(r;\alpha)\)；
2. 由 screened-Poisson 关系构造
   \[
   q(r;\alpha)=\alpha b(r;\alpha)-\Delta b(r;\alpha);
   \]
3. 再与可表示方向性/椭圆扩散的 Gaussian envelope 组合；
4. 做统一 normalization；
5. 多个 component 线性组合形成 \(S_{\rm city}\)。

如果要写出 \(b(r;\alpha)\)、\(q(r;\alpha)\) 的完整展开式，必须以最终 Stage 10 Method / 数学文档和源码为准，不要从早期聊天临时公式猜。

---

## 5. 正式 PINN 的论文口径

正式 PINN 以 Stage 10 的 **formal coordinate PINN 实现**为准，旧 `src/.../main.py` 一类代码若仍存在，只能视为 legacy/prototype，不能混进论文的正式方法。

正式 PINN 的核心特征：

- 使用 coordinate-based neural field 表示背景；
- 背景输出使用适当约束（Stage 10 采用 sigmoid 类约束）；
- 联合使用数据项、PDE 物理残差和稀疏性约束：
  \[
  L=L_{\rm data}
  +\lambda_{\rm physics}L_{\rm physics}
  +\lambda_{\rm sparse}L_{\rm sparse};
  \]
- \(\alpha=0.5\) 固定；
- 多源 forcing 使用有限个 component；
- 最大模型容量支持到 \(K_{\max}=3\)；
- 但当前 multi-source synthetic 实际只注入 **2 个源**。

因此论文可以写：

> “the model supports up to three components”

但不能写：

> “the method has been validated on three-source scenes”

除非之后确实补做了三源实验。

正式实现还包含源中心的 observed-only 初始化与有限范围 refinement，以及 component weight / scale 等受约束优化。具体边界和训练超参数请从最终 Stage 10 配置/源码读取，不要凭本交接说明硬写。

PINN 在 E2/E3/E4 中是**逐图重新初始化并优化**的，不是把 E1 训练得到的 checkpoint 直接拿去测试。

---

## 6. 四种方法的定位

论文中至少包含以下四类方法：

### 1) FFT-Gaussian / Gaussian filtering baseline
经典、简单的平滑背景估计方法。

它在论文中应被定位为：

> classical filtering / low-frequency background baseline

不要和 Fourier least-squares 解法混为一谈。

### 2) Fourier-LS
physics-matched classical optimization baseline。

它与 PINN 尽量共享：

- forcing family；
- 组件数量上限；
- 初始化原则；
- 几何参数范围；
- normalization 规则。

区别主要在求解器：

- PINN：神经场 + 数据项 + PDE + 稀疏联合优化；
- Fourier-LS：Fourier 域 screened-Poisson 解 + least-squares / active-set 类权重求解。

这个 baseline 很重要，因为它能回答：

> 性能提升究竟来自物理模型本身，还是来自 PINN 的非线性神经表示与优化？

### 3) U-Net（内部历史标识 `unet_small`）
这是一个 **supervised image-to-image background regression baseline**，不是分割网络实验。

当前正式结构已恢复为标准容量四层 U-Net：`64→128→256→512→1024`，卷积块使用 ReLU，最后 `1×1 Conv` 输出单通道连续背景；训练目标仍为 **L1 background regression**。内部文件名和 method id 为兼容旧管线仍保留 `unet_small` / `UNetSmall`，论文和图表统一写 **U-Net**。

正式训练保持 NCHW + ordinary Adam + AMP；推理固定为 tile 256、overlap 64、tile batch 1。U-Net 能访问 paired synthetic target，而 PINN/Fourier-LS 不使用 paired target，这个信息权限差异应当在 Experimental Setup / Discussion 中如实说明。

### 4) PINN
逐图 physics-informed neural optimization，是论文的主方法。

---

## 6.1 Stage10-A 当前执行冻结口径

正式 E2/E3/E4 固定使用单可见 GPU，逻辑设备为 `cuda:0`。PINN 的正式 Laplacian backend 为 nested autograd；项目保留的 analytic backend 只用于诊断，不进入正式结果。

Fourier-LS 使用 full-resolution CUDA `torch.fft` 和 `float64/complex128`。当前加速通过缓存 invariant spectral terms、只重建 Powell 当前变化的 component，以及利用 Parseval 等价关系计算 quadratic profile 实现；没有缩图、减少 candidate 或削减 `2 sweeps × 25 maxiter` 的正式预算，因此论文只需把这些作为 Implementation Details，而不要写成新方法贡献。

## 7. 实验协议：E0–E4

论文和实验说明应区分：

- **E0**：smoke / preflight，只用于确认流程、数据、四方法接口和输出能跑通，不作为论文正式性能结论；
- **E1**：PINN selector / validation selection；
- 另外还有 Gaussian、Fourier-LS、U-Net 各自的 selector；
- 四类 selection artifacts 完成后进入 **Global Configuration Lock**；
- **E2**：冻结配置后的主要 synthetic benchmark；
- **E3**：PINN 消融实验；
- **E4**：real/native image evaluation。

E1 选择的是 PINN 的冻结超参数配置，**不是 checkpoint**。

正式 test 在 Global Lock 后不能再用于调参。

如果最终代码中的 E1/E2/E3/E4 或 selector 命令与旧文档有冲突，必须以最终 Stage 10 源码和最终 Stage 10 文档为准。

---

## 8. native-background audit 的论文口径

Stage 10 对 native background correction 增加质量审计，目的不是证明 reference 是绝对 ground truth，而是确认预处理没有明显伤害星点。

审计重点包括：

- clean sky median；
- star matching；
- centroid preservation；
- median flux relative error；
- p95 flux relative error；
- additive identity。

最终已冻结的重要光度门槛：

- median flux relative error：\(\le 1\%\)
- p95 flux relative error：\(\le 5\%\)

旧的 3% p95 口径已经被修正，不应再写进最终论文。

---

## 9. 当前实现状态与论文声称边界

### 可以写
- Stage 10-A（非交叉 / 当前正式主流程）已经实现；
- PINN 支持最多三个 forcing components；
- 当前 multi-source synthetic 实际采用两个源；
- U-Net 是 supervised regression baseline；
- Fourier-LS 是 physics-matched classical baseline；
- PINN 与 Fourier-LS 尽量共享物理 forcing family，以提高比较公平性；
- 主 benchmark 不额外加入未经标定的独立 synthetic noise。

### 不能在没有证据时写
- “已经验证三源场景”；
- “全面优于深度学习”；
- “reference 是绝对无噪声 ground truth”；
- “\(\alpha=0.5\) 是从真实大气测量反演得到的物理常数”；
- “Stage 10-B grouped/cross 已完成”，如果源码仍未实现；
- 把 E0 / pilot / E1 中间结果当最终论文性能结论；
- 编造 E2–E4 尚未跑出的 PSNR/SSIM/flux error 等数值。

---

## 10. 推荐论文初稿结构

建议先写成：

1. **Abstract**
2. **Introduction**
   - 城市 skyglow / 光污染导致星空背景非均匀；
   - 背景抑制与真实弱星信号保护存在冲突；
   - 纯滤波、监督网络和物理建模各有局限；
   - 本文提出 physics-informed screened-Poisson background estimation。
3. **Related Work**
   - astronomical background estimation / sky subtraction；
   - classical filtering / Fourier methods；
   - U-Net / data-driven image restoration；
   - PINNs / physics-informed inverse problems。
4. **Problem Formulation**
   - \(I_{\rm obs}=I_{\rm star}+B\)；
   - Stage 10 synthetic data semantics；
   - native-background correction 与 audit。
5. **Proposed Method**
   - screened-Poisson physical prior；
   - signed effective forcing；
   - multi-component forcing construction；
   - coordinate PINN；
   - loss function；
   - optimization constraints。
6. **Baselines**
   - Gaussian filtering；
   - Fourier-LS；
   - U-Net-small。
7. **Experimental Setup**
   - synthetic / real data；
   - no extra uncalibrated independent noise；
   - E0–E4；
   - selection + Global Lock；
   - metrics。
8. **Results**
   - 先放表格、图和数值占位符；
   - 正式实验出来后填入。
9. **Ablation and Discussion**
   - data-only；
   - single vs multi component；
   - fixed vs bounded center refinement；
   - PINN vs Fourier-LS：同物理模型不同求解器；
   - U-Net 的监督信息优势与泛化问题。
10. **Conclusion**

---

## 11. 给下一位助手的直接工作指令

请先读取用户重新上传的**最终 Stage 10 维护版 ZIP / Stage 10 总文档 / Method / Experimental Setup**，将它们作为事实源，然后写论文初稿。

要求：

1. 先不要凭一般知识改写项目设定；
2. 若源码与交接说明冲突，以最终 Stage 10 源码和最终 Stage 10 文档为准；
3. 未跑出的结果全部留占位符，不得虚构；
4. 论文行文可以比项目文档更学术、更紧凑，但不要夸大物理真实性或实验结论；
5. 所有公式符号在全文统一；
6. Related Work 如需正式文献，请联网检索可靠论文/官方来源，不能只靠记忆；
7. 初稿优先把 Method 和 Experimental Setup 写扎实，再补 Results；
8. 如果用户要求“直接出完整初稿”，就一次性给出完整结构和正文，不要反复追问已经冻结的方案。

---

## 12. 新对话建议上传的材料

优先级从高到低：

1. **最终 `stage10_document_maintained_final.zip`**
2. **`PINN_Starlight_Stage10_完整项目说明_合并中文版.md` 或对应 Word**
3. **Stage 10 Method / Experimental Setup**
4. 如果要核对具体公式和参数，再上传最终 Stage 10 源码包

不要只上传当前较早的 Stage 9 / `core-sorrime` 快照来写最终论文，否则很容易重新把早期已经纠正的问题写回来。

