# Stage10-A 配置与加速冻结说明

## 结论

当前代码的科学方法与已经讨论并同意的 Stage10-A 口径一致。此次整理只把**执行策略显式冻结**，并修正运行流程；没有改变四种方法的科学超参数或 selection candidate 集合。

## 方法状态

### PINN

正式实现：

- coordinate PINN；
- fixed `alpha=0.5`；
- signed effective forcing；
- 最多 3 个 component；
- E1 / E2 / E3 / E4 均采用 nested-autograd Laplacian。

已保留的安全加速：

- 只在冻结的 history cadence 上执行 CUDA scalar `.item()`；
- forcing diagnostics 批量回传；
- E1 同一 validation image 的 observed-only forcing-center initialization 只计算一次。

未采用：

- analytic chain-rule Laplacian。2080 Ti A/B 结果数值接近，但训练速度为 reference 的约 0.687×，因此不进入正式 Stage10-A。

### Fourier-LS

正式实现保持：

- full-resolution Fourier screened-Poisson solver；
- validation-selected `lambda_ls` 与 sparse weight；
- bounded block-coordinate Powell；
- `max_sweeps=2`；
- `component_maxiter=25`；
- 非负 active-set weight profile。

实现级加速：

- CUDA `torch.fft`；
- float64 / complex128；
- 缓存 invariant spectral workspace；
- Powell 内只重建当前变化 component；
- 使用 Parseval 等价的 spectral Gram quadratic profile；
- 只在接受/final state 重建完整背景。

这属于数学等价实现优化，不需要在论文中作为新方法贡献。Implementation Details 中一句说明即可。

### U-Net

论文显示名：**U-Net**。  
内部 `unet_small` / `UNetSmall` 保留为历史兼容 identifier。

冻结架构：

- 单通道输入 / 单通道线性输出；
- 4 次下采样 / 4 次上采样；
- 双 3×3 Conv + ReLU；
- channels = 64, 128, 256, 512, 1024；
- MaxPool + ConvTranspose；
- skip concatenation；
- L1 background regression；
- Adam；
- AMP；
- 256×256 random patch training。

冻结 inference：

- tiled；
- tile size 256；
- overlap 64；
- NCHW；
- tile batch size 1。

U-Net backend v2 诊断后未采用 channels-last / fused Adam，因为稳态收益有限且训练轨迹产生明显数值分叉；batched tiled inference 也没有形成有意义的加速。

## Formal execution policy

Stage10-A 正式 E2/E3/E4：

- single visible GPU；
- logical device `cuda:0`；
- PINN Laplacian = `autograd`；
- Fourier-LS backend = `cuda`；
- Fourier-LS numeric dtype = float64/complex128；
- U-Net = NCHW + ordinary Adam + single-tile inference。

这些 execution settings 已进入中央 formal configuration snapshot，Global Lock 会一起冻结。

## 对正在进行的 selection 的影响

此次新增的 `FORMAL_EXECUTION_CONFIG` **没有加入各方法的 selection dependency config**，因此没有改变：

- PINN E1 candidate/config hash；
- Gaussian selector config；
- Fourier-LS selector candidate/config；
- U-Net training/selection config。

修改的 E2/E3/E4 只在 Global Lock 后运行。

因此：

- 如果你现在只是在跑 E1 / Gaussian / Fourier-LS / U-Net selection，已有 selection 不需要因为本次流程补丁重跑；
- 如果已经生成了 Global Configuration Lock，则应用本补丁后必须重新生成 Global Lock；
- 如果已经跑了 E2/E3/E4，则不应中途替换 formal source，应该另起一次完整 formal run。

## 论文口径

Fourier-LS 的 Parseval / caching 不应写成论文创新点。建议只在 Implementation Details 中说明：

> Fourier-LS implementation caches invariant spectral terms and evaluates the equivalent quadratic profile in the Fourier domain using Parseval's identity, without changing the screened-Poisson least-squares objective or the frozen Powell optimization protocol.

PINN analytic Laplacian 不进入正式结果，因此正文无需介绍。
