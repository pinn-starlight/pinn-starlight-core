# Stage10-A 最新同步报告

## 结论

上传的 `stage10.rar` 中关键正式代码已经与当前最新冻结候选一致；本次同步不修改任何 Python 源码，只修正文档中的旧口径，因此不会改变正在运行的 selection provenance。

## 关键代码 CRC 对比

| file | matches latest |
| --- | --- |
| `experiments/configs/formal_experiment.py` | YES |
| `experiments/scripts/e2/run.py` | YES |
| `experiments/scripts/e3/run.py` | YES |
| `experiments/scripts/e4/run.py` | YES |
| `run_pipeline.py` | YES |
| `流程.md` | YES |
| `Stage10A_配置与加速冻结说明.md` | YES |
| `experiments/common/baselines/fft_screened_poisson.py` | YES |
| `experiments/scripts/selection/fourier_ls/run.py` | YES |
| `experiments/common/baselines/coordinate_pinn.py` | YES |
| `src/pinn_starlight_core/nn/pinn_loss.py` | YES |
| `src/pinn_starlight_core/nn/physics_model.py` | YES |
| `experiments/scripts/e1/run.py` | YES |
| `experiments/common/baselines/unet_small.py` | YES |

## 本次更新的文档

- `README.md`
- `docs/编写/III_Method_中文草稿.md`
- `docs/编写/IV_Experimental_Setup_中文草稿.md`
- `PINN_Starlight_论文初稿_交接说明.md`
- `PINN_Starlight_Stage10_完整项目说明_合并中文版.md`

主要修正：

1. U-Net 从旧 `LeakyReLU + base_channels=16 + full-first inference` 更新为当前 `ReLU + base_channels=64 + fixed tiled inference`。
2. 论文显示名统一为 `U-Net`，内部 `unet_small` 仅作为兼容 identifier。
3. PINN 正式 Laplacian 固定 `autograd`；analytic backend 仅作为 diagnostic。
4. Fourier-LS 补充 full-resolution CUDA float64/complex128 与 Parseval/caching 的等价实现说明。
5. E2/E3/E4 补充 single-visible-GPU formal execution policy。
6. 更新 `run_pipeline.py` 状态：当前已修复，不再保留旧 `selection.pinn.run` 警告。
7. 明确 Stage10-B grouped cross-validation 尚未实现。