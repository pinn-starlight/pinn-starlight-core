# scripts

一次性脚本与论文图流水线，和可复用的 `experiments/`、`tools/` 分开。

- `figures/`：论文图 3/4/6 的资产收集、渲染、打包；历史版本在 `figures/legacy/`。
- `ops/`：已结束的一次性运维脚本（forced-reference 回填、PINN 保持性验证）。

这些脚本都只读取冻结结果，不训练模型、不修改原始输出。
