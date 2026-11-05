# scripts/ops

一次性运维脚本。它们只读取冻结预测，不训练模型、不修改 Global Lock。

| 脚本 | 作用 |
| --- | --- |
| `backfill_forced_reference.sh` | 对 Stage-A E2/E3、Stage-B benchmark、CV5 的冻结预测回填 forced-reference 指标 |
| `validate_pinn_preservation.sh` | 冻结 PINN 的保持性验证；强制 standard split |

脚本把仓库根目录解析为自身路径的 `../..`，因此可以从任意目录调用，例如：

```bash
bash scripts/ops/backfill_forced_reference.sh
bash scripts/ops/validate_pinn_preservation.sh
```
