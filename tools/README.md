# tools

Stage2 的冒烟测试、服务器结果搬运与离线重建工具，全部从仓库根目录运行。

| 脚本 | 作用 |
| --- | --- |
| `smoke_test.py` | CPU 冒烟测试：17 项端到端检查（导入、Laplacian、数据、四种方法、指标、生成器、机制探针、CLI），约 25 秒 |
| `fetch_smoke_data.sh` | 从服务器归档解出冒烟测试需要的最小数据集（约 280 MB） |
| `export_server_archives.py` | 在服务器上增量导出结果 ZIP（默认 `stage11_transfer_archives/`，默认 4 GiB 分卷） |
| `restore_server_archives.py` | 将 ZIP 恢复到本地项目树；支持 `--verify-only`、`--phases`、`--no-overwrite` |
| `rebuild_derived_outputs.py` | 由冻结的 `background_pred.tif` 重建 observed/residual TIFF 与 PNG；不重训 |
| `verify_rebuild_ready.py` | 校验恢复后的 manifest 与 TIFF 完整性；`--require-derived` 可检查派生文件 |
| `backfill_forced_reference_metrics.py` | 仅评估：对冻结预测回填 forced-reference 指标 |
| `run_diagnostics.sh` | 一键跑 `alpha_sweep` / `init_memory` 机制探针：预检解释器 / CUDA / validation 数据，双卡并行或单卡串行，日志落 `build/diag_logs/` |

示例：

```bash
# 冒烟测试（CPU，无需 GPU；缺数据时先跑 fetch_smoke_data.sh）
bash tools/fetch_smoke_data.sh
python -m tools.smoke_test

# 机制探针：先 smoke 验管道，再 full（双卡并行，自动分卡）
bash tools/run_diagnostics.sh --suite smoke
bash tools/run_diagnostics.sh --gpus 0,1

# 恢复
pixi run -e remote python tools/restore_server_archives.py \
    archived/transfers/stage11_transfer_archives .

# 重建派生文件并校验
pixi run -e remote python tools/rebuild_derived_outputs.py
pixi run -e remote python tools/verify_rebuild_ready.py --require-derived

# 只读指标回填
pixi run -e remote python tools/backfill_forced_reference_metrics.py
```

单图命令行在 `experiments/cli.py`（`python -m experiments.cli --help`），不依赖服务器归档；这里只保留数据搬运与自检工具。

注意：`export_server_archives.py` 的默认归档目录仍是仓库根目录下的 `stage11_transfer_archives/`；本地现有的归档已移动到 `../archived`。
