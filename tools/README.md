# tools

Stage11 服务器结果搬运与离线重建工具，全部从仓库根目录运行。

| 脚本 | 作用 |
| --- | --- |
| `export_server_archives.py` | 在服务器上增量导出结果 ZIP（默认 `stage11_transfer_archives/`，默认 4 GiB 分卷） |
| `restore_server_archives.py` | 将 ZIP 恢复到本地项目树；支持 `--verify-only`、`--phases`、`--no-overwrite` |
| `rebuild_derived_outputs.py` | 由冻结的 `background_pred.tif` 重建 observed/residual TIFF 与 PNG；不重训 |
| `verify_rebuild_ready.py` | 校验恢复后的 manifest 与 TIFF 完整性；`--require-derived` 可检查派生文件 |
| `backfill_forced_reference_metrics.py` | 仅评估：对冻结预测回填 forced-reference 指标 |

示例：

```bash
# 恢复
pixi run -e remote python tools/restore_server_archives.py \
    artifacts/transfers/stage11_transfer_archives .

# 重建派生文件并校验
pixi run -e remote python tools/rebuild_derived_outputs.py
pixi run -e remote python tools/verify_rebuild_ready.py --require-derived

# 只读指标回填
pixi run -e remote python tools/backfill_forced_reference_metrics.py
```

注意：`export_server_archives.py` 的默认归档目录仍是仓库根目录下的 `stage11_transfer_archives/`；本地现有的归档已移动到 `artifacts/transfers/`。
