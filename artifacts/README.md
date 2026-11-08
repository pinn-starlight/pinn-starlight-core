# artifacts

本地大文件归档，整个目录（除本 README）已被 `.gitignore` 忽略；确认已有备份后可整体删除。

| 路径 | 内容 |
| --- | --- |
| `transfers/stage11_transfer_archives/` | 服务器增量搬运 ZIP，约 9.6 GB，含 Stage-A CV5 等结果 |
| `transfers/transfer_forced_reference_20260912_225540/` | forced-reference 回填批次的归档分卷与索引 |
| `review/e4_review_for_paper.zip` | E4 审阅用图（约 49 MB） |
| `logs/` | 已结束的 backfill 进程日志、PID 与退出码 |
| `patches/PATCH_MANIFEST.json` | 历史补丁清单（其中记录的仓库根路径已随本次整理变动） |

这些文件原先有一部分在 git 跟踪中；本次整理把它们移入本目录并从版本控制移除（磁盘文件仍在）。提交前请确认你不再需要它们在仓库内。

恢复或校验这些归档时，请使用 `tools/restore_server_archives.py` 与 `tools/verify_rebuild_ready.py`，参见 `tools/README.md`。
