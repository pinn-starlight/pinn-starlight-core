# scripts/figures

论文图 3/4/6 的离线 / 服务器流水线。所有脚本都只读取冻结实验数组，不重跑实验，也不修改原始结果。

| 脚本 | 用途 |
| --- | --- |
| `collect_assets.py` | 自动匹配 Stage-A/B 的 TIFF/PNG，生成 Fig3/4/6 资产包（推荐；支持 `--dry-run`、`--strict`） |
| `pack_figure_assets.py` | 按固定样本把三张图的资产打包成独立 ZIP（服务器端，路径写死为 `/workspace/stage11`） |
| `render_figures.py` | 参数化 `--root` 的渲染器，适合本地或任意目录布局 |
| `render_figures_server_exact.py` | 针对 `/workspace/stage11` 固定目录的精确渲染器（字体修复版，重渲染推荐） |
| `legacy/` | 历史版本（`collect_figure_assets.py`、`collect_stage11_figure_assets.py`、`render_figures_exact_v1.py`），不再维护 |

常用命令（在仓库根目录执行）：

```bash
# 先干跑，确认样本/方法目录都能匹配
pixi run -e remote python scripts/figures/collect_assets.py --root /workspace/stage11 --dry-run

# 收集并打 zip
pixi run -e remote python scripts/figures/collect_assets.py --root /workspace/stage11 --zip

# 渲染当前论文图（固定服务器布局）
pixi run -e remote python scripts/figures/render_figures_server_exact.py --out paper_figures --dpi 600
```

注意：`render_figures_server_exact.py` 需要 `fc-list` 中可见的中文字体（SimSun、Noto Serif CJK SC、Source Han Serif 等）；缺字体时脚本会直接报错，而不是输出中文方框。`render_figures.py` 则支持通过 `--root`、`--outdir`、`--fig3-low` 等参数适配其他布局。
