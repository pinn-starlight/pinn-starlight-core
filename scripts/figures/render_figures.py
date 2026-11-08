#!/usr/bin/env python3
"""
直接在远程 Stage11 / pixi 环境生成论文图 3、图 4、图 6。
不打包原始数组，不重跑实验，不修改任何实验结果。

默认工程根目录：
    /workspace/stage11

默认输出：
    /workspace/stage11/paper_figures/
        Fig3_StageA_spatial_results.png/.pdf
        Fig4_PINN_ablation.png/.pdf
        Fig6_StageB_protected_structure.png/.pdf

推荐运行：
    cd /workspace/stage11
    pixi run python scripts/figures/render_figures.py --root . --dpi 600

可先只检查路径：
    pixi run python scripts/figures/render_figures.py --root . --check-only
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec

try:
    import tifffile
except Exception as e:
    raise SystemExit("缺少 tifffile。请在 pixi 环境确认依赖；错误：" + repr(e)) from e

# ---------------------------------------------------------------------
# 默认固定样本：避免看结果后再挑样本
# ---------------------------------------------------------------------

DEFAULT_FIG3_LOW = "clean_single_001_single_radial_low"
DEFAULT_FIG3_HIGH = "clean_single_001_multi_source_high"
DEFAULT_FIG4_SAMPLE = "clean_single_001_multi_source_high"
DEFAULT_FIG6_SAMPLE = "clean_single_001_protected_overlap_high"
DEFAULT_SEED = 20260728

# ---------------------------------------------------------------------
# 字体
# 中文优先宋体；英文优先 Times New Roman。
# 如果服务器缺字体，会打印实际 fallback。
# ---------------------------------------------------------------------

CN_CANDIDATES = [
    "SimSun",
    "宋体",
    "Songti SC",
    "STSong",
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "Source Han Serif CN",
]
EN_CANDIDATES = [
    "Times New Roman",
    "Times",
    "Nimbus Roman",
    "Liberation Serif",
]


def pick_font(candidates: Sequence[str]) -> font_manager.FontProperties:
    for name in candidates:
        try:
            path = font_manager.findfont(
                font_manager.FontProperties(family=name),
                fallback_to_default=False,
            )
            if path and Path(path).exists():
                return font_manager.FontProperties(fname=path)
        except Exception:
            pass
    # 最后 fallback；只用于保证脚本能跑
    path = font_manager.findfont("DejaVu Serif")
    return font_manager.FontProperties(fname=path)


CN_FONT = pick_font(CN_CANDIDATES)
EN_FONT = pick_font(EN_CANDIDATES)

# 数学字体用 STIX，外观接近 Times
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------
# 路径 / 数据读取
# ---------------------------------------------------------------------

SCI_EXTS = (".tif", ".tiff", ".npy", ".png")


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def first_existing_dir(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.is_dir():
            return p
    return None


def read_array(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        a = tifffile.imread(path)
    elif ext == ".npy":
        a = np.load(path)
    elif ext == ".png":
        a = plt.imread(path)
    else:
        raise ValueError(f"不支持的格式: {path}")

    a = np.asarray(a)
    if a.ndim == 3:
        # RGB/RGBA -> 线性灰度显示；科学数组通常不会走这里
        if a.shape[-1] >= 3:
            a = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
        else:
            a = a[..., 0]
    a = a.astype(np.float32, copy=False)
    return a


def resolve_source_file(sample_dir: Path, role: str, sid: str, required=True) -> Path | None:
    candidates = []
    for ext in SCI_EXTS:
        candidates.extend(
            [
                sample_dir / f"{role}_{sid}{ext}",
                sample_dir / f"{role}{ext}",
            ]
        )
    p = first_existing(candidates)
    if p is None and required:
        raise FileNotFoundError(
            f"找不到 {role}: sample={sid}\n目录: {sample_dir}\n"
            f"尝试了 {[x.name for x in candidates[:8]]} ..."
        )
    return p


def resolve_pred_file(method_dir: Path, role: str, required=True) -> Path | None:
    names = {
        "background_pred": [
            "background_pred",
            "pred_background",
            "estimated_background",
            "background",
        ],
        "residual_pred": [
            "residual_pred",
            "pred_residual",
            "background_subtracted",
            "corrected",
            "residual",
        ],
    }[role]
    candidates = []
    for stem in names:
        for ext in SCI_EXTS:
            candidates.append(method_dir / f"{stem}{ext}")
    p = first_existing(candidates)
    if p is None and required:
        raise FileNotFoundError(f"找不到 {role}\n目录: {method_dir}\n尝试文件名: {names}")
    return p


def locate_stageA_source(root: Path, sid: str) -> Path:
    cands = [
        root / "collections" / "synthetic_formal" / sid,
        root / "collections" / "stage_a" / sid,
        root / "collections" / "synthetic_stage_a" / sid,
    ]
    p = first_existing_dir(cands)
    if p is None:
        raise FileNotFoundError("找不到 Stage-A source sample:\n" + "\n".join(map(str, cands)))
    return p


def locate_stageB_source(root: Path, sid: str) -> Path:
    cands = [
        root / "collections" / "synthetic_stage_b" / sid,
        root / "collections" / "stage_b" / sid,
        root / "collections" / "synthetic_formal_stage_b" / sid,
    ]
    p = first_existing_dir(cands)
    if p is None:
        raise FileNotFoundError("找不到 Stage-B source sample:\n" + "\n".join(map(str, cands)))
    return p


def locate_e2_method(root: Path, method: str, sid: str, seed: int) -> Path:
    base = root / "experiments" / "outputs" / "e2_benchmark"
    method_candidates = {
        "gaussian": [
            base / "fft_gaussian" / "deterministic" / sid,
            base / "gaussian_filter" / "deterministic" / sid,
            base / "gaussian" / "deterministic" / sid,
        ],
        "dct": [
            base / "fourier_ls" / "deterministic" / sid,
            base / "dct2_ls" / "deterministic" / sid,
            base / "dct_ls" / "deterministic" / sid,
        ],
        "pinn": [
            base / "pinn" / f"seed_{seed}" / sid,
            base / "original_pinn" / f"seed_{seed}" / sid,
        ],
    }[method]
    p = first_existing_dir(method_candidates)
    if p is None:
        raise FileNotFoundError(
            f"找不到 Stage-A {method} 输出:\n" + "\n".join(map(str, method_candidates))
        )
    return p


def locate_ablation_variant(root: Path, variant: str, sid: str, seed: int) -> Path:
    base_candidates = [
        root / "experiments" / "outputs" / "e3_ablation",
        root / "experiments" / "outputs" / "ablation",
        root / "experiments" / "outputs" / "pinn_ablation",
    ]
    aliases = {
        "data_only": ["data_only", "data-only"],
        "single": ["single_component", "single_forcing", "single-forcing"],
        "fixed": ["multi_fixed_centers", "multi_forcing_fixed_centers", "multi-fixed-centers"],
        "bounded": [
            "multi_bounded_centers",
            "multi_forcing_bounded_centers",
            "multi-bounded-centers",
        ],
    }[variant]

    cands = []
    for base in base_candidates:
        for alias in aliases:
            cands.extend(
                [
                    base / alias / f"seed_{seed}" / sid,
                    base / alias / sid / f"seed_{seed}",
                    base / sid / alias / f"seed_{seed}",
                    base / sid / alias,
                ]
            )
    p = first_existing_dir(cands)
    if p is None:
        raise FileNotFoundError(
            f"找不到消融变体 {variant}, sample={sid}\n" + "\n".join(map(str, cands[:16]))
        )
    return p


def locate_stageB_benchmark(root: Path, method: str, sid: str, seed: int) -> Path:
    base_candidates = [
        root / "experiments" / "outputs" / "stage_b" / "benchmark",
        root / "experiments" / "outputs" / "stage_b_benchmark",
    ]
    cands = []
    for base in base_candidates:
        if method == "gaussian":
            cands += [
                base / "fft_gaussian" / "deterministic" / sid,
                base / "gaussian_filter" / "deterministic" / sid,
            ]
        elif method == "dct":
            cands += [
                base / "fourier_ls" / "deterministic" / sid,
                base / "dct2_ls" / "deterministic" / sid,
            ]
        elif method == "pinn":
            cands += [
                base / "pinn" / f"seed_{seed}" / sid,
                base / "original_pinn" / f"seed_{seed}" / sid,
            ]
    p = first_existing_dir(cands)
    if p is None:
        raise FileNotFoundError(
            f"找不到 Stage-B {method} benchmark:\n" + "\n".join(map(str, cands))
        )
    return p


def locate_stageB_pair(root: Path, branch: str, sid: str, seed: int) -> Path:
    bases = [
        root / "experiments" / "outputs" / "stage_b" / "final_pinn_validation",
        root / "experiments" / "outputs" / "stage_b" / "sgc_validation",
        root / "experiments" / "outputs" / "stage_b_final_pinn_validation",
    ]
    aliases = {
        "control": ["data_finetune_control", "matched_budget_control"],
        "sgc": ["pinn_decoupled_gradient", "sgc_pinn"],
    }[branch]
    cands = []
    for base in bases:
        for alias in aliases:
            cands += [
                base / f"seed_{seed}" / sid / alias,
                base / sid / f"seed_{seed}" / alias,
                base / alias / f"seed_{seed}" / sid,
            ]
    p = first_existing_dir(cands)
    if p is None:
        raise FileNotFoundError(
            f"找不到 Stage-B paired branch {branch}:\n" + "\n".join(map(str, cands))
        )
    return p


# ---------------------------------------------------------------------
# 显示范围
# ---------------------------------------------------------------------


def sampled_values(arr: np.ndarray, max_n=400_000) -> np.ndarray:
    x = arr[np.isfinite(arr)].ravel()
    if x.size <= max_n:
        return x
    step = max(1, x.size // max_n)
    return x[::step]


def pooled_range(arrays: Sequence[np.ndarray], lo=1.0, hi=99.5) -> tuple[float, float]:
    xs = [sampled_values(a) for a in arrays]
    x = np.concatenate(xs)
    vmin, vmax = np.percentile(x, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def one_range(a: np.ndarray, lo=1.0, hi=99.5) -> tuple[float, float]:
    return pooled_range([a], lo, hi)


# ---------------------------------------------------------------------
# 绘图辅助
# ---------------------------------------------------------------------


def imshow_panel(ax, img, vrange, title=None, title_en=False, contour=None):
    ax.imshow(img, cmap="gray", vmin=vrange[0], vmax=vrange[1], interpolation="nearest")
    if contour is not None:
        # 仅用于标明固定保护区域，不改变科学像素
        try:
            ax.contour(
                contour.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=0.55,
                linestyles="--",
            )
        except Exception:
            pass
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_edgecolor("0.45")
    if title:
        ax.set_title(
            title,
            fontproperties=EN_FONT if title_en else CN_FONT,
            fontsize=9.3,
            pad=4,
        )


def add_group_label(fig, x, y, text):
    fig.text(x, y, text, fontproperties=CN_FONT, fontsize=10.5, weight="bold")


def save_both(fig, outbase: Path, dpi: int):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(outbase.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {outbase.with_suffix('.png')}")
    print(f"[OK] {outbase.with_suffix('.pdf')}")


# ---------------------------------------------------------------------
# 图3
# ---------------------------------------------------------------------


def load_stageA_case(root: Path, sid: str, seed: int):
    src = locate_stageA_source(root, sid)
    obs = read_array(resolve_source_file(src, "observed", sid))
    bg_true = read_array(resolve_source_file(src, "background_true", sid))
    clean = read_array(resolve_source_file(src, "clean_true", sid))

    out = {"obs": obs, "bg_true": bg_true, "clean": clean}
    for key, method in [("gauss", "gaussian"), ("dct", "dct"), ("pinn", "pinn")]:
        d = locate_e2_method(root, method, sid, seed)
        out[f"{key}_bg"] = read_array(resolve_pred_file(d, "background_pred"))
        out[f"{key}_res"] = read_array(resolve_pred_file(d, "residual_pred"))
    return out


def render_fig3(root: Path, outdir: Path, low_sid: str, high_sid: str, seed: int, dpi: int):
    cases = [
        ("A  低污染 / 单源", load_stageA_case(root, low_sid, seed)),
        ("B  高污染 / 多源", load_stageA_case(root, high_sid, seed)),
    ]

    fig = plt.figure(figsize=(12.6, 7.55), facecolor="white")
    fig.suptitle(
        r"图3  $\mathrm{Stage\!-\!A}$ 代表性空间结果",
        fontproperties=CN_FONT,
        fontsize=17.5,
        y=0.985,
        color="#173f6b",
    )
    fig.text(
        0.5,
        0.948,
        "两组固定代表样本的背景估计与背景扣除结果",
        ha="center",
        fontproperties=CN_FONT,
        fontsize=9.8,
        color="0.25",
    )

    outer = GridSpec(
        4,
        5,
        figure=fig,
        left=0.035,
        right=0.985,
        bottom=0.055,
        top=0.89,
        hspace=0.32,
        wspace=0.045,
        height_ratios=[1.0, 0.95, 1.0, 0.95],
    )

    col_headers = [
        ("观测图像", False),
        ("真实背景", False),
        ("高斯滤波", False),
        ("DCT-II LS", True),
        ("原始 PINN", False),
    ]

    for gi, (group_name, d) in enumerate(cases):
        r0 = gi * 2
        bg_range = pooled_range([d["bg_true"], d["gauss_bg"], d["dct_bg"], d["pinn_bg"]])
        res_range = pooled_range([d["clean"], d["gauss_res"], d["dct_res"], d["pinn_res"]])
        obs_range = one_range(d["obs"])

        # 背景行
        imgs1 = [d["obs"], d["bg_true"], d["gauss_bg"], d["dct_bg"], d["pinn_bg"]]
        ranges1 = [obs_range, bg_range, bg_range, bg_range, bg_range]
        for c, (img, vr) in enumerate(zip(imgs1, ranges1)):
            ax = fig.add_subplot(outer[r0, c])
            imshow_panel(ax, img, vr)
            if gi == 0:
                title, is_en = col_headers[c]
                ax.set_title(
                    title,
                    fontproperties=EN_FONT if is_en else CN_FONT,
                    fontsize=10.0,
                    pad=7,
                    color="#173f6b",
                    weight="bold",
                )

        # 第二行：clean 跨前两列
        ax_clean = fig.add_subplot(outer[r0 + 1, 0:2])
        imshow_panel(ax_clean, d["clean"], res_range, title="干净参考 / 真实残差")
        for c, (key, title, is_en) in enumerate(
            [
                ("gauss_res", "高斯滤波扣除结果", False),
                ("dct_res", "DCT-II LS 扣除结果", True),
                ("pinn_res", "原始 PINN 扣除结果", False),
            ],
            start=2,
        ):
            ax = fig.add_subplot(outer[r0 + 1, c])
            imshow_panel(ax, d[key], res_range, title=title, title_en=is_en)

        # 组标签
        y = 0.905 if gi == 0 else 0.47
        add_group_label(fig, 0.035, y, group_name)

    save_both(fig, outdir / "Fig3_StageA_spatial_results", dpi)


# ---------------------------------------------------------------------
# 图4
# ---------------------------------------------------------------------


def render_fig4(root: Path, outdir: Path, sid: str, seed: int, dpi: int):
    src = locate_stageA_source(root, sid)
    obs = read_array(resolve_source_file(src, "observed", sid))
    clean = read_array(resolve_source_file(src, "clean_true", sid))

    variants = [
        ("data_only", "仅数据项"),
        ("single", "单一强迫分量"),
        ("fixed", "多分量固定中心"),
        ("bounded", "多分量有界中心"),
    ]
    data = {}
    for key, _ in variants:
        d = locate_ablation_variant(root, key, sid, seed)
        data[key] = {
            "bg": read_array(resolve_pred_file(d, "background_pred")),
            "res": read_array(resolve_pred_file(d, "residual_pred")),
        }

    bg_range = pooled_range([data[k]["bg"] for k, _ in variants])
    res_range = pooled_range([clean] + [data[k]["res"] for k, _ in variants])
    obs_range = one_range(obs)

    fig = plt.figure(figsize=(12.3, 5.45), facecolor="white")
    fig.suptitle(
        "图4  原始 PINN 消融实验代表性结果",
        fontproperties=CN_FONT,
        fontsize=17.2,
        y=0.98,
        color="#173f6b",
    )
    fig.text(
        0.5,
        0.935,
        f"固定样本：{sid}；固定随机种子：{seed}",
        ha="center",
        fontproperties=CN_FONT,
        fontsize=9.0,
        color="0.3",
    )

    gs = GridSpec(
        2,
        5,
        figure=fig,
        left=0.035,
        right=0.985,
        bottom=0.075,
        top=0.855,
        hspace=0.16,
        wspace=0.045,
    )

    # 第一列
    ax = fig.add_subplot(gs[0, 0])
    imshow_panel(ax, obs, obs_range, title="观测图像")
    ax = fig.add_subplot(gs[1, 0])
    imshow_panel(ax, clean, res_range, title="干净参考 / 真实残差")

    # 四变体
    for c, (key, title) in enumerate(variants, start=1):
        ax = fig.add_subplot(gs[0, c])
        imshow_panel(ax, data[key]["bg"], bg_range, title=title)
        ax = fig.add_subplot(gs[1, c])
        imshow_panel(ax, data[key]["res"], res_range, title="背景扣除结果")

    save_both(fig, outdir / "Fig4_PINN_ablation", dpi)


# ---------------------------------------------------------------------
# 图6
# ---------------------------------------------------------------------


def render_fig6(root: Path, outdir: Path, sid: str, seed: int, dpi: int):
    src = locate_stageB_source(root, sid)

    protected = read_array(resolve_source_file(src, "protected_true", sid))
    clean = read_array(resolve_source_file(src, "clean_true", sid))
    bg_true = read_array(resolve_source_file(src, "background_true", sid))
    obs = read_array(resolve_source_file(src, "observed", sid))

    methods = []

    for label, key in [
        ("高斯滤波", "gaussian"),
        ("DCT-II LS", "dct"),
        ("原始 PINN", "pinn"),
    ]:
        d = locate_stageB_benchmark(root, key, sid, seed)
        methods.append(
            (
                label,
                read_array(resolve_pred_file(d, "background_pred")),
                read_array(resolve_pred_file(d, "residual_pred")),
                key == "dct",
            )
        )

    for label, branch in [
        ("匹配预算对照", "control"),
        ("SGC-PINN", "sgc"),
    ]:
        d = locate_stageB_pair(root, branch, sid, seed)
        methods.append(
            (
                label,
                read_array(resolve_pred_file(d, "background_pred")),
                read_array(resolve_pred_file(d, "residual_pred")),
                label == "SGC-PINN",
            )
        )

    bg_range = pooled_range([bg_true] + [m[1] for m in methods])
    res_range = pooled_range([clean] + [m[2] for m in methods])
    protected_range = one_range(protected, 0.0, 99.8)
    obs_range = one_range(obs)

    peak = float(np.nanmax(protected))
    mask = protected >= (0.25 * peak) if peak > 0 else np.zeros_like(protected, dtype=bool)

    fig = plt.figure(figsize=(12.9, 7.35), facecolor="white")
    fig.suptitle(
        r"图6  $\mathrm{Stage\!-\!B}$ 低频保护结构对比",
        fontproperties=CN_FONT,
        fontsize=17.2,
        y=0.985,
        color="#173f6b",
    )
    fig.text(
        0.5,
        0.948,
        f"固定频率重叠样本：{sid}",
        ha="center",
        fontproperties=CN_FONT,
        fontsize=9.2,
        color="0.3",
    )

    outer = GridSpec(
        3,
        5,
        figure=fig,
        left=0.035,
        right=0.985,
        bottom=0.055,
        top=0.89,
        hspace=0.26,
        wspace=0.045,
        height_ratios=[0.82, 1.0, 1.0],
    )

    # 顶部源数据条：4格 + 1格说明
    source_panels = [
        (clean, res_range, "干净目标"),
        (bg_true, bg_range, "人工污染背景"),
        (protected, protected_range, "受保护低频结构"),
        (obs, obs_range, "观测图像"),
    ]
    for c, (img, vr, title) in enumerate(source_panels):
        ax = fig.add_subplot(outer[0, c])
        imshow_panel(ax, img, vr, title=title)

    ax_note = fig.add_subplot(outer[0, 4])
    ax_note.axis("off")
    ax_note.text(
        0.5,
        0.58,
        "固定保护区域\n"
        r"$P \geq 0.25\,P_{\mathrm{max}}$"
        "\n\n背景与残差分别\n采用统一显示范围",
        ha="center",
        va="center",
        fontproperties=CN_FONT,
        fontsize=10.2,
        bbox=dict(boxstyle="round,pad=0.55", fc="0.97", ec="0.65", lw=0.8),
    )

    # 两行方法结果
    for c, (label, bg, res, is_en) in enumerate(methods):
        ax = fig.add_subplot(outer[1, c])
        imshow_panel(ax, bg, bg_range, title=label, title_en=is_en)
        ax = fig.add_subplot(outer[2, c])
        imshow_panel(ax, res, res_range, title="背景扣除结果", contour=mask)

    add_group_label(fig, 0.035, 0.607, "恢复背景")
    add_group_label(fig, 0.035, 0.315, "背景扣除结果（虚线：固定保护区域）")

    save_both(fig, outdir / "Fig6_StageB_protected_structure", dpi)


# ---------------------------------------------------------------------
# 检查
# ---------------------------------------------------------------------


def check_paths(root: Path, args):
    print("中文字体:", CN_FONT.get_name(), CN_FONT.get_file())
    print("英文字体:", EN_FONT.get_name(), EN_FONT.get_file())
    print("\n[Fig3]")
    for sid in (args.fig3_low, args.fig3_high):
        print("sample:", sid)
        print(" source:", locate_stageA_source(root, sid))
        for m in ("gaussian", "dct", "pinn"):
            print(f" {m}:", locate_e2_method(root, m, sid, args.seed))

    print("\n[Fig4]")
    print("sample:", args.fig4_sample)
    print(" source:", locate_stageA_source(root, args.fig4_sample))
    for v in ("data_only", "single", "fixed", "bounded"):
        print(f" {v}:", locate_ablation_variant(root, v, args.fig4_sample, args.seed))

    print("\n[Fig6]")
    print("sample:", args.fig6_sample)
    print(" source:", locate_stageB_source(root, args.fig6_sample))
    for m in ("gaussian", "dct", "pinn"):
        print(f" {m}:", locate_stageB_benchmark(root, m, args.fig6_sample, args.seed))
    for b in ("control", "sgc"):
        print(f" {b}:", locate_stageB_pair(root, b, args.fig6_sample, args.seed))


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="直接从冻结 Stage11 实验数组渲染论文图3/图4/图6",
    )
    ap.add_argument("--root", type=Path, default=Path("/workspace/stage11"))
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)

    ap.add_argument("--fig3-low", default=DEFAULT_FIG3_LOW)
    ap.add_argument("--fig3-high", default=DEFAULT_FIG3_HIGH)
    ap.add_argument("--fig4-sample", default=DEFAULT_FIG4_SAMPLE)
    ap.add_argument("--fig6-sample", default=DEFAULT_FIG6_SAMPLE)

    ap.add_argument("--only", choices=["fig3", "fig4", "fig6", "all"], default="all")
    ap.add_argument("--check-only", action="store_true")

    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"工程根目录不存在: {root}")

    outdir = args.outdir.expanduser().resolve() if args.outdir else root / "paper_figures"

    print("root   =", root)
    print("outdir =", outdir)
    print("seed   =", args.seed)
    print("CN font=", CN_FONT.get_name())
    print("EN font=", EN_FONT.get_name())

    if args.check_only:
        check_paths(root, args)
        return

    if args.only in ("fig3", "all"):
        render_fig3(root, outdir, args.fig3_low, args.fig3_high, args.seed, args.dpi)
    if args.only in ("fig4", "all"):
        render_fig4(root, outdir, args.fig4_sample, args.seed, args.dpi)
    if args.only in ("fig6", "all"):
        render_fig6(root, outdir, args.fig6_sample, args.seed, args.dpi)

    print("\n完成。输出目录：", outdir)


if __name__ == "__main__":
    main()
