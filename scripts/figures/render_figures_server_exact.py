#!/usr/bin/env python3
"""
直接从 Stage11 冻结实验输出生成论文图 3 / 图 4 / 图 6。

已按当前服务器目录结构写死：
Stage-A source:
  /workspace/stage11/collections/synthetic_formal
Stage-A baseline:
  /workspace/stage11/experiments/outputs/e2_benchmark
Stage-A SGC paired:
  /workspace/stage11/experiments/outputs/final_pinn_validation
Stage-A ablation:
  /workspace/stage11/experiments/outputs/e3_ablation
Stage-B source:
  /workspace/stage11/collections/synthetic_stage_b
Stage-B baseline:
  /workspace/stage11/experiments/outputs/stage_b/benchmark
Stage-B SGC paired:
  /workspace/stage11/experiments/outputs/stage_b/final_pinn_validation

运行环境：
  pixi run -e remote python scripts/figures/render_figures_server_exact.py

不会重跑实验，不会修改原始文件。
"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec

try:
    import tifffile
except ImportError as e:
    raise SystemExit(
        "remote 环境缺少 tifffile；请先执行：pixi run -e remote python -c \"import tifffile\""
    ) from e


# ============================================================
# 固定工程路径
# ============================================================

ROOT = Path("/workspace/stage11")

STAGEA_SOURCE = ROOT / "collections" / "synthetic_formal"
STAGEA_BASELINE = ROOT / "experiments" / "outputs" / "e2_benchmark"
STAGEA_PAIR = ROOT / "experiments" / "outputs" / "final_pinn_validation"
STAGEA_ABLATION = ROOT / "experiments" / "outputs" / "e3_ablation"

STAGEB_SOURCE = ROOT / "collections" / "synthetic_stage_b"
STAGEB_BASELINE = ROOT / "experiments" / "outputs" / "stage_b" / "benchmark"
STAGEB_PAIR = ROOT / "experiments" / "outputs" / "stage_b" / "final_pinn_validation"

SEED = 20260728

# 固定代表样本，避免结果驱动选图
FIG3_LOW = "clean_single_001_single_radial_low"
FIG3_HIGH = "clean_single_001_multi_source_high"
FIG4_SAMPLE = "clean_single_001_multi_source_high"
FIG6_SAMPLE = "clean_single_001_protected_overlap_high"

# 你截图中确认的方法目录名
GAUSS_DIR = "fft_gaussian"
DCT_DIR = "fourier_ls"
PINN_DIR = "pinn"

# 你截图中确认的 paired branch 名
PAIR_CONTROL = "data_finetune_control"
PAIR_SGC = "pinn_decoupled_gradient"

# 你截图中确认的消融目录名
ABLATION_DIRS = {
    "data_only": "data_only",
    "single_component": "single_component",
    "multi_fixed_centers": "multi_fixed_centers",
    "multi_bounded_centers": "multi_bounded_centers",
}


# ============================================================
# 字体
# 重要：不再依赖 matplotlib 的字体名 fallback，避免中文变成方框。
# 直接通过 fontconfig 找到真实字体文件，并用 FontProperties(fname=...)。
#
# 中文优先：
#   SimSun/宋体 -> Noto Serif CJK SC -> Source Han Serif SC -> AR PL UMing CN
# 英文优先：
#   Times New Roman -> Times -> Liberation Serif -> Nimbus Roman
#
# 如果服务器没有任何中文字体，脚本会直接停止，并提示安装命令；
# 不会再默默生成“方框中文”的坏图。
# ============================================================

def _fc_list():
    try:
        out = subprocess.check_output(
            ["fc-list", "-f", "%{family}|%{file}\n"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        rows = []
        for line in out.splitlines():
            if "|" not in line:
                continue
            fam, file = line.split("|", 1)
            file = file.strip()
            if file and Path(file).is_file():
                rows.append((fam.strip(), file))
        return rows
    except Exception:
        return []

_FC_ROWS = _fc_list()

def find_real_font(candidates):
    # 必须真的在 fc-list 里出现，不能接受 fc-match 的隐式 fallback。
    for cand in candidates:
        cl = cand.lower()
        for fam, file in _FC_ROWS:
            if cl in fam.lower():
                return Path(file)
    return None

ZH_FONT_PATH = None
EN_FONT_PATH = None
ZH_FONT = None
EN_FONT = None


def resolve_fonts() -> None:
    """Resolve real font files lazily so that ``--help`` works on any machine."""
    global ZH_FONT_PATH, EN_FONT_PATH, ZH_FONT, EN_FONT

    ZH_FONT_PATH = find_real_font([
        "SimSun", "宋体",
        "Noto Serif CJK SC",
        "Source Han Serif SC",
        "Source Han Serif CN",
        "AR PL UMing CN",
        "WenQuanYi Zen Hei",  # 最后兜底：至少中文不变方框
    ])

    EN_FONT_PATH = find_real_font([
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "Nimbus Roman",
        "DejaVu Serif",
    ])

    if ZH_FONT_PATH is None:
        raise SystemExit(
            "\n[字体错误] 服务器没有可用中文字体，因此 matplotlib 会把中文画成方框。\n"
            "你当前是 root，最快修复：\n\n"
            "  apt-get update\n"
            "  apt-get install -y fonts-noto-cjk fonts-liberation2\n"
            "  fc-cache -fv\n"
            "  rm -rf ~/.cache/matplotlib\n\n"
            "然后重新运行：\n"
            "  pixi run -e remote python scripts/figures/render_figures_server_exact.py\n"
        )

    if EN_FONT_PATH is None:
        raise SystemExit(
            "\n[字体错误] 未找到英文衬线字体。建议：\n"
            "  apt-get update && apt-get install -y fonts-liberation2\n"
        )

    ZH_FONT = FontProperties(fname=str(ZH_FONT_PATH))
    EN_FONT = FontProperties(fname=str(EN_FONT_PATH))

# 数学字体使用 STIX，外观接近 Times。
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

NAVY = "#173f6b"

def has_cjk(s: str) -> bool:
    return any("\u3400" <= ch <= "\u9fff" for ch in s)

def fp_for_text(s: str):
    return ZH_FONT if has_cjk(s) else EN_FONT

def print_font_info():
    print("[FONT] 中文:", ZH_FONT.get_name(), "->", ZH_FONT_PATH)
    print("[FONT] 英文:", EN_FONT.get_name(), "->", EN_FONT_PATH)

# ============================================================
# 基础 I/O
# ============================================================

SCI_EXTS = (".tif", ".tiff", ".npy", ".png")


def die(msg: str):
    raise RuntimeError(msg)


def read_array(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        x = tifffile.imread(path)
    elif ext == ".npy":
        x = np.load(path)
    elif ext == ".png":
        x = plt.imread(path)
    else:
        die(f"不支持的数组格式: {path}")

    x = np.asarray(x)
    if x.ndim == 3:
        if x.shape[-1] >= 3:
            x = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
        else:
            x = x[..., 0]
    return x.astype(np.float32, copy=False)


def first_file(candidates: Iterable[Path], what: str) -> Path:
    for p in candidates:
        if p.is_file():
            return p
    die("找不到 " + what + "\n尝试路径：\n" + "\n".join(map(str, candidates)))


def source_file(sample_root: Path, sid: str, role: str) -> Path:
    cands = []
    for ext in SCI_EXTS:
        cands += [
            sample_root / f"{role}_{sid}{ext}",
            sample_root / f"{role}{ext}",
        ]
    return first_file(cands, f"{sid} / {role}")


def pred_file(run_dir: Path, role: str) -> Path:
    stems = {
        "background_pred": (
            "background_pred",
            "pred_background",
            "estimated_background",
            "background",
        ),
        "residual_pred": (
            "residual_pred",
            "pred_residual",
            "background_subtracted",
            "corrected",
            "residual",
        ),
    }[role]

    cands = []
    for stem in stems:
        for ext in SCI_EXTS:
            cands.append(run_dir / f"{stem}{ext}")
    return first_file(cands, f"{run_dir} / {role}")


def find_dir_exact_or_recursive(
    root: Path,
    required_parts: Sequence[str],
    prefer_parts: Sequence[str] = (),
) -> Path:
    """
    用于 e3_ablation，因为你只给了顶层结构；
    在确定 variant 根目录内递归寻找同时含 sample/seed 的目录。
    """
    required_parts = tuple(p.lower() for p in required_parts)
    prefer_parts = tuple(p.lower() for p in prefer_parts)

    matches = []
    for cur, _dirs, files in os.walk(root):
        p = Path(cur)
        s = str(p).lower()
        if all(part in s for part in required_parts):
            score = sum(part in s for part in prefer_parts)
            # 如果目录里直接有 background_pred/residual_pred，优先
            fs = {f.lower() for f in files}
            if any(f.startswith("background_pred") for f in fs):
                score += 10
            if any(f.startswith("residual_pred") for f in fs):
                score += 10
            matches.append((score, len(str(p)), p))

    if not matches:
        die(
            f"在 {root} 下找不到同时包含 {required_parts} 的目录"
        )

    matches.sort(key=lambda t: (-t[0], t[1]))
    return matches[0][2]


# ============================================================
# 精确实验目录
# ============================================================

def stagea_baseline_dir(method: str, sid: str) -> Path:
    if method == "gaussian":
        return STAGEA_BASELINE / GAUSS_DIR / "deterministic" / sid
    if method == "dct":
        return STAGEA_BASELINE / DCT_DIR / "deterministic" / sid
    if method == "pinn":
        # PINN 可能是 pinn/seed_x/sample 或 pinn/sample/seed_x，
        # 这里兼容二者。
        cands = [
            STAGEA_BASELINE / PINN_DIR / f"seed_{SEED}" / sid,
            STAGEA_BASELINE / PINN_DIR / sid / f"seed_{SEED}",
            STAGEA_BASELINE / PINN_DIR / sid,
        ]
        for p in cands:
            if p.is_dir():
                return p
        return find_dir_exact_or_recursive(
            STAGEA_BASELINE / PINN_DIR,
            required_parts=(sid,),
            prefer_parts=(f"seed_{SEED}",)
        )
    die(f"未知 Stage-A method: {method}")


def stagea_pair_dir(branch: str, sid: str) -> Path:
    # 截图已确认：
    # final_pinn_validation/seed_20260728/<sample>/{data_finetune_control,pinn_decoupled_gradient}
    p = STAGEA_PAIR / f"seed_{SEED}" / sid / branch
    if not p.is_dir():
        die(f"找不到 Stage-A paired 目录: {p}")
    return p


def ablation_dir(variant_key: str, sid: str) -> Path:
    variant_root = STAGEA_ABLATION / ABLATION_DIRS[variant_key]
    if not variant_root.is_dir():
        die(f"找不到消融目录: {variant_root}")

    # 常见结构优先
    cands = [
        variant_root / f"seed_{SEED}" / sid,
        variant_root / sid / f"seed_{SEED}",
        variant_root / sid,
    ]
    for p in cands:
        if p.is_dir() and (
            any((p / f"background_pred{ext}").is_file() for ext in SCI_EXTS)
            or any((p / f"residual_pred{ext}").is_file() for ext in SCI_EXTS)
        ):
            return p

    # 否则只在这个已确认 variant 下递归，不会串到别的消融
    return find_dir_exact_or_recursive(
        variant_root,
        required_parts=(sid,),
        prefer_parts=(f"seed_{SEED}",)
    )


def stageb_baseline_dir(method: str, sid: str) -> Path:
    if method == "gaussian":
        return STAGEB_BASELINE / GAUSS_DIR / "deterministic" / sid
    if method == "dct":
        return STAGEB_BASELINE / DCT_DIR / "deterministic" / sid
    if method == "pinn":
        cands = [
            STAGEB_BASELINE / PINN_DIR / f"seed_{SEED}" / sid,
            STAGEB_BASELINE / PINN_DIR / sid / f"seed_{SEED}",
            STAGEB_BASELINE / PINN_DIR / sid,
        ]
        for p in cands:
            if p.is_dir():
                return p
        return find_dir_exact_or_recursive(
            STAGEB_BASELINE / PINN_DIR,
            required_parts=(sid,),
            prefer_parts=(f"seed_{SEED}",)
        )
    die(f"未知 Stage-B method: {method}")


def stageb_pair_dir(branch: str, sid: str) -> Path:
    # 截图已确认结构和 branch 名
    p = STAGEB_PAIR / f"seed_{SEED}" / sid / branch
    if not p.is_dir():
        die(f"找不到 Stage-B paired 目录: {p}")
    return p


# ============================================================
# 显示范围
# ============================================================

def finite_sample(x: np.ndarray, max_n=350_000) -> np.ndarray:
    v = x[np.isfinite(x)].ravel()
    if v.size <= max_n:
        return v
    step = max(1, v.size // max_n)
    return v[::step]


def shared_range(arrays: Sequence[np.ndarray], lo=1.0, hi=99.5):
    v = np.concatenate([finite_sample(x) for x in arrays])
    a, b = np.percentile(v, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        a, b = np.nanmin(v), np.nanmax(v)
    if b <= a:
        b = a + 1e-6
    return float(a), float(b)


def single_range(x: np.ndarray, lo=1.0, hi=99.5):
    return shared_range([x], lo, hi)


# ============================================================
# 绘图辅助
# ============================================================

def style_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.55)
        sp.set_edgecolor("#777777")


def panel(ax, img, vrange, title=None, mask=None):
    ax.imshow(
        img, cmap="gray",
        vmin=vrange[0], vmax=vrange[1],
        interpolation="nearest"
    )
    if mask is not None and np.any(mask):
        ax.contour(
            mask.astype(float),
            levels=[0.5],
            colors="white",
            linewidths=0.55,
            linestyles="--"
        )
    style_ax(ax)
    if title:
        ax.set_title(
            title, fontsize=9.4, pad=4, color=NAVY,
            fontproperties=fp_for_text(title)
        )


def save(fig, outbase: Path, dpi: int):
    outbase.parent.mkdir(parents=True, exist_ok=True)
    png = outbase.with_suffix(".png")
    pdf = outbase.with_suffix(".pdf")
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[OK]", png)
    print("[OK]", pdf)


# ============================================================
# 图3：Stage-A
# ============================================================

def load_fig3_case(sid: str):
    src = STAGEA_SOURCE / sid
    if not src.is_dir():
        die(f"找不到 Stage-A source: {src}")

    d = {
        "obs": read_array(source_file(src, sid, "observed")),
        "bg_true": read_array(source_file(src, sid, "background_true")),
        "clean": read_array(source_file(src, sid, "clean_true")),
    }

    for tag, method in (
        ("g", "gaussian"),
        ("d", "dct"),
        ("p", "pinn"),
    ):
        rd = stagea_baseline_dir(method, sid)
        d[f"{tag}_bg"] = read_array(pred_file(rd, "background_pred"))
        d[f"{tag}_res"] = read_array(pred_file(rd, "residual_pred"))
    return d


def render_fig3(outdir: Path, dpi: int):
    groups = [
        ("A  低污染 / 单源", FIG3_LOW, load_fig3_case(FIG3_LOW)),
        ("B  高污染 / 多源", FIG3_HIGH, load_fig3_case(FIG3_HIGH)),
    ]

    fig = plt.figure(figsize=(12.4, 7.45), facecolor="white")
    fig.suptitle(
        "图3  Stage-A 代表性空间结果",
        fontsize=17, y=0.986, color=NAVY,
        fontproperties=ZH_FONT
    )
    fig.text(
        0.5, 0.952,
        "固定代表样本的背景估计与背景扣除结果",
        ha="center", fontsize=9.4, color="#555555",
        fontproperties=ZH_FONT
    )

    gs = GridSpec(
        4, 5, figure=fig,
        left=0.035, right=0.985, bottom=0.045, top=0.89,
        hspace=0.31, wspace=0.042,
        height_ratios=[1.0, 0.93, 1.0, 0.93]
    )

    headers = ["观测图像", "真实背景", "高斯滤波", "DCT-II LS", "原始 PINN"]

    for gi, (group_label, _sid, d) in enumerate(groups):
        r = gi * 2

        bg_v = shared_range([d["bg_true"], d["g_bg"], d["d_bg"], d["p_bg"]])
        res_v = shared_range([d["clean"], d["g_res"], d["d_res"], d["p_res"]])
        obs_v = single_range(d["obs"])

        firstrow = [d["obs"], d["bg_true"], d["g_bg"], d["d_bg"], d["p_bg"]]
        first_ranges = [obs_v, bg_v, bg_v, bg_v, bg_v]

        for c, (img, vr) in enumerate(zip(firstrow, first_ranges)):
            ax = fig.add_subplot(gs[r, c])
            panel(ax, img, vr)
            if gi == 0:
                ax.set_title(
                    headers[c], fontsize=9.9, pad=7, color=NAVY, weight="bold",
                    fontproperties=fp_for_text(headers[c])
                )

        ax = fig.add_subplot(gs[r+1, 0:2])
        panel(ax, d["clean"], res_v, "干净参考 / 真实残差")

        for c, (img, ttl) in enumerate([
            (d["g_res"], "高斯滤波扣除结果"),
            (d["d_res"], "DCT-II LS 扣除结果"),
            (d["p_res"], "原始 PINN 扣除结果"),
        ], start=2):
            ax = fig.add_subplot(gs[r+1, c])
            panel(ax, img, res_v, ttl)

        y = 0.905 if gi == 0 else 0.466
        fig.text(
            0.035, y, group_label, fontsize=10.4, color=NAVY, weight="bold",
            fontproperties=ZH_FONT
        )

    save(fig, outdir / "Fig3_StageA_spatial_results", dpi)


# ============================================================
# 图4：原始 PINN 消融
# ============================================================

def render_fig4(outdir: Path, dpi: int):
    sid = FIG4_SAMPLE
    src = STAGEA_SOURCE / sid
    obs = read_array(source_file(src, sid, "observed"))
    clean = read_array(source_file(src, sid, "clean_true"))

    variants = [
        ("data_only", "仅数据项"),
        ("single_component", "单一强迫分量"),
        ("multi_fixed_centers", "多分量固定中心"),
        ("multi_bounded_centers", "多分量有界中心"),
    ]

    vd = {}
    for key, _ in variants:
        rd = ablation_dir(key, sid)
        vd[key] = {
            "bg": read_array(pred_file(rd, "background_pred")),
            "res": read_array(pred_file(rd, "residual_pred")),
        }
        print(f"[Fig4] {key}: {rd}")

    bg_v = shared_range([vd[k]["bg"] for k, _ in variants])
    res_v = shared_range([clean] + [vd[k]["res"] for k, _ in variants])
    obs_v = single_range(obs)

    fig = plt.figure(figsize=(12.2, 5.25), facecolor="white")
    fig.suptitle(
        "图4  原始 PINN 消融实验代表性结果",
        fontsize=17, y=0.982, color=NAVY,
        fontproperties=ZH_FONT
    )
    fig.text(
        0.5, 0.938,
        f"固定样本：{sid}；固定随机种子：{SEED}",
        ha="center", fontsize=8.8, color="#555555",
        fontproperties=ZH_FONT
    )

    gs = GridSpec(
        2, 5, figure=fig,
        left=0.035, right=0.985, bottom=0.075, top=0.855,
        hspace=0.16, wspace=0.042
    )

    ax = fig.add_subplot(gs[0, 0])
    panel(ax, obs, obs_v, "观测图像")
    ax = fig.add_subplot(gs[1, 0])
    panel(ax, clean, res_v, "干净参考 / 真实残差")

    for c, (key, ttl) in enumerate(variants, start=1):
        ax = fig.add_subplot(gs[0, c])
        panel(ax, vd[key]["bg"], bg_v, ttl)
        ax = fig.add_subplot(gs[1, c])
        panel(ax, vd[key]["res"], res_v, "背景扣除结果")

    save(fig, outdir / "Fig4_PINN_ablation", dpi)


# ============================================================
# 图6：Stage-B
# ============================================================

def render_fig6(outdir: Path, dpi: int):
    sid = FIG6_SAMPLE
    src = STAGEB_SOURCE / sid
    if not src.is_dir():
        die(f"找不到 Stage-B source: {src}")

    clean = read_array(source_file(src, sid, "clean_true"))
    bg_true = read_array(source_file(src, sid, "background_true"))
    protected = read_array(source_file(src, sid, "protected_true"))
    obs = read_array(source_file(src, sid, "observed"))

    methods = []

    for label, method in [
        ("高斯滤波", "gaussian"),
        ("DCT-II LS", "dct"),
        ("原始 PINN", "pinn"),
    ]:
        rd = stageb_baseline_dir(method, sid)
        methods.append((
            label,
            read_array(pred_file(rd, "background_pred")),
            read_array(pred_file(rd, "residual_pred")),
        ))
        print(f"[Fig6] {label}: {rd}")

    for label, branch in [
        ("匹配预算对照", PAIR_CONTROL),
        ("SGC-PINN", PAIR_SGC),
    ]:
        rd = stageb_pair_dir(branch, sid)
        methods.append((
            label,
            read_array(pred_file(rd, "background_pred")),
            read_array(pred_file(rd, "residual_pred")),
        ))
        print(f"[Fig6] {label}: {rd}")

    bg_v = shared_range([bg_true] + [m[1] for m in methods])
    res_v = shared_range([clean] + [m[2] for m in methods])
    protected_v = single_range(protected, lo=0.0, hi=99.8)
    obs_v = single_range(obs)

    peak = float(np.nanmax(protected))
    mask = protected >= 0.25 * peak if peak > 0 else np.zeros_like(protected, dtype=bool)

    fig = plt.figure(figsize=(12.8, 7.15), facecolor="white")
    fig.suptitle(
        "图6  Stage-B 低频保护结构对比",
        fontsize=17, y=0.985, color=NAVY,
        fontproperties=ZH_FONT
    )
    fig.text(
        0.5, 0.949,
        f"固定频率重叠样本：{sid}",
        ha="center", fontsize=9.1, color="#555555",
        fontproperties=ZH_FONT
    )

    gs = GridSpec(
        3, 5, figure=fig,
        left=0.035, right=0.985, bottom=0.05, top=0.89,
        hspace=0.25, wspace=0.042,
        height_ratios=[0.82, 1.0, 1.0]
    )

    source_panels = [
        (clean, res_v, "干净目标"),
        (bg_true, bg_v, "人工污染背景"),
        (protected, protected_v, "受保护低频结构"),
        (obs, obs_v, "观测图像"),
    ]
    for c, (img, vr, ttl) in enumerate(source_panels):
        ax = fig.add_subplot(gs[0, c])
        panel(ax, img, vr, ttl)

    ax = fig.add_subplot(gs[0, 4])
    ax.axis("off")
    ax.text(
        0.5, 0.52,
        "保护区域阈值：\n"
        "P ≥ 0.25 × Pmax\n\n"
        "恢复背景共享显示范围\n"
        "扣除结果共享另一显示范围",
        ha="center", va="center", fontsize=9.6,
        fontproperties=ZH_FONT,
        bbox=dict(boxstyle="round,pad=0.55", fc="#f7f7f7", ec="#999999", lw=0.7)
    )

    for c, (ttl, bg, res) in enumerate(methods):
        ax = fig.add_subplot(gs[1, c])
        panel(ax, bg, bg_v, ttl)

        ax = fig.add_subplot(gs[2, c])
        panel(ax, res, res_v, "背景扣除结果", mask=mask)

    fig.text(
        0.035, 0.60, "恢复背景", fontsize=10.2, color=NAVY, weight="bold",
        fontproperties=ZH_FONT
    )
    fig.text(
        0.035, 0.307,
        "背景扣除结果（虚线：固定保护区域）",
        fontsize=10.2, color=NAVY, weight="bold",
        fontproperties=ZH_FONT
    )

    save(fig, outdir / "Fig6_StageB_protected_structure", dpi)


# ============================================================
# 检查路径
# ============================================================

def check():
    print("=== Font ===")
    print_font_info()

    print("\n=== Fig3 ===")
    for sid in (FIG3_LOW, FIG3_HIGH):
        print("\n", sid)
        print("source:", STAGEA_SOURCE / sid)
        for m in ("gaussian", "dct", "pinn"):
            print(m, ":", stagea_baseline_dir(m, sid))

    print("\n=== Fig4 ===")
    print("sample:", FIG4_SAMPLE)
    for k in ABLATION_DIRS:
        print(k, ":", ablation_dir(k, FIG4_SAMPLE))

    print("\n=== Fig6 ===")
    print("sample:", FIG6_SAMPLE)
    print("source:", STAGEB_SOURCE / FIG6_SAMPLE)
    for m in ("gaussian", "dct", "pinn"):
        print(m, ":", stageb_baseline_dir(m, FIG6_SAMPLE))
    print("control:", stageb_pair_dir(PAIR_CONTROL, FIG6_SAMPLE))
    print("sgc    :", stageb_pair_dir(PAIR_SGC, FIG6_SAMPLE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "paper_figures",
        help="输出目录"
    )
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument(
        "--only",
        choices=["fig3", "fig4", "fig6", "all"],
        default="all"
    )
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()

    out = args.out.expanduser().resolve()

    resolve_fonts()
    print_font_info()
    print("输出目录:", out)
    if args.check_only:
        check()
        return

    if args.only in ("fig3", "all"):
        render_fig3(out, args.dpi)
    if args.only in ("fig4", "all"):
        render_fig4(out, args.dpi)
    if args.only in ("fig6", "all"):
        render_fig6(out, args.dpi)

    print("\n全部完成。")


if __name__ == "__main__":
    main()
