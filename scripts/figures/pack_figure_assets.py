#!/usr/bin/env python3
"""
在远程服务器上把论文图 3 / 图 4 / 图 6 所需冻结实验数组整理并压缩。

运行：
  cd /workspace/stage11
  pixi run -e remote python scripts/figures/pack_figure_assets.py

输出：
  /workspace/stage11/figure_transfer/
    Fig3_assets.zip
    Fig4_assets.zip
    Fig6_assets.zip
    Figures_3_4_6_all.zip

不会重跑实验，不会修改任何原始结果。
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path("/workspace/stage11")
OUT = ROOT / "figure_transfer"
SEED = 20260728

# 固定代表样本
FIG3_LOW = "clean_single_001_single_radial_low"
FIG3_HIGH = "clean_single_001_multi_source_high"
FIG4_SAMPLE = "clean_single_001_multi_source_high"
FIG6_SAMPLE = "clean_single_001_protected_overlap_high"

STAGEA_SOURCE = ROOT / "collections" / "synthetic_formal"
STAGEA_BASELINE = ROOT / "experiments" / "outputs" / "e2_benchmark"
STAGEA_ABLATION = ROOT / "experiments" / "outputs" / "e3_ablation"

STAGEB_SOURCE = ROOT / "collections" / "synthetic_stage_b"
STAGEB_BASELINE = ROOT / "experiments" / "outputs" / "stage_b" / "benchmark"
STAGEB_PAIR = ROOT / "experiments" / "outputs" / "stage_b" / "final_pinn_validation"

GAUSS_DIR = "fft_gaussian"
DCT_DIR = "fourier_ls"
PINN_DIR = "pinn"

PAIR_CONTROL = "data_finetune_control"
PAIR_SGC = "pinn_decoupled_gradient"

ABLATION_DIRS = {
    "data_only": "data_only",
    "single_component": "single_component",
    "multi_fixed_centers": "multi_fixed_centers",
    "multi_bounded_centers": "multi_bounded_centers",
}

SCI_EXTS = (".tif", ".tiff", ".npy")
PREVIEW_EXTS = (".png",)
ALL_EXTS = SCI_EXTS + PREVIEW_EXTS

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def source_file(sample_dir: Path, sid: str, role: str) -> Path:
    cands = []
    for ext in ALL_EXTS:
        cands += [
            sample_dir / f"{role}_{sid}{ext}",
            sample_dir / f"{role}{ext}",
        ]
    p = first_existing(cands)
    if p is None:
        raise FileNotFoundError(f"找不到 source 文件: {sid} / {role}\n目录: {sample_dir}")
    return p

def pred_file(run_dir: Path, role: str) -> Path:
    stems = {
        "background_pred": ("background_pred", "pred_background", "estimated_background", "background"),
        "residual_pred": ("residual_pred", "pred_residual", "background_subtracted", "corrected", "residual"),
    }[role]
    cands = []
    for stem in stems:
        for ext in ALL_EXTS:
            cands.append(run_dir / f"{stem}{ext}")
    p = first_existing(cands)
    if p is None:
        raise FileNotFoundError(f"找不到 {role}: {run_dir}")
    return p

def find_run_dir(root: Path, sid: str, require_seed=True) -> Path:
    """
    在给定方法/变体根目录下寻找包含 sample id 且真正有 background_pred/residual_pred 的目录。
    """
    candidates = []
    for cur, _dirs, files in os.walk(root):
        p = Path(cur)
        s = str(p)
        if sid not in s:
            continue
        if require_seed and f"seed_{SEED}" not in s:
            # 若目录结构本身没有 seed，后面也允许低分匹配
            seed_score = 0
        else:
            seed_score = 10
        fn = set(files)
        has_bg = any(x.startswith("background_pred") for x in fn)
        has_res = any(x.startswith("residual_pred") for x in fn)
        if has_bg or has_res:
            candidates.append((seed_score + int(has_bg) + int(has_res), len(s), p))
    if not candidates:
        raise FileNotFoundError(f"在 {root} 下找不到 sample={sid} 的输出目录")
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]

def stagea_method_dir(method: str, sid: str) -> Path:
    if method == "gaussian":
        p = STAGEA_BASELINE / GAUSS_DIR / "deterministic" / sid
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    if method == "dct":
        p = STAGEA_BASELINE / DCT_DIR / "deterministic" / sid
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    if method == "pinn":
        return find_run_dir(STAGEA_BASELINE / PINN_DIR, sid, require_seed=True)
    raise ValueError(method)

def ablation_run_dir(variant: str, sid: str) -> Path:
    vr = STAGEA_ABLATION / ABLATION_DIRS[variant]
    if not vr.is_dir():
        raise FileNotFoundError(vr)
    return find_run_dir(vr, sid, require_seed=True)

def stageb_method_dir(method: str, sid: str) -> Path:
    if method == "gaussian":
        p = STAGEB_BASELINE / GAUSS_DIR / "deterministic" / sid
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    if method == "dct":
        p = STAGEB_BASELINE / DCT_DIR / "deterministic" / sid
        if not p.is_dir():
            raise FileNotFoundError(p)
        return p
    if method == "pinn":
        return find_run_dir(STAGEB_BASELINE / PINN_DIR, sid, require_seed=True)
    raise ValueError(method)

def stageb_pair_dir(branch: str, sid: str) -> Path:
    p = STAGEB_PAIR / f"seed_{SEED}" / sid / branch
    if not p.is_dir():
        raise FileNotFoundError(p)
    return p

def copy_science_and_preview(src: Path, dst_dir: Path, dst_stem: str):
    dst_dir.mkdir(parents=True, exist_ok=True)
    # 复制选中的主文件
    shutil.copy2(src, dst_dir / f"{dst_stem}{src.suffix.lower()}")
    # 若主文件是 tif/tiff，同时复制同 stem 的 png 预览
    if src.suffix.lower() in SCI_EXTS:
        png = src.with_suffix(".png")
        if png.is_file():
            shutil.copy2(png, dst_dir / f"{dst_stem}.png")

def copy_source_roles(sample_root: Path, sid: str, roles, dst: Path):
    for role in roles:
        src = source_file(sample_root / sid, sid, role)
        copy_science_and_preview(src, dst, role)

def copy_method_pair(run_dir: Path, dst: Path):
    for role in ("background_pred", "residual_pred"):
        src = pred_file(run_dir, role)
        copy_science_and_preview(src, dst, role)

def write_readme(path: Path, txt: str):
    path.write_text(txt.strip() + "\n", encoding="utf-8")

def zip_folder(folder: Path, zip_path: Path):
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(folder.parent))

def reset_dir(p: Path):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)

def build_fig3():
    stage = OUT / "_stage_Fig3"
    reset_dir(stage)
    for label, sid in (("A_low_single", FIG3_LOW), ("B_high_multi", FIG3_HIGH)):
        d = stage / label
        copy_source_roles(
            STAGEA_SOURCE, sid,
            ("observed", "background_true", "clean_true"),
            d / "source"
        )
        copy_method_pair(stagea_method_dir("gaussian", sid), d / "gaussian")
        copy_method_pair(stagea_method_dir("dct", sid), d / "dct2_ls")
        copy_method_pair(stagea_method_dir("pinn", sid), d / "original_pinn")

    write_readme(stage / "README.txt", f"""
图3 Stage-A 代表性空间结果
A = {FIG3_LOW}
B = {FIG3_HIGH}
seed = {SEED}

正式出图使用 TIFF 科学数组。
每个场景：
- background_true + 三个 background_pred 共用一组显示范围；
- clean_true + 三个 residual_pred 共用另一组显示范围；
- observed 可独立显示范围。
""")
    z = OUT / "Fig3_assets.zip"
    zip_folder(stage, z)
    return z

def build_fig4():
    stage = OUT / "_stage_Fig4"
    reset_dir(stage)
    sid = FIG4_SAMPLE

    copy_source_roles(
        STAGEA_SOURCE, sid,
        ("observed", "background_true", "clean_true"),
        stage / "source"
    )
    for v in ABLATION_DIRS:
        rd = ablation_run_dir(v, sid)
        print(f"[Fig4] {v} -> {rd}")
        copy_method_pair(rd, stage / v)

    # 小表也一起带回
    for name in ("ablation_table.csv", "metrics.csv", "summary.json", "config.json"):
        p = STAGEA_ABLATION / name
        if p.is_file():
            shutil.copy2(p, stage / name)

    write_readme(stage / "README.txt", f"""
图4 原始 PINN 消融
sample = {sid}
seed = {SEED}

变体：
- data_only
- single_component
- multi_fixed_centers
- multi_bounded_centers

所有变体必须使用同一个样本、同一个显示范围。
""")
    z = OUT / "Fig4_assets.zip"
    zip_folder(stage, z)
    return z

def build_fig6():
    stage = OUT / "_stage_Fig6"
    reset_dir(stage)
    sid = FIG6_SAMPLE

    copy_source_roles(
        STAGEB_SOURCE, sid,
        ("base_clean", "protected_true", "clean_true", "background_true", "observed"),
        stage / "source"
    )
    copy_method_pair(stageb_method_dir("gaussian", sid), stage / "gaussian")
    copy_method_pair(stageb_method_dir("dct", sid), stage / "dct2_ls")
    copy_method_pair(stageb_method_dir("pinn", sid), stage / "original_pinn")
    copy_method_pair(stageb_pair_dir(PAIR_CONTROL, sid), stage / "matched_budget_control")
    copy_method_pair(stageb_pair_dir(PAIR_SGC, sid), stage / "sgc_pinn")

    write_readme(stage / "README.txt", f"""
图6 Stage-B 低频保护结构
sample = {sid}
seed = {SEED}

方法：
- gaussian
- dct2_ls
- original_pinn
- matched_budget_control
- sgc_pinn

保护区域定义：
protected_true >= 0.25 * peak

恢复背景共享一组显示范围；
所有背景扣除结果共享另一组显示范围。
""")
    z = OUT / "Fig6_assets.zip"
    zip_folder(stage, z)
    return z

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    zips = [build_fig3(), build_fig4(), build_fig6()]

    all_zip = OUT / "Figures_3_4_6_all.zip"
    if all_zip.exists():
        all_zip.unlink()
    with zipfile.ZipFile(all_zip, "w", zipfile.ZIP_STORED) as zf:
        for z in zips:
            zf.write(z, z.name)

    print("\n完成：")
    for z in zips + [all_zip]:
        print(f"{z}  ({z.stat().st_size / 1024**2:.1f} MB)")

if __name__ == "__main__":
    main()
