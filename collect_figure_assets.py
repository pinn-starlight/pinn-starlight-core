#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
收集论文作图素材：
1) 图3：Stage-A 代表性空间结果
2) 图4：原始 PINN 消融
3) 图6：Stage-B 保护结构
4) 可选：图8/图9 真实图像定性图

用法示例：
pixi run python collect_figure_assets.py \
    --out ./figure_assets_bundle \
    --zip

如果你的路径不一致，直接改下面 CONFIG 即可。
"""

from __future__ import annotations
import argparse
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional


# ============================================================
# 你只需要优先改这里
# ============================================================
CONFIG = {
    # 总输出目录（命令行 --out 会覆盖）
    "default_out_dir": "./figure_assets_bundle",

    # -------------------------
    # Stage-A：图3 用
    # 两个代表样本：
    #   A: 低污染 / 单源
    #   B: 高污染 / 多源
    # -------------------------
    "stageA": {
        # 这里填你 Stage-A 总结果目录
        # 例如: "/data/xxx/outputs/stageA"
        "root": "./outputs/stageA",

        # 两个要拿来作图3的样本 ID
        # 改成你实际存在的名字
        "cases": {
            "A_low_single": "clean_single_002__single_low",
            "B_high_multi": "clean_stack_001__multi_high",
        },

        # 四个方法的目录名（如果你的目录名字不同就改这里）
        "methods": {
            "gaussian": "gaussian_filter",
            "dct": "dct_ls",
            "pinn": "original_pinn",
            "sgc": "sgc_pinn",
        },

        # 各类图像可能的文件名候选（按顺序查找）
        "filename_candidates": {
            "observed": [
                "observed.png", "I_obs.png", "input.png", "obs.png"
            ],
            "true_background": [
                "background_true.png", "B_true.png", "true_background.png", "gt_background.png"
            ],
            "true_residual": [
                "residual_true.png", "R_true.png", "true_residual.png", "gt_residual.png"
            ],
            "estimated_background": [
                "background.png", "estimated_background.png", "B_pred.png", "pred_background.png"
            ],
            "residual": [
                "residual.png", "corrected.png", "subtracted.png", "I_minus_B.png"
            ],
        },
    },

    # -------------------------
    # 原始 PINN 消融：图4 用
    # -------------------------
    "ablation": {
        "root": "./outputs/ablation_pinn",

        # 需要收集的变体名；可删可加
        "variants": [
            "data_only",
            "single_forcing",
            "multi_forcing_fixed_centers",
            "multi_forcing_bounded_centers",
        ],

        # 代表展示样本（可多个）
        # 可以只放 1 个，也可以放你论文最终图里要用的那几个
        "cases": [
            "clean_single_002__single_low",
            "clean_stack_001__multi_high",
        ],

        "filename_candidates": {
            "observed": [
                "observed.png", "I_obs.png", "input.png", "obs.png"
            ],
            "true_background": [
                "background_true.png", "B_true.png", "true_background.png", "gt_background.png"
            ],
            "true_residual": [
                "residual_true.png", "R_true.png", "true_residual.png", "gt_residual.png"
            ],
            "estimated_background": [
                "background.png", "estimated_background.png", "B_pred.png", "pred_background.png"
            ],
            "residual": [
                "residual.png", "corrected.png", "subtracted.png", "I_minus_B.png"
            ],
        },
    },

    # -------------------------
    # Stage-B：图6 用
    # -------------------------
    "stageB": {
        "root": "./outputs/stageB",

        # 代表样本：四类场景各给一个
        # 你可以按你的最终图替换这些名字
        "cases": {
            "edge_asym": "B001_edge_asym_high",
            "multiscale": "B002_multiscale_high",
            "irregular": "B003_irregular_high",
            "overlap": "B004_overlap_high",
        },

        "methods": {
            "gaussian": "gaussian_filter",
            "dct": "dct_ls",
            "pinn": "original_pinn",
            "sgc": "sgc_pinn",
        },

        "filename_candidates": {
            "observed": [
                "observed.png", "I_obs.png", "input.png", "obs.png"
            ],
            "protected_structure": [
                "protected_structure.png", "P.png", "structure.png", "target_structure.png"
            ],
            "estimated_background": [
                "background.png", "estimated_background.png", "B_pred.png", "pred_background.png"
            ],
            "residual": [
                "residual.png", "corrected.png", "subtracted.png", "I_minus_B.png"
            ],
            "projection_map": [
                "projection_map.png", "proj_map.png", "projection.png"
            ],
        },
    },

    # -------------------------
    # 真实图像：图8/9 用（可选）
    # -------------------------
    "real_eval": {
        "root": "./outputs/real_eval",
        "cases": {
            "real_high_001": "real_high_001",
            "real_high_016": "real_high_016",
        },
        "methods": {
            "gaussian": "gaussian_filter",
            "dct": "dct_ls",
            "pinn": "original_pinn",
            "sgc": "sgc_pinn",
        },
        "filename_candidates": {
            "observed": [
                "observed.png", "I_obs.png", "input.png", "obs.png"
            ],
            "estimated_background": [
                "background.png", "estimated_background.png", "B_pred.png", "pred_background.png"
            ],
            "residual": [
                "residual.png", "corrected.png", "subtracted.png", "I_minus_B.png"
            ],
        },
    },
}


# ============================================================
# 工具函数
# ============================================================
def mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_first_existing(base: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        p = base / name
        if p.exists():
            return p
    return None


def copy_if_exists(src: Optional[Path], dst: Path, verbose: bool = True):
    if src is None:
        if verbose:
            print(f"[MISS] {dst.name}")
        return
    mkdir(dst.parent)
    shutil.copy2(src, dst)
    if verbose:
        print(f"[OK]   {src}  ->  {dst}")


def zip_dir(src_dir: Path, zip_path: Path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(src_dir))
    print(f"[ZIP]  {zip_path}")


def collect_stageA(cfg: dict, out_root: Path):
    root = Path(cfg["root"])
    out_dir = out_root / "Fig3_StageA_representative"
    mkdir(out_dir)

    methods = cfg["methods"]
    fn = cfg["filename_candidates"]

    for label, case_id in cfg["cases"].items():
        case_out = out_dir / label
        mkdir(case_out)

        # 原始观测 / 真值
        case_root = root / case_id
        copy_if_exists(
            find_first_existing(case_root, fn["observed"]),
            case_out / "00_observed.png"
        )
        copy_if_exists(
            find_first_existing(case_root, fn["true_background"]),
            case_out / "01_true_background.png"
        )
        copy_if_exists(
            find_first_existing(case_root, fn["true_residual"]),
            case_out / "02_true_residual.png"
        )

        # 方法结果
        order = [
            ("gaussian", "03_gaussian"),
            ("dct",      "04_dct_ls"),
            ("pinn",     "05_original_pinn"),
            ("sgc",      "06_sgc_pinn"),
        ]
        for method_key, prefix in order:
            method_dir = case_root / methods[method_key]
            copy_if_exists(
                find_first_existing(method_dir, fn["estimated_background"]),
                case_out / f"{prefix}_background.png"
            )
            copy_if_exists(
                find_first_existing(method_dir, fn["residual"]),
                case_out / f"{prefix}_residual.png"
            )


def collect_ablation(cfg: dict, out_root: Path):
    root = Path(cfg["root"])
    out_dir = out_root / "Fig4_OriginalPINN_ablation"
    mkdir(out_dir)

    fn = cfg["filename_candidates"]

    for case_id in cfg["cases"]:
        case_out = out_dir / case_id
        mkdir(case_out)

        case_root = root / case_id

        # 公共输入/真值
        copy_if_exists(
            find_first_existing(case_root, fn["observed"]),
            case_out / "00_observed.png"
        )
        copy_if_exists(
            find_first_existing(case_root, fn["true_background"]),
            case_out / "01_true_background.png"
        )
        copy_if_exists(
            find_first_existing(case_root, fn["true_residual"]),
            case_out / "02_true_residual.png"
        )

        for i, variant in enumerate(cfg["variants"], start=1):
            var_root = case_root / variant
            tag = f"{i:02d}_{variant}"
            copy_if_exists(
                find_first_existing(var_root, fn["estimated_background"]),
                case_out / f"{tag}_background.png"
            )
            copy_if_exists(
                find_first_existing(var_root, fn["residual"]),
                case_out / f"{tag}_residual.png"
            )


def collect_stageB(cfg: dict, out_root: Path):
    root = Path(cfg["root"])
    out_dir = out_root / "Fig6_StageB_protected_structure"
    mkdir(out_dir)

    methods = cfg["methods"]
    fn = cfg["filename_candidates"]

    for label, case_id in cfg["cases"].items():
        case_out = out_dir / label
        mkdir(case_out)

        case_root = root / case_id

        copy_if_exists(
            find_first_existing(case_root, fn["observed"]),
            case_out / "00_observed.png"
        )
        copy_if_exists(
            find_first_existing(case_root, fn["protected_structure"]),
            case_out / "01_protected_structure.png"
        )

        order = [
            ("gaussian", "02_gaussian"),
            ("dct",      "03_dct_ls"),
            ("pinn",     "04_original_pinn"),
            ("sgc",      "05_sgc_pinn"),
        ]
        for method_key, prefix in order:
            method_dir = case_root / methods[method_key]
            copy_if_exists(
                find_first_existing(method_dir, fn["estimated_background"]),
                case_out / f"{prefix}_background.png"
            )
            copy_if_exists(
                find_first_existing(method_dir, fn["residual"]),
                case_out / f"{prefix}_residual.png"
            )
            copy_if_exists(
                find_first_existing(method_dir, fn["projection_map"]),
                case_out / f"{prefix}_projection_map.png"
            )


def collect_real_eval(cfg: dict, out_root: Path):
    root = Path(cfg["root"])
    out_dir = out_root / "Fig8_Fig9_real_eval"
    mkdir(out_dir)

    methods = cfg["methods"]
    fn = cfg["filename_candidates"]

    for label, case_id in cfg["cases"].items():
        case_out = out_dir / label
        mkdir(case_out)

        case_root = root / case_id
        copy_if_exists(
            find_first_existing(case_root, fn["observed"]),
            case_out / "00_observed.png"
        )

        order = [
            ("gaussian", "01_gaussian"),
            ("dct",      "02_dct_ls"),
            ("pinn",     "03_original_pinn"),
            ("sgc",      "04_sgc_pinn"),
        ]
        for method_key, prefix in order:
            method_dir = case_root / methods[method_key]
            copy_if_exists(
                find_first_existing(method_dir, fn["estimated_background"]),
                case_out / f"{prefix}_background.png"
            )
            copy_if_exists(
                find_first_existing(method_dir, fn["residual"]),
                case_out / f"{prefix}_residual.png"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=CONFIG["default_out_dir"], help="输出目录")
    parser.add_argument("--zip", action="store_true", help="收集后自动压缩")
    parser.add_argument("--skip-real", action="store_true", help="不收集真实图像图8/图9素材")
    args = parser.parse_args()

    out_root = Path(args.out).resolve()
    mkdir(out_root)

    print(f"[INFO] 输出目录: {out_root}")

    collect_stageA(CONFIG["stageA"], out_root)
    collect_ablation(CONFIG["ablation"], out_root)
    collect_stageB(CONFIG["stageB"], out_root)

    if not args.skip_real:
        collect_real_eval(CONFIG["real_eval"], out_root)

    if args.zip:
        zip_dir(out_root, out_root.with_suffix(".zip"))

    print("[DONE] 收集完成。")


if __name__ == "__main__":
    main()
