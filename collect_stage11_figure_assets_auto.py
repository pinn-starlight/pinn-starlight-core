#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-match and collect frozen Stage11 scientific arrays for manuscript figures.

Targets
-------
Fig3  Stage-A representative spatial results
Fig4  Original PINN ablation
Fig6  Stage-B protected low-frequency structure

Design goals
------------
- NO experiment rerun.
- NO modification of original outputs.
- Prefer scientific TIFF/TIF arrays; copy PNG previews when present.
- Auto-detect sample IDs, method folders, ablation variants, and common output names.
- Produce a readable match report before/after packaging.
- Fail loudly in --strict mode if critical assets are missing.

Typical remote usage
--------------------
cd /workspace/stage11
pixi run python collect_stage11_figure_assets_auto.py --root . --zip

Recommended first pass:
pixi run python collect_stage11_figure_assets_auto.py --root . --dry-run

Optional overrides:
pixi run python collect_stage11_figure_assets_auto.py --root . \
    --fig3-low clean_single_001_single_radial_low \
    --fig3-high clean_single_001_multi_source_high \
    --fig6-high clean_single_001_protected_overlap_high \
    --zip
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

# ---------------------------------------------------------------------
# Constants / aliases
# ---------------------------------------------------------------------

PRUNE_DIRS = {
    ".git", ".pixi", ".cache", "__pycache__", "node_modules",
    "wandb", "tensorboard", "tb_logs", "figure_assets_export",
    "figure_assets_auto", "dist", "build",
}

SCI_EXTS = {".tif", ".tiff", ".npy", ".npz"}
PREVIEW_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TABLE_EXTS = {".csv", ".json", ".txt", ".md"}

METHOD_ALIASES = {
    "gaussian": (
        "fft_gaussian", "gaussian_filter", "gaussian", "gauss",
    ),
    "dct2_ls": (
        "fourier_ls", "dct2_ls", "dct_ii_ls", "dct-ii_ls", "dct_ls", "dct",
    ),
    "original_pinn": (
        "original_pinn", "raw_pinn", "baseline_pinn", "pinn",
    ),
    "matched_budget_control": (
        "data_finetune_control", "matched_budget_control",
        "matched_budget", "budget_control", "control",
    ),
    "sgc_pinn": (
        "pinn_decoupled_gradient", "sgc_pinn", "sgc-pinn",
        "staged_gradient", "gradient_coord",
    ),
}

ABLATION_ALIASES = {
    "data_only": (
        "data_only", "data-only", "dataonly",
    ),
    "single_forcing": (
        "single_forcing", "single-forcing", "single_component",
        "single-component", "one_forcing", "one_component",
    ),
    "multi_fixed_centers": (
        "multi_fixed_centers", "multi-fixed-centers",
        "multi_forcing_fixed_centers", "fixed_centers",
        "multi_fixed", "fixed_center",
    ),
    "multi_bounded_centers": (
        "multi_bounded_centers", "multi-bounded-centers",
        "multi_forcing_bounded_centers", "bounded_centers",
        "bounded_center", "multi_bounded", "adjustable_centers",
    ),
}

ROLE_ALIASES = {
    "observed": (
        "observed", "i_obs", "input", "obs",
    ),
    "background_true": (
        "background_true", "true_background", "gt_background", "b_true",
    ),
    "clean_true": (
        "clean_true", "reference_clean", "clean_reference",
        "true_residual", "residual_true", "target_clean",
    ),
    "base_clean": (
        "base_clean", "base_reference", "base_target",
    ),
    "protected_true": (
        "protected_true", "protected_structure", "target_structure",
        "protected", "structure_true",
    ),
    "background_pred": (
        "background_pred", "pred_background", "estimated_background",
        "background_est", "b_pred",
    ),
    "residual_pred": (
        "residual_pred", "pred_residual", "background_subtracted",
        "subtracted", "corrected", "residual",
    ),
    "mask": (
        "protected_mask", "mask", "structure_mask",
    ),
}

SEED_PREFERENCE = ("20260728", "20260729", "20260730")


# ---------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------

@dataclass
class Match:
    role: str
    path: Optional[Path]
    score: float = 0.0
    note: str = ""


class Reporter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.missing: list[str] = []

    def log(self, msg: str = "") -> None:
        print(msg)
        self.lines.append(msg)

    def miss(self, msg: str) -> None:
        print("[MISSING]", msg, file=sys.stderr)
        self.lines.append("[MISSING] " + msg)
        self.missing.append(msg)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def norm(s: str) -> str:
    s = s.lower().replace("\\", "/")
    s = re.sub(r"[\s\-]+", "_", s)
    return s


def contains_alias(text: str, aliases: Sequence[str]) -> bool:
    t = norm(text)
    return any(norm(a) in t for a in aliases)


def token_score(text: str, positive: Sequence[str], negative: Sequence[str] = ()) -> float:
    t = norm(text)
    score = 0.0
    for x in positive:
        if norm(x) in t:
            score += 3.0
    for x in negative:
        if norm(x) in t:
            score -= 4.0
    return score


def iter_files(root: Path, allowed_exts: Optional[set[str]] = None) -> Iterable[Path]:
    root = root.resolve()
    for cur, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in PRUNE_DIRS and not d.startswith(".")]
        curp = Path(cur)
        for fn in files:
            p = curp / fn
            if allowed_exts is None or p.suffix.lower() in allowed_exts:
                yield p


def sample_id_candidates_from_manifests(root: Path) -> set[str]:
    out: set[str] = set()
    for p in iter_files(root, {".csv"}):
        if p.name.lower() != "manifest.csv":
            continue
        try:
            with p.open("r", encoding="utf-8-sig", newline="") as f:
                r = csv.DictReader(f)
                if not r.fieldnames:
                    continue
                fields = {norm(x): x for x in r.fieldnames}
                sid_field = None
                for k in ("sample_id", "sample", "id", "case_id", "scene_id"):
                    if k in fields:
                        sid_field = fields[k]
                        break
                if not sid_field:
                    continue
                for row in r:
                    sid = (row.get(sid_field) or "").strip()
                    if sid:
                        out.add(sid)
        except Exception:
            pass
    return out


def sample_id_candidates_from_paths(root: Path) -> set[str]:
    out: set[str] = set()
    # Prefer scientific arrays and common source names.
    for p in iter_files(root, SCI_EXTS | PREVIEW_EXTS):
        name = p.stem
        low = norm(name)
        for prefix in (
            "observed_", "background_true_", "clean_true_",
            "base_clean_", "protected_true_",
        ):
            if low.startswith(prefix):
                out.add(name[len(prefix):])
        # Parent directory may itself be a sample folder.
        par = p.parent.name
        if any(k in norm(par) for k in (
            "single", "multi", "radial", "eccentric",
            "protected", "overlap", "irregular", "edge",
        )):
            out.add(par)
    return out


def find_sample_ids(root: Path) -> list[str]:
    ids = sample_id_candidates_from_manifests(root)
    ids |= sample_id_candidates_from_paths(root)
    return sorted(ids)


def choose_sample(
    ids: Sequence[str],
    positive: Sequence[str],
    negative: Sequence[str] = (),
    prefer_prefixes: Sequence[str] = ("clean_single_001", "clean_single_002", "clean_stack_001"),
) -> Optional[str]:
    if not ids:
        return None
    ranked = []
    for sid in ids:
        s = token_score(sid, positive, negative)
        for i, pref in enumerate(prefer_prefixes):
            if norm(pref) in norm(sid):
                s += max(0.0, 1.5 - 0.3 * i)
        # Prefer concise IDs; usually less likely to be derivative artifact folders.
        s -= min(len(sid), 200) * 0.0005
        ranked.append((s, sid))
    ranked.sort(reverse=True)
    return ranked[0][1] if ranked and ranked[0][0] > 0 else None


def role_score(path: Path, role: str) -> float:
    aliases = ROLE_ALIASES[role]
    nm = norm(path.stem)
    full = norm(str(path))
    score = 0.0
    for a in aliases:
        aa = norm(a)
        if nm == aa:
            score += 20
        elif nm.startswith(aa + "_") or nm.endswith("_" + aa):
            score += 14
        elif aa in nm:
            score += 9
        elif aa in full:
            score += 4

    # Prefer scientific arrays over preview images.
    ext = path.suffix.lower()
    if ext in (".tif", ".tiff"):
        score += 8
    elif ext in (".npy", ".npz"):
        score += 5
    elif ext == ".png":
        score += 2

    # Penalize obvious masks when looking for residual/background, etc.
    if role not in ("mask", "protected_true") and "mask" in nm:
        score -= 8
    return score


def method_score(path: Path, method: str) -> float:
    full = norm(str(path))
    score = 0.0
    for a in METHOD_ALIASES[method]:
        aa = norm(a)
        if f"/{aa}/" in full or full.endswith("/" + aa):
            score += 12
        elif aa in full:
            score += 6

    # Avoid accidentally selecting ablation outputs as original PINN.
    if method == "original_pinn":
        for aliases in ABLATION_ALIASES.values():
            if contains_alias(full, aliases):
                score -= 15

    # Seed preference
    for i, seed in enumerate(SEED_PREFERENCE):
        if seed in full:
            score += 5 - i
    return score


def variant_score(path: Path, variant: str) -> float:
    full = norm(str(path))
    score = 0.0
    for a in ABLATION_ALIASES[variant]:
        aa = norm(a)
        if f"/{aa}/" in full:
            score += 15
        elif aa in full:
            score += 8
    for i, seed in enumerate(SEED_PREFERENCE):
        if seed in full:
            score += 4 - i
    return score


def select_best_file(
    files: Sequence[Path],
    role: str,
    *,
    sample_id: Optional[str] = None,
    method: Optional[str] = None,
    variant: Optional[str] = None,
    extra_positive: Sequence[str] = (),
) -> Match:
    ranked: list[tuple[float, Path]] = []
    for p in files:
        s = role_score(p, role)
        full = norm(str(p))

        if sample_id:
            sid = norm(sample_id)
            if sid in full:
                s += 24
            else:
                # Very strong penalty; prevents cross-sample leakage.
                s -= 40

        if method:
            s += method_score(p, method)

        if variant:
            s += variant_score(p, variant)

        for tok in extra_positive:
            if norm(tok) in full:
                s += 2

        ranked.append((s, p))

    ranked.sort(key=lambda x: (x[0], -len(str(x[1]))), reverse=True)
    if not ranked or ranked[0][0] <= 0:
        return Match(role, None, 0.0, "no positive match")
    return Match(role, ranked[0][1], ranked[0][0], "")


def copy_match(m: Match, dst: Path, reporter: Reporter, required: bool = True, dry_run: bool = False) -> None:
    if m.path is None:
        msg = f"{dst}: {m.role} not found"
        if required:
            reporter.miss(msg)
        else:
            reporter.log("[optional missing] " + msg)
        return
    reporter.log(f"[MATCH {m.score:5.1f}] {m.role:18s} <- {m.path}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(m.path, dst)


def copy_related_preview(src: Path, dst_dir: Path, reporter: Reporter, dry_run: bool = False) -> None:
    # If we chose TIFF/NPY, also copy same-stem preview if available.
    if src.suffix.lower() in PREVIEW_EXTS:
        return
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = src.with_suffix(ext)
        if p.is_file():
            reporter.log(f"[PREVIEW] {p}")
            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst_dir / p.name)
            break


def copy_nearby_tables(anchor: Path, dst_dir: Path, reporter: Reporter, dry_run: bool = False) -> None:
    # Search a few ancestors for relevant tables; copy only small text/table files.
    seen: set[Path] = set()
    for anc in [anchor.parent, *list(anchor.parents)[:4]]:
        if not anc.is_dir():
            continue
        for name in (
            "metrics.csv", "manifest.csv", "summary.json", "grouped_results.csv",
            "main_result_table.csv", "paired_comparisons.csv", "ablation_table.csv",
            "variant_config.json", "config.json",
        ):
            p = anc / name
            if p.is_file() and p not in seen:
                seen.add(p)
                reporter.log(f"[TABLE] {p}")
                if not dry_run:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    # disambiguate by nearest parent name
                    out_name = f"{p.parent.name}__{p.name}"
                    shutil.copy2(p, dst_dir / out_name)


def make_zip(folder: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(folder.parent))


# ---------------------------------------------------------------------
# Figure collectors
# ---------------------------------------------------------------------

def collect_fig3(
    all_files: Sequence[Path],
    sample_ids: Sequence[str],
    out_root: Path,
    reporter: Reporter,
    low_id: Optional[str],
    high_id: Optional[str],
    dry_run: bool,
) -> Path:
    low_id = low_id or choose_sample(
        sample_ids,
        positive=("single", "radial", "low"),
        negative=("protected", "stage_b", "high"),
    )
    high_id = high_id or choose_sample(
        sample_ids,
        positive=("multi", "source", "high"),
        negative=("protected", "stage_b", "low"),
    )

    reporter.log("\n=== FIGURE 3 / Stage-A representative ===")
    reporter.log(f"selected low-single : {low_id}")
    reporter.log(f"selected high-multi : {high_id}")

    stage = out_root / "_staging" / "Fig3_StageA_main_assets"
    if stage.exists() and not dry_run:
        shutil.rmtree(stage)
    if not dry_run:
        stage.mkdir(parents=True, exist_ok=True)

    for label, sid in (("A_low_single", low_id), ("B_high_multi", high_id)):
        if not sid:
            reporter.miss(f"Fig3 sample selection failed: {label}")
            continue
        case_dir = stage / label

        # source
        source_roles = ("observed", "background_true", "clean_true")
        for role in source_roles:
            m = select_best_file(all_files, role, sample_id=sid)
            ext = m.path.suffix.lower() if m.path else ".tif"
            dst = case_dir / "source" / f"{role}{ext}"
            copy_match(m, dst, reporter, required=True, dry_run=dry_run)
            if m.path:
                copy_related_preview(m.path, case_dir / "source" / "preview", reporter, dry_run)

        # methods
        for method in ("gaussian", "dct2_ls", "original_pinn"):
            for role in ("background_pred", "residual_pred"):
                m = select_best_file(all_files, role, sample_id=sid, method=method)
                ext = m.path.suffix.lower() if m.path else ".tif"
                dst = case_dir / method / f"{role}{ext}"
                copy_match(m, dst, reporter, required=True, dry_run=dry_run)
                if m.path:
                    copy_related_preview(m.path, case_dir / method / "preview", reporter, dry_run)

    if not dry_run:
        (stage / "README.txt").write_text(
            "Figure 3 assets. Use TIFF/TIF scientific arrays for final rendering.\n"
            "For each scene: ground-truth background + three estimated backgrounds share one display range;\n"
            "clean_true + three residuals share another display range.\n"
            "Do not replace the frozen representative seed by visual selection.\n",
            encoding="utf-8",
        )
    zip_path = out_root / "Fig3_StageA_main_assets.zip"
    if not dry_run:
        make_zip(stage, zip_path)
    return zip_path


def collect_fig4(
    all_files: Sequence[Path],
    sample_ids: Sequence[str],
    out_root: Path,
    reporter: Reporter,
    sample_override: Optional[str],
    dry_run: bool,
) -> Path:
    # Prefer a difficult high-pollution case where capacity differences are visible.
    sid = sample_override or choose_sample(
        sample_ids,
        positive=("multi", "high"),
        negative=("protected", "stage_b"),
    )
    if not sid:
        sid = choose_sample(
            sample_ids,
            positive=("eccentric", "high"),
            negative=("protected", "stage_b"),
        )

    reporter.log("\n=== FIGURE 4 / Original PINN ablation ===")
    reporter.log(f"selected sample: {sid}")

    stage = out_root / "_staging" / "Fig4_PINN_ablation_assets"
    if stage.exists() and not dry_run:
        shutil.rmtree(stage)
    if not dry_run:
        stage.mkdir(parents=True, exist_ok=True)

    if sid:
        # shared source
        for role in ("observed", "background_true", "clean_true"):
            m = select_best_file(all_files, role, sample_id=sid)
            ext = m.path.suffix.lower() if m.path else ".tif"
            copy_match(m, stage / "source" / f"{role}{ext}", reporter, required=(role == "observed"), dry_run=dry_run)
            if m.path:
                copy_related_preview(m.path, stage / "source" / "preview", reporter, dry_run)

        # variants
        for variant in (
            "data_only",
            "single_forcing",
            "multi_fixed_centers",
            "multi_bounded_centers",
        ):
            for role in ("background_pred", "residual_pred"):
                m = select_best_file(all_files, role, sample_id=sid, variant=variant)
                ext = m.path.suffix.lower() if m.path else ".tif"
                copy_match(m, stage / variant / f"{role}{ext}", reporter, required=True, dry_run=dry_run)
                if m.path:
                    copy_related_preview(m.path, stage / variant / "preview", reporter, dry_run)
                    copy_nearby_tables(m.path, stage / "tables", reporter, dry_run)
    else:
        reporter.miss("Fig4 sample selection failed")

    if not dry_run:
        (stage / "README.txt").write_text(
            "Figure 4 assets. Keep one sample fixed across all ablation variants.\n"
            "Variants: data_only, single_forcing, multi_fixed_centers, multi_bounded_centers.\n"
            "Do not choose a different sample/seed after inspecting visual quality.\n",
            encoding="utf-8",
        )
    zip_path = out_root / "Fig4_PINN_ablation_assets.zip"
    if not dry_run:
        make_zip(stage, zip_path)
    return zip_path


def collect_fig6(
    all_files: Sequence[Path],
    sample_ids: Sequence[str],
    out_root: Path,
    reporter: Reporter,
    high_override: Optional[str],
    low_override: Optional[str],
    dry_run: bool,
) -> Path:
    high_id = high_override or choose_sample(
        sample_ids,
        positive=("protected", "overlap", "high"),
        negative=("low",),
    )
    low_id = low_override or choose_sample(
        sample_ids,
        positive=("protected", "overlap", "low"),
        negative=("high",),
    )

    # fallback: if naming does not contain "protected_overlap"
    if not high_id:
        high_id = choose_sample(sample_ids, positive=("overlap", "high"))
    if not low_id:
        low_id = choose_sample(sample_ids, positive=("overlap", "low"))

    reporter.log("\n=== FIGURE 6 / Stage-B protected structure ===")
    reporter.log(f"selected high-overlap: {high_id}")
    reporter.log(f"selected low-overlap : {low_id}")

    stage = out_root / "_staging" / "Fig6_StageB_assets"
    if stage.exists() and not dry_run:
        shutil.rmtree(stage)
    if not dry_run:
        stage.mkdir(parents=True, exist_ok=True)

    for label, sid in (("high_overlap", high_id), ("low_overlap", low_id)):
        if not sid:
            reporter.miss(f"Fig6 sample selection failed: {label}")
            continue
        case_dir = stage / label

        for role in ("base_clean", "protected_true", "clean_true", "background_true", "observed"):
            m = select_best_file(all_files, role, sample_id=sid)
            ext = m.path.suffix.lower() if m.path else ".tif"
            copy_match(m, case_dir / "source" / f"{role}{ext}", reporter, required=(role != "base_clean"), dry_run=dry_run)
            if m.path:
                copy_related_preview(m.path, case_dir / "source" / "preview", reporter, dry_run)

        # benchmark + paired SGC paths
        for method in (
            "gaussian",
            "dct2_ls",
            "original_pinn",
            "matched_budget_control",
            "sgc_pinn",
        ):
            for role in ("background_pred", "residual_pred"):
                m = select_best_file(all_files, role, sample_id=sid, method=method)
                ext = m.path.suffix.lower() if m.path else ".tif"
                copy_match(m, case_dir / method / f"{role}{ext}", reporter, required=True, dry_run=dry_run)
                if m.path:
                    copy_related_preview(m.path, case_dir / method / "preview", reporter, dry_run)
                    copy_nearby_tables(m.path, case_dir / "tables", reporter, dry_run)

        # optional mask
        mm = select_best_file(all_files, "mask", sample_id=sid)
        if mm.path:
            ext = mm.path.suffix.lower()
            copy_match(mm, case_dir / "source" / f"protected_mask{ext}", reporter, required=False, dry_run=dry_run)

    if not dry_run:
        (stage / "README.txt").write_text(
            "Figure 6 assets. Recommended final panel: high_overlap.\n"
            "Keep the protected region identical across methods.\n"
            "Use one shared display range for recovered backgrounds and another for residuals.\n"
            "Projection ratio ideal value is 1; larger alone does not establish correct identification.\n",
            encoding="utf-8",
        )
    zip_path = out_root / "Fig6_StageB_assets.zip"
    if not dry_run:
        make_zip(stage, zip_path)
    return zip_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Auto-match frozen Stage11 outputs and package assets for Figures 3, 4, and 6.",
    )
    ap.add_argument("--root", type=Path, default=Path("."), help="Stage11/project root")
    ap.add_argument("--out", type=Path, default=None, help="Output directory")
    ap.add_argument("--dry-run", action="store_true", help="Only report matches; copy nothing")
    ap.add_argument("--strict", action="store_true", help="Exit nonzero if any required asset is missing")
    ap.add_argument("--zip", action="store_true", help="Also create one combined all-assets ZIP")

    ap.add_argument("--fig3-low", default=None, help="Override Fig3 low/single sample ID")
    ap.add_argument("--fig3-high", default=None, help="Override Fig3 high/multi sample ID")
    ap.add_argument("--fig4-sample", default=None, help="Override Fig4 ablation sample ID")
    ap.add_argument("--fig6-high", default=None, help="Override Fig6 protected-overlap high sample ID")
    ap.add_argument("--fig6-low", default=None, help="Override Fig6 protected-overlap low sample ID")

    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] root not found: {root}", file=sys.stderr)
        return 2

    out_root = (args.out.expanduser().resolve() if args.out
                else root / "figure_assets_auto")
    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)

    rep = Reporter()
    rep.log(f"[ROOT] {root}")
    rep.log(f"[OUT ] {out_root}")
    rep.log("[SCAN] indexing scientific arrays / previews ...")

    all_files = list(iter_files(root, SCI_EXTS | PREVIEW_EXTS))
    sample_ids = find_sample_ids(root)

    rep.log(f"[SCAN] indexed files: {len(all_files)}")
    rep.log(f"[SCAN] sample-id candidates: {len(sample_ids)}")

    if sample_ids:
        rep.log("[SCAN] first sample IDs:")
        for sid in sample_ids[:20]:
            rep.log("       " + sid)

    zips = []
    zips.append(collect_fig3(
        all_files, sample_ids, out_root, rep,
        args.fig3_low, args.fig3_high, args.dry_run
    ))
    zips.append(collect_fig4(
        all_files, sample_ids, out_root, rep,
        args.fig4_sample, args.dry_run
    ))
    zips.append(collect_fig6(
        all_files, sample_ids, out_root, rep,
        args.fig6_high, args.fig6_low, args.dry_run
    ))

    report_path = out_root / "AUTO_MATCH_REPORT.txt"
    if not args.dry_run:
        rep.write(report_path)

    if args.zip and not args.dry_run:
        combined = out_root / "All_Fig3_Fig4_Fig6_assets.zip"
        if combined.exists():
            combined.unlink()
        with zipfile.ZipFile(combined, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(out_root.glob("Fig*_assets.zip")):
                zf.write(p, p.name)
            if report_path.exists():
                zf.write(report_path, report_path.name)
        rep.log(f"[ZIP] {combined}")

    rep.log("\n=== SUMMARY ===")
    for p in zips:
        if args.dry_run:
            rep.log(f"would create: {p}")
        elif p.exists():
            rep.log(f"created: {p} ({p.stat().st_size / 1024**2:.1f} MB)")
        else:
            rep.log(f"not created: {p}")

    if rep.missing:
        rep.log(f"required items missing: {len(rep.missing)}")
        if args.strict:
            return 3
        rep.log("Tip: rerun with explicit --fig3-low/--fig3-high/--fig4-sample/--fig6-high overrides if needed.")
    else:
        rep.log("all required auto-matched assets found.")

    rep.log("No experiments were rerun. No original outputs were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
