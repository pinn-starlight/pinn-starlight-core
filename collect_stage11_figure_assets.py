#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect the frozen scientific arrays needed for manuscript Figures 3, 4 and 6
from a completed Stage11 server workspace, then create three separate ZIP files.

Default project root:
    /workspace/stage11

Outputs:
    figure_assets_export/Fig3_StageA_main_assets.zip
    figure_assets_export/Fig4_PINN_ablation_assets.zip
    figure_assets_export/Fig6_StageB_assets.zip

This script DOES NOT rerun experiments and DOES NOT modify original outputs.
It only copies frozen files into temporary staging directories and zips them.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import zipfile
from pathlib import Path


DEFAULT_ROOT = Path("/workspace/stage11")
DEFAULT_SEED = 20260728

# Figure 3: pre-fixed representative Stage-A samples.
FIG3_SAMPLES = (
    "clean_single_001_single_radial_low",
    "clean_single_001_multi_source_high",
)

# Figure 4: package two pre-fixed candidates so the final representative
# can be chosen by scientific relevance, not by post-hoc visual quality.
FIG4_SAMPLES = (
    "clean_single_001_single_eccentric_high",
    "clean_single_001_multi_source_high",
)

# Figure 6: protected-overlap Stage-B pair.
# The high-pollution case is the recommended manuscript panel; low is included
# because both share the same protected target and is useful for checking.
FIG6_SAMPLES = (
    "clean_single_001_protected_overlap_low",
    "clean_single_001_protected_overlap_high",
)

ABLATION_VARIANTS = (
    "data_only",
    "single_component",
    "multi_fixed_centers",
    "multi_bounded_centers",
)

PRED_ARRAYS = (
    "observed.tif",
    "background_pred.tif",
    "residual_pred.tif",
)
PRED_PREVIEWS = (
    "observed.png",
    "background_pred.png",
    "residual_pred.png",
)


class Collector:
    def __init__(self, root: Path, out_root: Path, seed: int):
        self.root = root.resolve()
        self.out_root = out_root.resolve()
        self.seed = int(seed)
        self.missing: list[str] = []
        self.copied: list[tuple[str, str]] = []

    def require_file(self, src: Path, dst: Path) -> None:
        if not src.is_file():
            self.missing.append(str(src))
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.copied.append((str(src), str(dst)))

    def optional_file(self, src: Path, dst: Path) -> None:
        if not src.is_file():
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        self.copied.append((str(src), str(dst)))

    def copy_dir_files(self, src_dir: Path, dst_dir: Path, *, required_tif=True) -> None:
        for name in PRED_ARRAYS:
            fn = src_dir / name
            if required_tif:
                self.require_file(fn, dst_dir / name)
            else:
                self.optional_file(fn, dst_dir / name)
        for name in PRED_PREVIEWS:
            self.optional_file(src_dir / name, dst_dir / name)

    def copy_source_sample(self, collection_root: Path, sample_id: str, dst_dir: Path,
                           names: tuple[str, ...]) -> None:
        src_dir = collection_root / sample_id
        for base in names:
            self.require_file(
                src_dir / f"{base}_{sample_id}.tif",
                dst_dir / f"{base}_{sample_id}.tif",
            )
            self.optional_file(
                src_dir / f"{base}_{sample_id}.png",
                dst_dir / f"{base}_{sample_id}.png",
            )
        self.optional_file(
            src_dir / f"metadata_{sample_id}.json",
            dst_dir / f"metadata_{sample_id}.json",
        )

    def filter_csv(self, src: Path, dst: Path, sample_ids: tuple[str, ...],
                   sample_field: str = "sample_id") -> None:
        if not src.is_file():
            self.missing.append(str(src))
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = [r for r in reader if r.get(sample_field) in sample_ids]
            fieldnames = reader.fieldnames or []
        with dst.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def copy_table(self, src: Path, dst: Path, required=True) -> None:
        if required:
            self.require_file(src, dst)
        else:
            self.optional_file(src, dst)

    @staticmethod
    def write_readme(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.strip() + "\n", encoding="utf-8")

    @staticmethod
    def zip_dir(folder: Path, zip_path: Path) -> None:
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(folder.parent))

    def _reset_stage(self, name: str) -> Path:
        stage = self.out_root / "_staging" / name
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True, exist_ok=True)
        return stage

    def collect_fig3(self) -> Path:
        stage = self._reset_stage("Fig3_StageA_main_assets")
        synth = self.root / "collections" / "synthetic_formal"
        e2 = self.root / "experiments" / "outputs" / "e2_benchmark"

        for sid in FIG3_SAMPLES:
            self.copy_source_sample(
                synth, sid, stage / "source" / sid,
                ("observed", "background_true", "clean_true"),
            )
            self.copy_dir_files(
                e2 / "fft_gaussian" / "deterministic" / sid,
                stage / "gaussian" / sid,
            )
            self.copy_dir_files(
                e2 / "fourier_ls" / "deterministic" / sid,
                stage / "dct2_ls" / sid,
            )
            self.copy_dir_files(
                e2 / "pinn" / f"seed_{self.seed}" / sid,
                stage / "original_pinn" / sid,
            )

        self.filter_csv(
            synth / "manifest.csv",
            stage / "tables" / "stageA_manifest_selected.csv",
            FIG3_SAMPLES,
        )
        self.filter_csv(
            e2 / "metrics.csv",
            stage / "tables" / "e2_metrics_selected.csv",
            FIG3_SAMPLES,
        )
        for name in ("main_result_table.csv", "grouped_results.csv", "summary.json"):
            self.copy_table(e2 / name, stage / "tables" / name, required=False)

        self.write_readme(
            stage / "README.txt",
            f"""
FIGURE 3 — Stage-A representative spatial results

Fixed samples:
  A low-pollution/single-source:
    {FIG3_SAMPLES[0]}
  B high-pollution/multi-source:
    {FIG3_SAMPLES[1]}

Original PINN representative seed:
  {self.seed}

Suggested panel mapping:
  Observed          -> source/<sample>/observed_<sample>.tif
  Ground truth      -> source/<sample>/background_true_<sample>.tif
  Clean reference   -> source/<sample>/clean_true_<sample>.tif
  Gaussian          -> gaussian/<sample>/background_pred.tif / residual_pred.tif
  DCT-II LS         -> dct2_ls/<sample>/background_pred.tif / residual_pred.tif
  Original PINN     -> original_pinn/<sample>/background_pred.tif / residual_pred.tif

Display rule:
  - Use TIFF scientific arrays for the final figure.
  - Background_true and all estimated backgrounds in one scene share one vmin/vmax.
  - Clean_true and all background-subtracted residuals in one scene share another vmin/vmax.
  - Do not choose another seed by visual appearance.
""",
        )
        zip_path = self.out_root / "Fig3_StageA_main_assets.zip"
        self.zip_dir(stage, zip_path)
        return zip_path

    def collect_fig4(self) -> Path:
        stage = self._reset_stage("Fig4_PINN_ablation_assets")
        synth = self.root / "collections" / "synthetic_formal"
        e3 = self.root / "experiments" / "outputs" / "e3_ablation"

        for sid in FIG4_SAMPLES:
            self.copy_source_sample(
                synth, sid, stage / "source" / sid,
                ("observed", "background_true", "clean_true"),
            )
            for variant in ABLATION_VARIANTS:
                self.copy_dir_files(
                    e3 / variant / f"seed_{self.seed}" / sid,
                    stage / "variants" / variant / sid,
                )
                self.optional_file(
                    e3 / variant / "variant_config.json",
                    stage / "variant_configs" / f"{variant}.json",
                )

        self.filter_csv(
            e3 / "metrics.csv",
            stage / "tables" / "e3_metrics_selected.csv",
            FIG4_SAMPLES,
        )
        self.copy_table(
            e3 / "ablation_table.csv",
            stage / "tables" / "ablation_table.csv",
            required=True,
        )
        self.copy_table(e3 / "summary.json", stage / "tables" / "summary.json", required=False)

        self.write_readme(
            stage / "README.txt",
            f"""
FIGURE 4 — Original PINN ablation

Pre-fixed candidate samples:
  Center-freedom candidate:
    {FIG4_SAMPLES[0]}
  Multi-component-capacity candidate:
    {FIG4_SAMPLES[1]}

Representative seed:
  {self.seed}

Variants:
  data_only
  single_component
  multi_fixed_centers
  multi_bounded_centers

Use one sample consistently across all four variants in the final panel.
Do not select a different seed or sample because it looks better.

Each variant folder contains:
  observed.tif
  background_pred.tif
  residual_pred.tif
and optional PNG previews.
""",
        )
        zip_path = self.out_root / "Fig4_PINN_ablation_assets.zip"
        self.zip_dir(stage, zip_path)
        return zip_path

    def collect_fig6(self) -> Path:
        stage = self._reset_stage("Fig6_StageB_assets")
        synth_b = self.root / "collections" / "synthetic_stage_b"
        bench = self.root / "experiments" / "outputs" / "stage_b" / "benchmark"
        pair = self.root / "experiments" / "outputs" / "stage_b" / "final_pinn_validation"

        for sid in FIG6_SAMPLES:
            self.copy_source_sample(
                synth_b, sid, stage / "source" / sid,
                ("base_clean", "protected_true", "clean_true", "background_true", "observed"),
            )
            self.copy_dir_files(
                bench / "fft_gaussian" / "deterministic" / sid,
                stage / "gaussian" / sid,
            )
            self.copy_dir_files(
                bench / "fourier_ls" / "deterministic" / sid,
                stage / "dct2_ls" / sid,
            )
            self.copy_dir_files(
                bench / "pinn" / f"seed_{self.seed}" / sid,
                stage / "original_pinn" / sid,
            )
            self.copy_dir_files(
                pair / f"seed_{self.seed}" / sid / "data_finetune_control",
                stage / "matched_budget_control" / sid,
            )
            self.copy_dir_files(
                pair / f"seed_{self.seed}" / sid / "pinn_decoupled_gradient",
                stage / "sgc_pinn" / sid,
            )

        self.filter_csv(
            synth_b / "manifest.csv",
            stage / "tables" / "stageB_manifest_selected.csv",
            FIG6_SAMPLES,
        )
        self.filter_csv(
            bench / "metrics.csv",
            stage / "tables" / "stageB_benchmark_metrics_selected.csv",
            FIG6_SAMPLES,
        )
        self.filter_csv(
            pair / "metrics.csv",
            stage / "tables" / "stageB_pair_metrics_selected.csv",
            FIG6_SAMPLES,
        )
        for name in ("main_result_table.csv", "grouped_results.csv", "summary.json"):
            self.copy_table(bench / name, stage / "tables" / f"benchmark_{name}", required=False)
        for name in ("paired_comparisons.csv", "summary.json"):
            self.copy_table(pair / name, stage / "tables" / f"pair_{name}", required=False)

        self.write_readme(
            stage / "README.txt",
            f"""
FIGURE 6 — Stage-B protected low-frequency structure

Fixed Stage-B protected-overlap samples:
  low : {FIG6_SAMPLES[0]}
  high: {FIG6_SAMPLES[1]}

Recommended manuscript representative:
  {FIG6_SAMPLES[1]}

Representative seed for Original PINN and the paired SGC experiment:
  {self.seed}

Source arrays:
  base_clean       = camera star-field reference before protected structure
  protected_true   = injected protected low-frequency structure
  clean_true       = base_clean + protected_true
  background_true  = artificial pollution background
  observed         = clean_true + background_true

Methods:
  gaussian/
  dct2_ls/
  original_pinn/
  matched_budget_control/
  sgc_pinn/

For the final Figure 6:
  - Use the same protected region/mask across all methods.
  - Use one shared display range for recovered backgrounds.
  - Use another shared display range for residual/cleaned images.
  - Keep scientific TIFF pixels unchanged.
  - Do not interpret larger projection ratio alone as better identification.
""",
        )
        zip_path = self.out_root / "Fig6_StageB_assets.zip"
        self.zip_dir(stage, zip_path)
        return zip_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect frozen Stage11 assets for manuscript Figures 3, 4 and 6."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Stage11 project root (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <root>/figure_assets_export)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Representative frozen seed (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    root = args.root.expanduser()
    out = (args.out.expanduser() if args.out else root / "figure_assets_export")

    if not root.is_dir():
        print(f"[ERROR] Project root does not exist: {root}", file=sys.stderr)
        return 2

    c = Collector(root, out, args.seed)
    print(f"[INFO] root = {c.root}")
    print(f"[INFO] out  = {c.out_root}")
    print(f"[INFO] seed = {c.seed}")

    zips = [
        c.collect_fig3(),
        c.collect_fig4(),
        c.collect_fig6(),
    ]

    if c.missing:
        print("\n[ERROR] Some required files are missing; ZIPs may be incomplete:", file=sys.stderr)
        for p in c.missing:
            print("  MISSING:", p, file=sys.stderr)
        print("\nNo source files were modified.", file=sys.stderr)
        return 3

    print("\n[OK] Created:")
    for p in zips:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p}  ({size_mb:.1f} MB)")

    print("\n[OK] No experiments were rerun and no original outputs were modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
