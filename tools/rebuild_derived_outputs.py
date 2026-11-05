"""Rebuild derivable Stage11 TIFF/PNG output files locally.

This is intentionally *not* model retraining.  It rebuilds only data that was
omitted from the high-priority transfer because it is mathematically
deterministic from either:
    local/generated synthetic observed TIFF + transferred background_pred.tif
or, for E4:
    the frozen real-image source + transferred background_pred.tif

Reconstructed:
- experiments/outputs/**/observed.tif
- experiments/outputs/**/residual_pred.tif
- observed/background/residual display PNGs
- synthetic_formal/synthetic_stage_b preview PNGs when omitted

The scientific prediction itself (background_pred.tif) is never synthesized.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Import from the restored Stage11 project.
from experiments.common.utils import experiment_utils as utils
from experiments.scripts.e4 import run as e4_run


def read_manifest(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["sample_id"]: row for row in rows}


def load_maps():
    a = read_manifest(PROJECT_ROOT / "collections/synthetic_formal/manifest.csv")
    b = read_manifest(PROJECT_ROOT / "collections/synthetic_stage_b/manifest.csv")
    return a, b


def load_e4_sources() -> dict[str, Path]:
    sources = {}
    for path in utils.supported_image_files(utils.REAL_HIGH_POLLUTION_ROOT):
        if path.stem in sources:
            raise ValueError(f"duplicate E4 source stem: {path.stem}")
        sources[path.stem] = path
    return sources


def load_e4_observed(image_name: str, sources: dict[str, Path]) -> np.ndarray:
    source = sources.get(image_name)
    if source is None:
        raise FileNotFoundError(
            f"E4 source image {image_name!r} was not found in "
            f"{utils.REAL_HIGH_POLLUTION_ROOT}"
        )
    observed = utils.load_gray_image(source, downsample=e4_run.DOWNSAMPLE)
    return e4_run._center_crop(observed, e4_run.CROP_SIZE)


def resolve_observed(row: dict) -> Path:
    raw = Path(row["observed"])
    if raw.is_absolute():
        return raw
    # Manifest paths are project/data-root relative in the formal dataset.
    if raw.parts and raw.parts[0] == "collections":
        return PROJECT_ROOT / raw
    if len(raw.parts) >= 2 and raw.parts[:2] == ("data", "collections"):
        return PROJECT_ROOT / Path(*raw.parts[1:])
    return PROJECT_ROOT / raw


def rebuild_result_dirs(*, force=False):
    map_a, map_b = load_maps()
    e4_sources = load_e4_sources()
    outputs = PROJECT_ROOT / "experiments/outputs"
    if not outputs.exists():
        print("No experiments/outputs directory yet.")
        return {"rebuilt": 0, "skipped_unknown": 0}

    rebuilt = 0
    skipped_unknown = 0
    for bg_path in outputs.rglob("background_pred.tif"):
        result_dir = bg_path.parent
        sample_id = result_dir.name
        rel_parts = result_dir.relative_to(outputs).parts

        if rel_parts and rel_parts[0] == "e4_real":
            if len(rel_parts) < 3:
                skipped_unknown += 1
                continue
            image_name = rel_parts[1]
            observed = load_e4_observed(image_name, e4_sources)
        else:
            is_stage_b = "stage_b" in rel_parts
            row = (map_b if is_stage_b else map_a).get(sample_id)
            if row is None:
                skipped_unknown += 1
                continue
            observed_path = resolve_observed(row)
            if not observed_path.is_file():
                raise FileNotFoundError(
                    f"observed TIFF for {sample_id} not found: {observed_path}"
                )
            observed = utils.load_float_tiff(observed_path)

        background = utils.load_float_tiff(bg_path)
        if background.shape != observed.shape:
            raise ValueError(
                f"shape mismatch at {bg_path}: observed={observed.shape}, "
                f"background={background.shape}"
            )
        residual = utils.reconstruct_residual(observed, background)

        out_observed = result_dir / "observed.tif"
        out_residual = result_dir / "residual_pred.tif"
        if force or not out_observed.is_file():
            utils.save_float_tiff(out_observed, observed)
        if force or not out_residual.is_file():
            utils.save_float_tiff(out_residual, residual)

        display = {
            "observed": observed,
            "background_pred": background,
            "residual_pred": residual,
        }
        for name, image in display.items():
            png = result_dir / f"{name}.png"
            if force or not png.is_file():
                utils.save_display_png(png, image, clip=True)
        rebuilt += 1

    return {"rebuilt": rebuilt, "skipped_unknown": skipped_unknown}


def rebuild_synthetic_previews(*, force=False):
    count = 0
    for root in (
        PROJECT_ROOT / "collections/synthetic_formal",
        PROJECT_ROOT / "collections/synthetic_stage_b",
    ):
        if not root.exists():
            continue
        for tif in root.rglob("*.tif"):
            png = tif.with_suffix(".png")
            if png.is_file() and not force:
                continue
            image = utils.load_float_tiff(tif)
            # The Stage-A native-background audit deliberately uses a percentile
            # display only for clean_true_flattened; all other generated previews
            # use unit-range clipping.
            clip = tif.stem != "clean_true_flattened"
            utils.save_display_png(png, image, clip=clip)
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-synthetic-previews", action="store_true")
    args = parser.parse_args()

    result = rebuild_result_dirs(force=args.force)
    preview_count = 0
    if not args.skip_synthetic_previews:
        preview_count = rebuild_synthetic_previews(force=args.force)

    print("result directories rebuilt:", result["rebuilt"])
    print("non-synthetic result directories skipped:", result["skipped_unknown"])
    print("synthetic preview PNGs rebuilt:", preview_count)


if __name__ == "__main__":
    main()
