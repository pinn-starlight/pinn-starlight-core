"""Verify that a restored Stage11 tree is ready for offline paper analysis."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import tifffile as tif

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def finite_tiff(path: Path):
    arr = tif.imread(path)
    return arr.ndim == 2 and np.isfinite(arr).all()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-derived", action="store_true")
    args = parser.parse_args()

    required = [
        PROJECT_ROOT / "collections/synthetic_formal/manifest.csv",
        PROJECT_ROOT / "collections/synthetic_stage_b/manifest.csv",
    ]
    missing = [str(p) for p in required if not p.is_file()]

    outputs = PROJECT_ROOT / "experiments/outputs"
    backgrounds = list(outputs.rglob("background_pred.tif")) if outputs.exists() else []
    bad_tiffs = []
    missing_derived = []
    for bg in backgrounds:
        try:
            if not finite_tiff(bg):
                bad_tiffs.append(str(bg))
        except Exception:
            bad_tiffs.append(str(bg))
        if args.require_derived:
            for name in ("observed.tif", "residual_pred.tif", "observed.png", "background_pred.png", "residual_pred.png"):
                if not (bg.parent / name).is_file():
                    missing_derived.append(str(bg.parent / name))

    key_tables = list(outputs.rglob("metrics.csv")) if outputs.exists() else []
    key_locks = list(outputs.rglob("*configuration_lock*.json")) if outputs.exists() else []

    print("synthetic manifests:", 2 - len(missing), "/ 2")
    print("background prediction TIFFs:", len(backgrounds))
    print("metrics.csv files:", len(key_tables))
    print("configuration locks:", len(key_locks))
    print("bad prediction TIFFs:", len(bad_tiffs))
    if args.require_derived:
        print("missing derived files:", len(missing_derived))

    if missing:
        print("\nMissing required:")
        for x in missing:
            print(" -", x)
    if bad_tiffs:
        print("\nUnreadable/non-finite background predictions:")
        for x in bad_tiffs[:50]:
            print(" -", x)
    if missing_derived:
        print("\nMissing derived outputs:")
        for x in missing_derived[:50]:
            print(" -", x)

    if missing or bad_tiffs or (args.require_derived and missing_derived):
        raise SystemExit(2)
    print("Stage11 offline reconstruction readiness: PASS")


if __name__ == "__main__":
    main()
