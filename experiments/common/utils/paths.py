"""Project layout constants and manifest helpers (standard library only).

This module exists so the release CLI can answer "where is the data?" without
importing numpy / torch / tifffile / PIL.  Keep it dependency-free: importing
anything heavy here would make ``pinn-starlight doctor`` crash on a machine
that has not installed the estimator extras yet.

``experiments.common.utils.experiment_utils`` re-exports every name defined
here, so existing call sites keep working unchanged.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from experiments.configs.formal_experiment import FORMAL_SEEDS
from experiments.configs.split_protocols import active_protocol_from_env

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_OUTPUT_ROOT = PROJECT_ROOT / "experiments/outputs"
SPLIT_MODE, CV_FOLD = active_protocol_from_env()
if SPLIT_MODE == "standard":
    OUTPUT_ROOT = BASE_OUTPUT_ROOT
    OUTPUT_NAMESPACE = "standard"
else:
    OUTPUT_ROOT = BASE_OUTPUT_ROOT / "cv5" / f"fold_{CV_FOLD}"
    OUTPUT_NAMESPACE = f"cv5/fold_{CV_FOLD}"


def _resolve_data_root() -> Path:
    """Resolve the external data root without changing the experiment protocol.

    Supported layouts, in priority order:
    1. ``PINN_STARLIGHT_DATA_ROOT`` points to a directory containing
       ``collections/`` and ``test/``;
    2. root-level ``collections/`` / ``test/`` beside the project files;
    3. legacy ``data/collections`` / ``data/test``.
    """
    configured = os.environ.get("PINN_STARLIGHT_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    root_layout = PROJECT_ROOT
    legacy_layout = PROJECT_ROOT / "data"
    if (root_layout / "collections").exists() or (root_layout / "test").exists():
        return root_layout
    if (legacy_layout / "collections").exists() or (legacy_layout / "test").exists():
        return legacy_layout

    # The user's current formal layout is root-level. Returning PROJECT_ROOT when
    # no data are mounted also keeps imports side-effect free; preflight gives a
    # precise error before any formal run if the data are elsewhere.
    return root_layout


DATA_ROOT = _resolve_data_root()
COLLECTION_ROOT = DATA_ROOT / "collections"
COLLECTION_MANIFEST = COLLECTION_ROOT / "manifest.csv"
TEST_ROOT = DATA_ROOT / "test"
NATIVE_TEST_IMAGE = TEST_ROOT / "native_test.tif"
REAL_HIGH_POLLUTION_ROOT = COLLECTION_ROOT / "real_high_pollution"

# ``collections/synthetic`` is the legacy 10-sample set in the user's current
# data tree. Formal Stage-7+ data are isolated in a new directory.
LEGACY_SYNTHETIC_ROOT = COLLECTION_ROOT / "synthetic"
SYNTHETIC_DATA_ROOT = COLLECTION_ROOT / "synthetic_formal"
if SPLIT_MODE == "standard":
    SYNTHETIC_ROOT = SYNTHETIC_DATA_ROOT
else:
    SYNTHETIC_ROOT = SYNTHETIC_DATA_ROOT / "cv5" / f"fold_{CV_FOLD}"
SYNTHETIC_MANIFEST = SYNTHETIC_ROOT / "manifest.csv"
SELECTION_ROOT = OUTPUT_ROOT / "selection"
PINN_SELECTION_ROOT = SELECTION_ROOT / "pinn"
GAUSSIAN_SELECTION_ROOT = SELECTION_ROOT / "fft_gaussian"
FOURIER_LS_SELECTION_ROOT = SELECTION_ROOT / "fourier_ls"
PINN_SELECTION_ARTIFACT = PINN_SELECTION_ROOT / "pinn_selected.json"
GAUSSIAN_SELECTION_ARTIFACT = GAUSSIAN_SELECTION_ROOT / "gaussian_selected.json"
FOURIER_LS_SELECTION_ARTIFACT = FOURIER_LS_SELECTION_ROOT / "fourier_ls_selected.json"
GLOBAL_CONFIGURATION_LOCK = OUTPUT_ROOT / "configuration_lock/global_configuration_lock.json"
SEEDS = tuple(FORMAL_SEEDS)


def to_absolute_path(path):
    """把 manifest 中的项目相对路径转换为绝对路径。"""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_collection_path(path) -> Path:
    """Resolve one ``collections/manifest.csv`` path in all supported layouts."""
    path = Path(path)
    if path.is_absolute():
        return path
    parts = path.parts
    if len(parts) >= 2 and parts[:2] == ("data", "collections"):
        return DATA_ROOT / Path(*parts[1:])
    if parts and parts[0] == "collections":
        return DATA_ROOT / path
    return COLLECTION_ROOT / path


def supported_image_files(directory: Path):
    """Return deterministic real-image candidates from one directory."""
    directory = Path(directory)
    extensions = {
        ".nef",
        ".cr2",
        ".cr3",
        ".arw",
        ".dng",
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    }
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in extensions
    )


def project_relative(path):
    """返回便于移动项目的相对路径。"""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def load_manifest(path=SYNTHETIC_MANIFEST, split: str | None = None):
    """读取正式合成数据 manifest，可按 split 过滤。"""
    path = to_absolute_path(path)
    filtered_rows = []
    if not path.is_file():
        raise FileNotFoundError(f"合成数据 manifest 不存在：{path}")

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if split is not None:
        for row in rows:
            if row["split"] == split:
                filtered_rows.append(row)
        rows = filtered_rows

    return rows


__all__ = [
    "PROJECT_ROOT",
    "BASE_OUTPUT_ROOT",
    "SPLIT_MODE",
    "CV_FOLD",
    "OUTPUT_ROOT",
    "OUTPUT_NAMESPACE",
    "DATA_ROOT",
    "COLLECTION_ROOT",
    "COLLECTION_MANIFEST",
    "TEST_ROOT",
    "NATIVE_TEST_IMAGE",
    "REAL_HIGH_POLLUTION_ROOT",
    "LEGACY_SYNTHETIC_ROOT",
    "SYNTHETIC_DATA_ROOT",
    "SYNTHETIC_ROOT",
    "SYNTHETIC_MANIFEST",
    "SELECTION_ROOT",
    "PINN_SELECTION_ROOT",
    "GAUSSIAN_SELECTION_ROOT",
    "FOURIER_LS_SELECTION_ROOT",
    "PINN_SELECTION_ARTIFACT",
    "GAUSSIAN_SELECTION_ARTIFACT",
    "FOURIER_LS_SELECTION_ARTIFACT",
    "GLOBAL_CONFIGURATION_LOCK",
    "SEEDS",
    "to_absolute_path",
    "resolve_collection_path",
    "supported_image_files",
    "project_relative",
    "load_manifest",
]
