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


def _data_layout_under(directory: Path, *, names=("collections",)) -> Path | None:
    """Return the data root implied by ``directory``, or None.

    Two layouts are recognised: ``directory/<name>`` (current) and
    ``directory/data/<name>`` (legacy).
    """
    for root in (directory, directory / "data"):
        if any((root / name).is_dir() for name in names):
            return root
    return None


def _search_upward(start: Path, *, names=("collections",)) -> Path | None:
    start = Path(start).resolve()
    for directory in (start, *start.parents):
        found = _data_layout_under(directory, names=names)
        if found is not None:
            return found
    return None


def _resolve_data_root_with_source() -> tuple[Path, str]:
    """Resolve the external data root without changing the experiment protocol.

    Priority (the CLI's ``--data-root`` is applied by exporting variable 1):
    1. ``PINN_STARLIGHT_DATA_ROOT``;
    2. nearest ancestor of the current working directory with ``collections/``;
    3. nearest ancestor of this package with ``collections/`` or ``test/``
       (the source checkout; an installed wheel finds nothing here);
    4. ``PROJECT_ROOT`` as a placeholder, so importing stays side-effect free.
       Callers that need data raise an actionable error via ``data_root_hint``.

    Returns ``(path, source)`` where source is ``env``, ``cwd``, ``package`` or
    ``default``.
    """
    configured = os.environ.get("PINN_STARLIGHT_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve(), "env"

    try:
        from_cwd = _search_upward(Path.cwd())
    except OSError:  # cwd was deleted
        from_cwd = None
    if from_cwd is not None:
        return from_cwd, "cwd"

    from_package = _search_upward(Path(__file__).parent, names=("collections", "test"))
    if from_package is not None:
        return from_package, "package"
    return PROJECT_ROOT, "default"


DATA_ROOT, DATA_ROOT_SOURCE = _resolve_data_root_with_source()
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


def data_root_hint() -> str:
    """Actionable next steps when the synthetic datasets cannot be found."""
    return (
        f"data root is {DATA_ROOT} (source: {DATA_ROOT_SOURCE}) and it holds no dataset. "
        "Point the tool at a directory that contains "
        "collections/synthetic_formal/manifest.csv:\n"
        "  pinn-starlight <command> --data-root /path/to/data\n"
        "  export PINN_STARLIGHT_DATA_ROOT=/path/to/data\n"
        "  or run from the repository checkout after `bash tools/fetch_smoke_data.sh`\n"
        "`estimate --image PHOTO` needs no data root."
    )


def to_absolute_path(path):
    """把 manifest 中的项目相对路径转换为绝对路径。

    ``collections/...`` paths follow ``DATA_ROOT`` when it differs from the
    project root (installed wheel, ``--data-root``); in a checkout the two
    coincide and the result is unchanged.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    if DATA_ROOT != PROJECT_ROOT and (
        path.parts[:1] == ("collections",) or path.parts[:2] == ("data", "collections")
    ):
        return resolve_collection_path(path)
    return PROJECT_ROOT / path


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
    "DATA_ROOT_SOURCE",
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
    "data_root_hint",
    "to_absolute_path",
    "resolve_collection_path",
    "supported_image_files",
    "project_relative",
    "load_manifest",
]
