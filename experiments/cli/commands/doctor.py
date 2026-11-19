"""``pinn-starlight doctor`` - report environment and data readiness.

This command must stay importable *without* numpy / torch / tifffile: it is the
first thing a user runs, and on a machine where those are missing it has to
report the gap instead of crashing.  Layout facts therefore come from
``experiments.common.utils.paths`` (standard library only).
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

from experiments.cli.context import (
    EXIT_ENV,
    EXIT_OK,
    TOOL_NAME,
    apply_data_root,
    base_payload,
    print_json,
)

REQUIRED_PACKAGES = (
    "torch",
    "numpy",
    "scipy",
    "scikit-image",
    "photutils",
    "tifffile",
    "matplotlib",
)
OPTIONAL_PACKAGES = ("rawpy", "tqdm")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nef", ".cr2", ".dng", ".arw"}


def _check_packages() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for name in REQUIRED_PACKAGES + OPTIONAL_PACKAGES:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _count_images(directory) -> int:
    path = Path(directory)
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.suffix.lower() in IMAGE_SUFFIXES)


def _torch_info() -> dict:
    try:
        import torch

        return {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
        }
    except Exception as exc:  # noqa: BLE001 - doctor must report, not crash
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def cmd_doctor(args) -> int:
    apply_data_root(args)

    from experiments.common.utils import paths

    warnings: list[str] = []
    python_ok = sys.version_info >= (3, 11)
    if not python_ok:
        warnings.append("Python >= 3.11 is required by the formal pipeline")

    packages = _check_packages()
    missing = [name for name, version in packages.items() if version is None]

    torch_info = _torch_info()
    if not torch_info.get("available"):
        warnings.append(f"torch is not importable: {torch_info.get('error', 'unknown error')}")
        warnings.append("estimate needs the estimator extras; doctor/samples do not")
    elif not torch_info.get("cuda_available"):
        warnings.append("CUDA is not available; formal Stage-A/B/CV5/E4 cannot run here")

    data = {
        "root": str(paths.DATA_ROOT),
        "root_source": paths.DATA_ROOT_SOURCE,
        "collections_manifest": Path(paths.COLLECTION_MANIFEST).is_file(),
        "stage_a_manifest": Path(paths.SYNTHETIC_DATA_ROOT / "manifest.csv").is_file(),
        "stage_b_manifest": Path(
            paths.DATA_ROOT / "collections/synthetic_stage_b/manifest.csv"
        ).is_file(),
        "stage_a_lock": Path(paths.GLOBAL_CONFIGURATION_LOCK).is_file(),
        "stage_b_lock": Path(
            paths.BASE_OUTPUT_ROOT
            / "stage_b/configuration_lock/stage_b_global_configuration_lock.json"
        ).is_file(),
        "real_high_pollution_images": _count_images(paths.REAL_HIGH_POLLUTION_ROOT),
        "native_test_image": Path(paths.NATIVE_TEST_IMAGE).is_file(),
    }
    for key in ("stage_a_manifest", "stage_b_manifest"):
        if not data[key]:
            warnings.append(f"{key} is missing; --sample runs need the extracted dataset")
    if not (data["stage_a_manifest"] or data["stage_b_manifest"]):
        warnings.append(paths.data_root_hint())

    output_root = Path(paths.BASE_OUTPUT_ROOT)
    output_writable = os.access(
        output_root if output_root.exists() else output_root.parent, os.W_OK
    )
    if not output_writable:
        warnings.append(f"output root is not writable: {paths.project_relative(output_root)}")

    payload = {
        **base_payload("doctor"),
        "ok": bool(python_ok and torch_info.get("available")),
        "python": {
            "version": platform.python_version(),
            "ok": python_ok,
            "executable": sys.executable,
        },
        "packages": packages,
        "missing_packages": missing,
        "torch": torch_info,
        "experiments_importable": True,
        "data": data,
        "output_writable": bool(output_writable),
        "warnings": warnings,
    }
    if args.json:
        print_json(payload)
    else:
        print(f"{TOOL_NAME} doctor")
        print(
            f"  python      : {payload['python']['version']} ({'ok' if python_ok else 'too old'})"
        )
        print(f"  torch       : {torch_info.get('version', 'missing')}")
        print(f"  CUDA        : {torch_info.get('cuda_available')}")
        print(f"  data root   : {data['root']} ({data['root_source']})")
        print(f"  Stage-A     : {data['stage_a_manifest']}")
        print(f"  Stage-B     : {data['stage_b_manifest']}")
        print(f"  outputs     : {'writable' if output_writable else 'not writable'}")
        for warning in warnings:
            print(f"  warning     : {warning}".replace("\n", "\n                "))
    return EXIT_OK if payload["ok"] else EXIT_ENV
