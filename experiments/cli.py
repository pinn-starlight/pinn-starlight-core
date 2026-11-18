"""``pinn-starlight`` command line (release entry point).

Examples
--------
    python -m experiments.cli version
    python -m experiments.cli doctor --json
    python -m experiments.cli samples --stage a
    python -m experiments.cli estimate --image photo.jpg --mode star
    python -m experiments.cli estimate --sample clean_single_002_multi_source_high

The CLI runs the same demo budgets as ``tools/code_check_demo.py``: it is a
release/demonstration tool, not the formal Stage-A/B/CV5/E4 pipeline.  Values
printed for a photograph are descriptive only; values for a synthetic sample
use the frozen star reference but still at reduced budgets.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

from experiments.configs.demo_methods import (
    DISPLAY_NAMES,
    METHOD_NAMES,
    MODE_TO_METHOD,
    PAPER_METHODS,
)
from experiments.configs.formal_experiment import FORMAL_SEEDS

TOOL_NAME = "pinn-starlight"
SCHEMA_VERSION = 1

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_ENV = 3
EXIT_INPUT = 4
EXIT_METHOD = 5

METHOD_CHOICES = METHOD_NAMES


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _version() -> str:
    try:
        from importlib.metadata import version

        return version("pinn-starlight-core")
    except Exception:  # noqa: BLE001 - source checkout without an installed dist
        from pinn_starlight_core import __version__

        return __version__


def _json_safe(value):
    import math

    import numpy as np

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _print_json(payload: dict) -> None:
    print(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False))


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


# --------------------------------------------------------------------------
# version
# --------------------------------------------------------------------------
def cmd_version(args) -> int:
    from pinn_starlight_core import __version__

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "version": _version(),
        "core_version": __version__,
        "python": platform.python_version(),
    }
    if args.json:
        _print_json(payload)
    else:
        print(f"{TOOL_NAME} {payload['version']} (core {payload['core_version']})")
    return EXIT_OK


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------
def _check_packages() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    required = ("torch", "numpy", "scipy", "scikit-image", "photutils", "tifffile", "matplotlib")
    optional = ("rawpy", "tqdm")
    result = {}
    for name in required + optional:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def cmd_doctor(args) -> int:
    if args.data_root:
        os.environ["PINN_STARLIGHT_DATA_ROOT"] = str(Path(args.data_root).expanduser())

    from experiments.common.utils import experiment_utils as utils

    warnings: list[str] = []
    python_ok = sys.version_info >= (3, 11)
    if not python_ok:
        warnings.append("Python >= 3.11 is required by the formal pipeline")

    packages = _check_packages()
    missing = [name for name, version in packages.items() if version is None]
    torch_info: dict = {"available": False}
    try:
        import torch

        torch_info = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "device_count": int(torch.cuda.device_count()),
        }
        if not torch.cuda.is_available():
            warnings.append("CUDA is not available; formal Stage-A/B/CV5/E4 cannot run here")
    except Exception as exc:  # noqa: BLE001 - doctor must report, not crash
        warnings.append(f"torch import failed: {type(exc).__name__}: {exc}")

    data = {
        "root": utils.project_relative(utils.DATA_ROOT),
        "collections_manifest": Path(utils.COLLECTION_MANIFEST).is_file(),
        "stage_a_manifest": Path(utils.SYNTHETIC_DATA_ROOT / "manifest.csv").is_file(),
        "stage_b_manifest": Path(
            utils.DATA_ROOT / "collections/synthetic_stage_b/manifest.csv"
        ).is_file(),
        "stage_a_lock": Path(utils.GLOBAL_CONFIGURATION_LOCK).is_file(),
        "stage_b_lock": Path(
            utils.BASE_OUTPUT_ROOT
            / "stage_b/configuration_lock/stage_b_global_configuration_lock.json"
        ).is_file(),
        "real_high_pollution_images": _count_images(utils.REAL_HIGH_POLLUTION_ROOT),
        "native_test_image": Path(utils.NATIVE_TEST_IMAGE).is_file(),
    }
    for key in ("stage_a_manifest", "stage_b_manifest"):
        if not data[key]:
            warnings.append(f"{key} is missing; --sample runs need the extracted dataset")

    output_root = Path(utils.BASE_OUTPUT_ROOT)
    output_writable = os.access(
        output_root if output_root.exists() else output_root.parent, os.W_OK
    )
    if not output_writable:
        warnings.append(f"output root is not writable: {utils.project_relative(output_root)}")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "version": _version(),
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
        _print_json(payload)
    else:
        print(f"{TOOL_NAME} doctor")
        print(
            f"  python      : {payload['python']['version']} ({'ok' if python_ok else 'too old'})"
        )
        print(f"  torch       : {torch_info.get('version', 'missing')}")
        print(f"  CUDA        : {torch_info.get('cuda_available')}")
        print(f"  data root   : {data['root']}")
        print(f"  Stage-A     : {data['stage_a_manifest']}")
        print(f"  Stage-B     : {data['stage_b_manifest']}")
        print(f"  outputs     : {'writable' if output_writable else 'not writable'}")
        for warning in warnings:
            print(f"  warning     : {warning}")
    return EXIT_OK if payload["ok"] else EXIT_ENV


def _count_images(directory) -> int:
    suffixes = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".nef", ".cr2", ".dng", ".arw"}
    path = Path(directory)
    if not path.is_dir():
        return 0
    return sum(1 for item in path.iterdir() if item.suffix.lower() in suffixes)


# --------------------------------------------------------------------------
# samples
# --------------------------------------------------------------------------
def cmd_samples(args) -> int:
    if args.data_root:
        os.environ["PINN_STARLIGHT_DATA_ROOT"] = str(Path(args.data_root).expanduser())
    from experiments.common.utils import experiment_utils as utils

    datasets = {}
    if args.stage in ("a", "all"):
        datasets["stage_a"] = _list_manifest(utils.SYNTHETIC_DATA_ROOT / "manifest.csv")
    if args.stage in ("b", "all"):
        datasets["stage_b"] = _list_manifest(
            utils.DATA_ROOT / "collections/synthetic_stage_b/manifest.csv"
        )
    payload = {"schema_version": SCHEMA_VERSION, "tool": TOOL_NAME, "datasets": datasets}
    if args.json:
        _print_json(payload)
    else:
        for name, rows in datasets.items():
            print(f"{name}: {len(rows)} validation rows")
            for row in rows:
                print(
                    f"  {row['sample_id']:<52s} {row.get('scene_type') or row.get('background_type')}"
                )
    return EXIT_OK


def _list_manifest(path) -> list[dict]:
    from experiments.common.utils import experiment_utils as utils

    if not Path(path).is_file():
        return []
    rows = utils.load_manifest(path, split="validation")
    return [
        {
            "sample_id": row["sample_id"],
            "scene_type": row.get("scene_type", row.get("background_type")),
            "intensity_level": row.get("intensity_level"),
        }
        for row in rows
    ]


# --------------------------------------------------------------------------
# estimate
# --------------------------------------------------------------------------
def _resolve_methods(args) -> tuple[list[str], list[str]]:
    warnings = []
    if args.mode:
        if args.methods != "all":
            raise ValueError("give either --mode or --methods, not both")
        return [MODE_TO_METHOD[args.mode]], warnings
    if args.methods == "all":
        return list(PAPER_METHODS), warnings
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHOD_CHOICES))
    if unknown:
        raise ValueError(f"unknown method(s): {unknown}; expected {METHOD_CHOICES} or all")
    if "balanced" in methods:
        warnings.append("balanced mode is experimental (physics-first init + anchored finetune)")
    return methods, warnings


def _render_preview(path: Path, observed, backgrounds: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - preview is optional
        return

    from experiments.common.utils import experiment_utils as utils

    columns = [("Observed", observed, None)] + [
        (DISPLAY_NAMES.get(name, name), background, None)
        for name, background in backgrounds.items()
    ]
    figure, axes = plt.subplots(2, len(columns), figsize=(3.0 * len(columns), 6.2))
    axes = axes.reshape(2, len(columns))
    vmin, vmax = float(observed.min()), float(observed.max())
    for col, (title, background, _) in enumerate(columns):
        axes[0, col].imshow(background, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(f"{title}\nbackground", fontsize=9)
        residual = observed if col == 0 else utils.reconstruct_residual(observed, background)
        axes[1, col].imshow(residual, cmap="gray", vmin=vmin, vmax=vmax)
        axes[1, col].set_title("observed" if col == 0 else "residual", fontsize=9)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(f"{TOOL_NAME} - CPU/demo budgets, not a paper result", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def cmd_estimate(args) -> int:
    if args.data_root:
        os.environ["PINN_STARLIGHT_DATA_ROOT"] = str(Path(args.data_root).expanduser())

    try:
        methods, warnings = _resolve_methods(args)
    except ValueError as exc:
        return _fail(str(exc), EXIT_USAGE)

    from experiments.common import pipeline
    from experiments.common.utils import experiment_utils as utils

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return _fail(f"torch import failed: {type(exc).__name__}: {exc}", EXIT_ENV)

    device_name = str(args.device).lower()
    if device_name in {"auto", ""}:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_name.startswith("cuda") and not torch.cuda.is_available():
        return _fail("CUDA requested but not available; use --device cpu", EXIT_ENV)
    else:
        device = torch.device(device_name)

    try:
        if args.image:
            sample = pipeline.load_image_array(args.image, max_side=args.max_side)
        else:
            sample = pipeline.load_dataset_sample(
                args.sample,
                max_side=args.max_side,
                dataset=args.dataset,
            )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return _fail(str(exc), EXIT_INPUT)

    observed = sample["observed"]
    out_root = Path(args.out) if args.out else Path.cwd() / "pinn_starlight_output"
    run_dir = out_root / sample["sample_id"]
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _fail(f"cannot create output directory {run_dir}: {exc}", EXIT_ENV)

    results: dict[str, dict] = {}
    backgrounds: dict[str, object] = {}
    for method in methods:
        try:
            run = pipeline.run_methods(
                observed,
                steps=args.steps,
                device=device,
                methods=[method],
                seed=args.seed,
                show_progress=not args.quiet,
            )
            backgrounds[method] = run["backgrounds"][method]
            results[method] = {
                "display": DISPLAY_NAMES.get(method, method),
                "runtime_s": float(run["timings"][method]),
            }
        except Exception as exc:  # noqa: BLE001 - one method must not kill the CLI
            results[method] = {
                "display": DISPLAY_NAMES.get(method, method),
                "error": f"{type(exc).__name__}: {exc}",
            }
            warnings.append(f"{method} failed: {type(exc).__name__}: {exc}")

    if not backgrounds:
        return _fail("every requested method failed; see warnings", EXIT_METHOD)

    if sample.get("has_truth"):
        try:
            table = pipeline.evaluate_synthetic(sample, backgrounds)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"truth metrics failed: {type(exc).__name__}: {exc}")
            table = {}
        for name, row in table.items():
            if sample["downsample"] > 1:
                # The frozen star reference is defined on the full-resolution grid.
                for key in (
                    "star_f1",
                    "matched_star_mean_relative_flux_error",
                    "faint_star_recall",
                ):
                    if key in row:
                        row[key] = None
            results.setdefault(name, {})["metrics"] = row
        if sample["downsample"] > 1:
            warnings.append(
                f"star metrics are not comparable at downsample x{sample['downsample']}"
            )
    else:
        from experiments.common.utils import metrics

        for name, background in backgrounds.items():
            residual = utils.reconstruct_residual(observed, background)
            results[name]["metrics"] = metrics.evaluate_real(observed, background, residual)
        warnings.append("no ground truth: only descriptive residual metrics are reported")

    utils.save_float_tiff(run_dir / "observed.tif", observed)
    for name, background in backgrounds.items():
        method_dir = run_dir / name
        residual = utils.reconstruct_residual(observed, background)
        utils.save_float_tiff(method_dir / "background.tif", background)
        utils.save_float_tiff(method_dir / "residual.tif", residual)
        utils.save_display_png(method_dir / "background.png", background)
        utils.save_display_png(method_dir / "residual.png", residual)

    preview = run_dir / "preview.png"
    if not args.no_figure:
        _render_preview(preview, observed, backgrounds)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": TOOL_NAME, "version": _version()},
        "command": "estimate",
        "input": {
            "kind": "image" if args.image else "sample",
            "path": sample.get("source"),
            "sample_id": sample["sample_id"],
            "dataset": sample.get("dataset"),
            "shape": list(observed.shape),
            "downsample": int(sample["downsample"]),
            "has_truth": bool(sample.get("has_truth")),
        },
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "methods": results,
        "outputs": {
            "root": str(run_dir),
            "observed": str(run_dir / "observed.tif"),
            "preview": str(preview) if not args.no_figure else None,
        },
        "warnings": warnings,
        "note": "CPU/demo budgets; not a paper result",
    }
    (run_dir / "summary.json").write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if args.json:
        _print_json(payload)
    elif not args.quiet:
        print(f"{TOOL_NAME} estimate -> {run_dir}")
        for item in results.values():
            if "error" in item:
                print(f"  {item['display']:<18s} ERROR {item['error']}")
                continue
            metric = item.get("metrics", {})
            detail = ""
            if "bg_mae" in metric and metric["bg_mae"] is not None:
                detail = f"bg_mae={metric['bg_mae']:.6g}"
            elif "residual_std" in metric:
                detail = f"residual_std={metric['residual_std']:.6g}"
            print(f"  {item['display']:<18s} {item['runtime_s']:7.2f}s  {detail}")
        for warning in warnings:
            print(f"  warning: {warning}")
        print(f"  summary: {run_dir / 'summary.json'}")
    return EXIT_OK


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="PINN-Starlight demo CLI (not the formal experiment pipeline).",
    )
    parser.add_argument("--version", action="version", version=f"{TOOL_NAME} {_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    version = subparsers.add_parser("version", help="print the tool and core versions")
    version.add_argument("--json", action="store_true")
    version.set_defaults(func=cmd_version)

    doctor = subparsers.add_parser("doctor", help="report environment and data readiness")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--data-root", type=Path, default=None)
    doctor.set_defaults(func=cmd_doctor)

    samples = subparsers.add_parser("samples", help="list validation samples on disk")
    samples.add_argument("--stage", choices=("a", "b", "all"), default="all")
    samples.add_argument("--json", action="store_true")
    samples.add_argument("--data-root", type=Path, default=None)
    samples.set_defaults(func=cmd_samples)

    estimate = subparsers.add_parser("estimate", help="estimate the background of one image")
    source = estimate.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="any supported image file")
    source.add_argument("--sample", help="validation sample_id from the extracted dataset")
    estimate.add_argument("--dataset", choices=("stage_a", "stage_b"), default="stage_a")
    estimate.add_argument(
        "--methods",
        default="all",
        help="all or comma separated fft_gaussian,fourier_ls,pinn,final_pinn,balanced",
    )
    estimate.add_argument(
        "--mode",
        choices=tuple(MODE_TO_METHOD),
        default=None,
        help="shortcut: star=final_pinn, nebula=pinn, balanced=anchored",
    )
    estimate.add_argument("--max-side", type=int, default=384)
    estimate.add_argument("--steps", type=int, default=500)
    estimate.add_argument("--seed", type=int, default=FORMAL_SEEDS[0])
    estimate.add_argument("--device", default="auto")
    estimate.add_argument("--out", type=Path, default=None)
    estimate.add_argument("--json", action="store_true")
    estimate.add_argument("--no-figure", action="store_true")
    estimate.add_argument("--quiet", action="store_true")
    estimate.add_argument("--data-root", type=Path, default=None)
    estimate.set_defaults(func=cmd_estimate)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
