"""``pinn-starlight estimate`` - estimate the background of one image."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.cli.context import (
    EXIT_ENV,
    EXIT_INPUT,
    EXIT_METHOD,
    EXIT_OK,
    EXIT_USAGE,
    METHOD_CHOICES,
    TOOL_NAME,
    apply_data_root,
    base_payload,
    fail,
    json_safe,
    print_json,
)
from experiments.cli.render import render_preview
from experiments.configs.demo_methods import DISPLAY_NAMES, MODE_TO_METHOD, PAPER_METHODS


def resolve_methods(args) -> tuple[list[str], list[str]]:
    warnings = []
    if args.mode:
        if args.methods is not None:
            raise ValueError("give either --mode or --methods, not both")
        modes = [item.strip() for item in args.mode.split(",") if item.strip()]
        unknown = [item for item in modes if item not in MODE_TO_METHOD]
        if unknown or not modes:
            raise ValueError(
                f"unknown mode(s): {unknown or args.mode!r}; "
                f"expected a comma separated subset of {','.join(MODE_TO_METHOD)}"
            )
        methods = list(dict.fromkeys(MODE_TO_METHOD[item] for item in modes))
        if "balanced" in methods:
            warnings.append(
                "balanced mode is experimental (physics-first init + anchored finetune)"
            )
        return methods, warnings
    if args.methods in (None, "all"):
        return list(PAPER_METHODS), warnings
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(METHOD_CHOICES))
    if unknown:
        raise ValueError(f"unknown method(s): {unknown}; expected {METHOD_CHOICES} or all")
    if "balanced" in methods:
        warnings.append("balanced mode is experimental (physics-first init + anchored finetune)")
    return methods, warnings


def _run_requested(sample, methods, args, device) -> tuple[dict, dict, list[str]]:
    from experiments.common import pipeline

    results: dict[str, dict] = {}
    backgrounds: dict[str, object] = {}
    warnings: list[str] = []
    for method in methods:
        try:
            run = pipeline.run_methods(
                sample["observed"],
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
    return results, backgrounds, warnings


def _attach_metrics(sample, observed, backgrounds, results, warnings) -> None:
    from experiments.common import pipeline
    from experiments.common.utils import experiment_utils as utils

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
        return

    from experiments.common.utils import metrics

    for name, background in backgrounds.items():
        residual = utils.reconstruct_residual(observed, background)
        results[name]["metrics"] = metrics.evaluate_real(observed, background, residual)
    warnings.append("no ground truth: only descriptive residual metrics are reported")


def _write_outputs(run_dir: Path, observed, backgrounds, args) -> Path:
    from experiments.common.utils import experiment_utils as utils

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
        render_preview(preview, observed, backgrounds)
    return preview


def cmd_estimate(args) -> int:
    apply_data_root(args)

    if args.steps < 1 or args.max_side < 8:
        return fail("--steps must be >= 1 and --max-side >= 8", EXIT_USAGE)

    try:
        methods, method_warnings = resolve_methods(args)
    except ValueError as exc:
        return fail(str(exc), EXIT_USAGE)

    try:
        import torch

        from experiments.common import pipeline
    except Exception as exc:  # noqa: BLE001
        return fail(
            f"estimator dependencies are missing: {type(exc).__name__}: {exc}; "
            "install them with pip install 'pinn-starlight-core[estimator,cpu-torch]'",
            EXIT_ENV,
        )

    device_name = str(args.device).lower()
    if device_name in {"auto", ""}:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_name.startswith("cuda") and not torch.cuda.is_available():
        return fail("CUDA requested but not available; use --device cpu", EXIT_ENV)
    else:
        try:
            device = torch.device(device_name)
        except (RuntimeError, ValueError):
            return fail(f"invalid --device {args.device!r}; use auto, cpu or cuda:N", EXIT_USAGE)

    try:
        if args.image:
            sample = pipeline.load_image_array(args.image, max_side=args.max_side)
        else:
            sample = pipeline.load_dataset_sample(
                args.sample,
                max_side=args.max_side,
                dataset=args.dataset,
            )
    except (FileNotFoundError, KeyError, ValueError, OSError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return fail(str(message), EXIT_INPUT)

    observed = sample["observed"]
    out_root = Path(args.out) if args.out else Path.cwd() / "pinn_starlight_output"
    run_dir = out_root / sample["sample_id"]
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return fail(f"cannot create output directory {run_dir}: {exc}", EXIT_ENV)

    results, backgrounds, warnings = _run_requested(sample, methods, args, device)
    warnings = method_warnings + warnings
    if not backgrounds:
        detail = "; ".join(warnings)
        return fail(f"every requested method failed: {detail}", EXIT_METHOD)

    _attach_metrics(sample, observed, backgrounds, results, warnings)
    preview = _write_outputs(run_dir, observed, backgrounds, args)

    payload = {
        **base_payload("estimate"),
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
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    if args.json:
        print_json(payload)
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
