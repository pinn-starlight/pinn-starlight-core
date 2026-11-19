"""Initialization-memory probe (non-formal).

Question
--------
Does the physics-first initialization advantage in ``protected_projection_ratio``
survive the later coordinated finetune, and can an explicit anchor to the
initialization offset the erosion?

Design
------
Stage-B **validation** only, one fixed seed, two initializations
(``data_first`` / ``physics_first``) and four finetune arms
(``none`` / ``data_only`` / ``sgc`` / ``anchored(weight)``).  The anchored arm
adds ``weight * mean((B - B_init)^2)`` to the data side before the physics
gradient is projected away, using the unmodified Stage-2 forcing fit and the
same Stage-3 budget.

This module never opens the test split and never writes into a formal output
directory.  It is a mechanism probe, not a paper result.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from experiments.common.baselines import coordinate_pinn as base_pinn
from experiments.common.baselines import sgc_pinn
from experiments.common.baselines.anchored_finetune import finetune_anchored
from experiments.common.utils import experiment_utils as utils
from experiments.configs import diagnostics as diag
from experiments.scripts.diagnostics import _harness as harness

PROBE = "init_memory"
DATASET = "stage_b"
VALUE_KEYS = (
    "bg_mae",
    "bg_ssim",
    "residual_ssim",
    "projection_ratio",
    "projection_abs_error",
    "protected_region_mae",
    "faint_star_recall",
    "star_f1",
    "matched_star_flux_error",
    "runtime_s",
)


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------
def _cell_row(
    *,
    sample: dict,
    suite: str,
    init: str,
    arm: str,
    anchor_weight: float,
    steps: int,
    background,
    runtime_s: float,
    curve_step: int | None = None,
    extra: dict | None = None,
) -> dict:
    evaluated = harness.evaluate(sample, background)
    row = {
        "probe": PROBE,
        "suite": suite,
        "sample_id": sample["sample_id"],
        "scene_type": sample.get("scene_type"),
        "intensity_level": sample.get("intensity_level"),
        "stride": sample["stride"],
        "init": init,
        "arm": arm,
        "anchor_weight": float(anchor_weight),
        "steps": int(steps),
        "curve_step": curve_step,
        "bg_mae": evaluated.get("bg_mae"),
        "bg_ssim": evaluated.get("bg_ssim"),
        "residual_ssim": evaluated.get("residual_ssim"),
        "projection_ratio": evaluated.get("protected_projection_ratio"),
        "projection_abs_error": evaluated.get("protected_projection_abs_error"),
        "protected_region_mae": evaluated.get("protected_region_mae"),
        "faint_star_recall": evaluated.get("faint_star_recall"),
        "star_f1": evaluated.get("star_f1"),
        "matched_star_flux_error": evaluated.get("matched_star_mean_relative_flux_error"),
        "star_metrics_skipped": bool(evaluated.get("star_metrics_skipped", False)),
        "runtime_s": float(runtime_s),
    }
    row.update(extra or {})
    return row


def _train_init(
    sample: dict,
    *,
    suite: str,
    init: str,
    device,
    seed: int,
    show_progress: bool,
    steps: int | None = None,
):
    config = base_pinn.normalized_config(diag.stage1_config(suite=suite, init=init, steps=steps))
    centers = base_pinn.initial_centers_from_observed(sample["observed"], config)
    started = time.perf_counter()
    result = base_pinn.train_background(
        sample["observed"],
        config=config,
        device=device,
        seed=seed,
        show_progress=show_progress,
        return_state=True,
        laplacian_backend="analytic",
        initial_centers=centers,
    )
    return config, centers, result, time.perf_counter() - started


def _fit_forcing(
    sample: dict, init_result: dict, config: dict, centers, *, suite: str, device, seed: int
):
    forcing_state, diagnostics = sgc_pinn.fit_fixed_forcing(
        observed=sample["observed"],
        model_state=init_result["state"]["model_state"],
        stage1_config=config,
        stage2_config=diag.stage2_config(suite=suite),
        initial_centers=centers,
        device=device,
        seed=int(seed) + 1009,
    )
    return forcing_state, diagnostics


def _run_finetune(
    sample: dict,
    init_result: dict,
    config: dict,
    centers,
    forcing_state,
    *,
    arm: str,
    suite: str,
    anchor_weight: float,
    steps: int,
    device,
    seed: int,
    show_progress: bool,
):
    stage3 = diag.stage3_config(suite=suite, steps=steps)
    if arm == "data_only":
        return (
            sgc_pinn.finetune(
                observed=sample["observed"],
                model_state=init_result["state"]["model_state"],
                forcing_state=None,
                stage1_config=config,
                stage3_config=stage3,
                initial_centers=centers,
                device=device,
                seed=int(seed) + 2003,
                use_physics=False,
                show_progress=show_progress,
            ),
            None,
        )
    if arm == "sgc":
        return (
            sgc_pinn.finetune(
                observed=sample["observed"],
                model_state=init_result["state"]["model_state"],
                forcing_state=forcing_state,
                stage1_config=config,
                stage3_config=stage3,
                initial_centers=centers,
                device=device,
                seed=int(seed) + 2003,
                use_physics=True,
                show_progress=show_progress,
            ),
            None,
        )
    if arm == "anchored":
        return (
            finetune_anchored(
                observed=sample["observed"],
                model_state=init_result["state"]["model_state"],
                forcing_state=forcing_state,
                stage1_config=config,
                stage3_config=stage3,
                initial_centers=centers,
                device=device,
                seed=int(seed) + 2003,
                anchor_background=init_result["background_pred"],
                anchor_weight=anchor_weight,
                show_progress=show_progress,
            ),
            None,
        )
    raise ValueError(f"unknown finetune arm: {arm!r}")


# --------------------------------------------------------------------------
# attribution
# --------------------------------------------------------------------------
def _finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def attribution_rows(cell_rows: list[dict]) -> list[dict]:
    """Decompose the projection change into init / refinement / interaction."""
    lookup = {
        (row["sample_id"], row["init"], row["arm"]): row
        for row in cell_rows
        if row.get("curve_step") is None
    }
    sample_ids = sorted({row["sample_id"] for row in cell_rows})
    arm_names = sorted({row["arm"] for row in cell_rows if row["arm"] != "none"})
    output = []
    for sample_id in sample_ids:
        data_init = _finite(
            lookup.get((sample_id, "data_first", "none"), {}).get("projection_ratio")
        )
        phys_init = _finite(
            lookup.get((sample_id, "physics_first", "none"), {}).get("projection_ratio")
        )
        init_effect = phys_init - data_init
        for arm in arm_names:
            data_final = _finite(
                lookup.get((sample_id, "data_first", arm), {}).get("projection_ratio")
            )
            phys_final = _finite(
                lookup.get((sample_id, "physics_first", arm), {}).get("projection_ratio")
            )
            refine_data = data_final - data_init
            refine_phys = phys_final - phys_init
            retention = (
                (phys_final - data_final) / init_effect
                if np.isfinite(init_effect) and abs(init_effect) > 1e-12
                else float("nan")
            )
            output.append(
                {
                    "probe": PROBE,
                    "sample_id": sample_id,
                    "arm": arm,
                    "data_init_projection": data_init,
                    "physics_init_projection": phys_init,
                    "init_effect": init_effect,
                    "refine_effect_data": refine_data,
                    "refine_effect_physics": refine_phys,
                    "interaction": refine_phys - refine_data,
                    "retention": retention,
                    "bg_mae_data_final": _finite(
                        lookup.get((sample_id, "data_first", arm), {}).get("bg_mae")
                    ),
                    "bg_mae_physics_final": _finite(
                        lookup.get((sample_id, "physics_first", arm), {}).get("bg_mae")
                    ),
                }
            )
    return output


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _fmt(value) -> str:
    number = _finite(value)
    return "n/a" if not np.isfinite(number) else f"{number:.4f}"


def _markdown(summary: dict, attribution_summary: list[dict], suite: str) -> str:
    lines = [
        f"# init_memory · {suite}",
        "",
        "> Non-formal mechanism probe. Validation split only. "
        + (
            "Smoke budgets: plumbing check, **not** a scientific result."
            if suite == "smoke"
            else "Declared diagnostic budget; still not a paper result."
        ),
        "",
        "## Projection by init / arm",
        "",
        "| init | arm | mean projection | mean bg MAE | n |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary.get("cells", []):
        lines.append(
            f"| {row['init']} | {row['arm']} | {_fmt(row.get('projection_ratio_mean'))} "
            f"| {_fmt(row.get('bg_mae_mean'))} | {row.get('count')} |"
        )
    lines += [
        "",
        "## Init / refinement attribution",
        "",
        "| arm | init effect | refine (data-first) | refine (physics-first) | interaction | retention |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in attribution_summary:
        lines.append(
            f"| {row['arm']} | {_fmt(row.get('init_effect_mean'))} "
            f"| {_fmt(row.get('refine_effect_data_mean'))} "
            f"| {_fmt(row.get('refine_effect_physics_mean'))} "
            f"| {_fmt(row.get('interaction_mean'))} "
            f"| {_fmt(row.get('retention_mean'))} |"
        )
    lines += ["", "## Observations", ""]
    lines.extend(f"- {item}" for item in summary.get("observations", []))
    lines.append("")
    return "\n".join(lines)


def _observations(summary: dict) -> list[str]:
    cells = summary.get("cells", [])
    attribution = summary.get("attribution", [])

    def cell(init: str, arm: str):
        return next((row for row in cells if row["init"] == init and row["arm"] == arm), None)

    init_row = cell("data_first", "none")
    phys_row = cell("physics_first", "none")
    if init_row is None or phys_row is None:
        return ["initialization cells missing"]
    observations = [
        "initialization: data-first projection "
        f"{_fmt(init_row.get('projection_ratio_mean'))}, physics-first "
        f"{_fmt(phys_row.get('projection_ratio_mean'))}",
    ]
    for row in attribution:
        observations.append(
            f"{row['arm']}: retention {_fmt(row.get('retention_mean'))}, "
            f"interaction {_fmt(row.get('interaction_mean'))}, "
            f"physics-first bg MAE {_fmt(row.get('bg_mae_physics_final_mean'))}"
        )
    return observations


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=diag.SUITES, default="smoke")
    parser.add_argument("--samples", default="", help="comma separated validation sample_id")
    parser.add_argument("--max-side", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=diag.DIAGNOSTIC_SEED)
    parser.add_argument("--init-steps", type=int, default=None)
    parser.add_argument("--finetune-steps", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--inits", default=",".join(diag.INIT_NAMES))
    parser.add_argument("--finetunes", default="none,data_only,sgc,anchored")
    parser.add_argument("--anchor-weights", default=",".join(str(v) for v in diag.ANCHOR_WEIGHTS))
    parser.add_argument("--curves", action="store_true", help="run truncated step curves")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    harness.require_standard_split()
    device = harness.resolve_device(args.device)
    suite = args.suite
    budget = diag.BUDGETS[suite]
    max_side = args.max_side if args.max_side is not None else budget["max_side"]

    rows = harness.load_rows(DATASET, args.samples or None)
    limit = args.max_samples or int(budget["max_samples"])
    rows = harness.require_available(
        rows, DATASET, allow_subset=(suite == "smoke" and not args.samples), limit=limit
    )
    samples = [harness.load_sample(row, DATASET, max_side) for row in rows]

    inits = [item.strip() for item in args.inits.split(",") if item.strip()]
    finetunes = [item.strip() for item in args.finetunes.split(",") if item.strip()]
    unknown = sorted(set(inits) - set(diag.INIT_NAMES))
    if unknown:
        raise ValueError(f"unknown init(s): {unknown}")
    unknown = sorted(set(finetunes) - set(diag.FINETUNE_ARMS))
    if unknown:
        raise ValueError(f"unknown finetune arm(s): {unknown}")
    weights = [float(item) for item in args.anchor_weights.split(",") if item.strip()]

    run_dir = harness.new_run_dir(PROBE, suite, args.out)
    budgets = {
        "init_steps": int(args.init_steps or budget["init_steps"]),
        "finetune_steps": int(args.finetune_steps or budget["finetune_steps"]),
        "curve_steps": list(budget["curve_steps"]) if args.curves else [],
        "anchor_weights": weights,
    }
    harness.write_json(
        run_dir / "run_metadata.json",
        harness.run_metadata(
            probe=PROBE,
            suite=suite,
            dataset=DATASET,
            device=device,
            seed=args.seed,
            budgets=budgets,
            samples=samples,
            args=vars(args),
            extra={"inits": inits, "finetunes": finetunes},
        ),
    )

    cell_rows: list[dict] = []
    for sample in samples:
        print(f"[init_memory] {sample['sample_id']} stride={sample['stride']}", flush=True)
        for init in inits:
            config, centers, init_result, init_seconds = _train_init(
                sample,
                suite=suite,
                init=init,
                device=device,
                seed=args.seed,
                show_progress=not args.no_progress,
                steps=budgets["init_steps"],
            )
            print(
                f"  [{sample['sample_id']}] init={init} {budgets['init_steps']} steps "
                f"in {init_seconds:.1f}s",
                flush=True,
            )
            if "none" in finetunes:
                cell_rows.append(
                    _cell_row(
                        sample=sample,
                        suite=suite,
                        init=init,
                        arm="none",
                        anchor_weight=0.0,
                        steps=0,
                        background=init_result["background_pred"],
                        runtime_s=init_seconds,
                        extra={"physics_loss_final": init_result.get("physics_loss")},
                    )
                )

            forcing_state = None
            if any(arm != "data_only" for arm in finetunes if arm != "none"):
                forcing_started = time.perf_counter()
                forcing_state, _ = _fit_forcing(
                    sample, init_result, config, centers, suite=suite, device=device, seed=args.seed
                )
                print(
                    f"  [{sample['sample_id']}] init={init} forcing fit "
                    f"in {time.perf_counter() - forcing_started:.1f}s",
                    flush=True,
                )

            for arm in [item for item in finetunes if item != "none"]:
                arms = [(arm, 0.0)]
                if arm == "anchored":
                    arms = [(arm, weight) for weight in weights]
                for arm_name, weight in arms:
                    label = arm_name if arm_name != "anchored" else f"anchored@{weight:g}"
                    started = time.perf_counter()
                    result, _ = _run_finetune(
                        sample,
                        init_result,
                        config,
                        centers,
                        forcing_state,
                        arm=arm_name,
                        suite=suite,
                        anchor_weight=weight,
                        steps=budgets["finetune_steps"],
                        device=device,
                        seed=args.seed,
                        show_progress=not args.no_progress,
                    )
                    elapsed = time.perf_counter() - started
                    print(
                        f"  [{sample['sample_id']}] init={init} {label} "
                        f"{budgets['finetune_steps']} steps in {elapsed:.1f}s",
                        flush=True,
                    )
                    cell_rows.append(
                        _cell_row(
                            sample=sample,
                            suite=suite,
                            init=init,
                            arm=label,
                            anchor_weight=weight,
                            steps=budgets["finetune_steps"],
                            background=result["background_pred"],
                            runtime_s=elapsed,
                            extra={
                                "physics_loss_final": result.get("final_physics_loss"),
                                "data_loss_final": result.get("final_data_loss"),
                            },
                        )
                    )

            if args.curves:
                for curve_step in budgets["curve_steps"]:
                    if int(curve_step) <= 0 or int(curve_step) >= budgets["finetune_steps"]:
                        continue
                    result, _ = _run_finetune(
                        sample,
                        init_result,
                        config,
                        centers,
                        forcing_state,
                        arm="sgc",
                        suite=suite,
                        anchor_weight=0.0,
                        steps=int(curve_step),
                        device=device,
                        seed=args.seed,
                        show_progress=False,
                    )
                    print(
                        f"  [{sample['sample_id']}] init={init} curve sgc {int(curve_step)} steps",
                        flush=True,
                    )
                    cell_rows.append(
                        _cell_row(
                            sample=sample,
                            suite=suite,
                            init=init,
                            arm="sgc",
                            anchor_weight=0.0,
                            steps=int(curve_step),
                            background=result["background_pred"],
                            runtime_s=0.0,
                            curve_step=int(curve_step),
                        )
                    )

    attribution = attribution_rows(cell_rows)
    final_rows = [row for row in cell_rows if row.get("curve_step") is None]
    cell_summary = harness.summarise(final_rows, ("init", "arm"), VALUE_KEYS)
    attribution_summary = harness.summarise(
        attribution,
        ("arm",),
        (
            "init_effect",
            "refine_effect_data",
            "refine_effect_physics",
            "interaction",
            "retention",
            "bg_mae_data_final",
            "bg_mae_physics_final",
        ),
    )
    summary = {
        "schema_version": harness.SCHEMA_VERSION,
        "probe": PROBE,
        "suite": suite,
        "dataset": DATASET,
        "note": "non-formal diagnostic; not a paper result",
        "cells": cell_summary,
        "attribution": attribution_summary,
    }
    summary["observations"] = _observations(summary)

    harness.write_rows(run_dir / "runs.csv", cell_rows)
    harness.write_rows(run_dir / "attribution.csv", attribution)
    harness.write_rows(
        run_dir / "anchor_curve.csv",
        [row for row in cell_rows if str(row["arm"]).startswith("anchored")],
    )
    harness.write_json(run_dir / "summary.json", summary)
    (run_dir / "findings.md").write_text(
        _markdown(summary, attribution_summary, suite), encoding="utf-8"
    )

    print()
    print(f"init_memory ({suite}) -> {utils.project_relative(run_dir)}")
    for row in cell_summary:
        print(
            f"  {row['init']:>13s} {row['arm']:>14s} "
            f"projection={_fmt(row.get('projection_ratio_mean'))} "
            f"bg_mae={_fmt(row.get('bg_mae_mean'))} n={row['count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
