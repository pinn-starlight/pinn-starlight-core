"""Compare the star / nebula / balanced operating points (non-formal).

Stage-B **validation** only.  Arms:

* ``pinn_joint`` - joint data + physics, i.e. the ``nebula`` mode;
* ``sgc`` - the frozen decoupled pipeline, i.e. the ``star`` mode;
* ``balanced@w`` - physics-first stage 1 + anchored coordinated finetune.

The evaluation reports background quality, star photometry and the Stage-B
protected-structure projection together, so the trade-off is visible instead of
collapsed into one number.  The balanced acceptance rule is pre-declared in
``experiments/configs/diagnostics.py`` and is advisory: it never selects a
formal method.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.common.baselines import mode_switch
from experiments.common.utils import experiment_utils as utils
from experiments.configs import diagnostics as diag
from experiments.scripts.diagnostics import _harness as harness

PROBE = "mode_eval"
DATASET = "stage_b"
VALUE_KEYS = (
    "bg_mae",
    "bg_ssim",
    "residual_ssim",
    "projection_ratio",
    "protected_region_mae",
    "faint_star_recall",
    "star_f1",
    "matched_star_flux_error",
    "runtime_s",
)
DELTA_KEYS = (
    "projection_ratio",
    "bg_mae",
    "faint_star_recall",
    "star_f1",
    "matched_star_flux_error",
)


def _evaluate_row(*, sample: dict, suite: str, arm: str, anchor_weight: float, run: dict) -> dict:
    evaluated = harness.evaluate(sample, run["background_pred"])
    return {
        "probe": PROBE,
        "suite": suite,
        "sample_id": sample["sample_id"],
        "scene_type": sample.get("scene_type"),
        "intensity_level": sample.get("intensity_level"),
        "stride": sample["stride"],
        "arm": arm,
        "anchor_weight": float(anchor_weight),
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
        "runtime_s": float(run["runtime_s"]),
    }


def _finite(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _paired_deltas(rows: list[dict], baseline: str = "sgc") -> list[dict]:
    by_key = {(row["sample_id"], row["arm"]): row for row in rows}
    sample_ids = sorted({row["sample_id"] for row in rows})
    arms = sorted({row["arm"] for row in rows if row["arm"] != baseline})
    output = []
    for sample_id in sample_ids:
        reference = by_key.get((sample_id, baseline))
        if reference is None:
            continue
        for arm in arms:
            candidate = by_key.get((sample_id, arm))
            if candidate is None:
                continue
            row = {
                "probe": PROBE,
                "sample_id": sample_id,
                "arm": arm,
                "baseline": baseline,
            }
            for key in DELTA_KEYS:
                row[f"{key}_delta"] = _finite(candidate.get(key)) - _finite(reference.get(key))
            output.append(row)
    return output


def _fmt(value) -> str:
    number = _finite(value)
    return "n/a" if not np.isfinite(number) else f"{number:.4f}"


def _acceptance(rows: list[dict], summary: list[dict]) -> list[dict]:
    by_arm = {row["arm"]: row for row in summary}
    baseline = by_arm.get("sgc")
    if baseline is None:
        return []
    baseline_bg = _finite(baseline.get("bg_mae_mean"))
    output = []
    for arm in sorted(by_arm):
        if not arm.startswith("balanced@"):
            continue
        row = by_arm[arm]
        projection = _finite(row.get("projection_ratio_mean"))
        bg_mae = _finite(row.get("bg_mae_mean"))
        ratio = bg_mae / baseline_bg if baseline_bg and np.isfinite(baseline_bg) else float("nan")
        output.append(
            {
                "arm": arm,
                "projection_ratio_mean": projection,
                "bg_mae_mean": bg_mae,
                "bg_mae_ratio_vs_sgc": ratio,
                "min_projection_ratio": diag.BALANCED_ACCEPTANCE["min_projection_ratio"],
                "max_bg_mae_ratio_vs_sgc": diag.BALANCED_ACCEPTANCE["max_bg_mae_ratio_vs_sgc"],
                "accepted": bool(
                    np.isfinite(projection)
                    and projection >= diag.BALANCED_ACCEPTANCE["min_projection_ratio"]
                    and np.isfinite(ratio)
                    and ratio <= diag.BALANCED_ACCEPTANCE["max_bg_mae_ratio_vs_sgc"]
                ),
            }
        )
    return output


def _markdown(summary: list[dict], acceptance: list[dict], suite: str) -> str:
    lines = [
        f"# mode_eval · {suite}",
        "",
        "> Non-formal operating-point comparison. Validation split only. "
        + (
            "Smoke budgets: plumbing check, **not** a scientific result."
            if suite == "smoke"
            else "Declared diagnostic budget; still not a paper result."
        ),
        "",
        "| arm | projection | bg MAE | star F1 | flux error | n |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary:
        lines.append(
            f"| {row['arm']} | {_fmt(row.get('projection_ratio_mean'))} "
            f"| {_fmt(row.get('bg_mae_mean'))} | {_fmt(row.get('star_f1_mean'))} "
            f"| {_fmt(row.get('matched_star_flux_error_mean'))} | {row['count']} |"
        )
    if acceptance:
        lines += [
            "",
            "## Balanced acceptance (pre-declared, advisory)",
            "",
            "| arm | projection | bg ratio vs sgc | accepted |",
            "| --- | --- | --- | --- |",
        ]
        for row in acceptance:
            lines.append(
                f"| {row['arm']} | {_fmt(row['projection_ratio_mean'])} "
                f"| {_fmt(row['bg_mae_ratio_vs_sgc'])} | {row['accepted']} |"
            )
    lines.append("")
    return "\n".join(lines)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=diag.SUITES, default="smoke")
    parser.add_argument("--samples", default="")
    parser.add_argument("--max-side", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=diag.DIAGNOSTIC_SEED)
    parser.add_argument("--anchor-weights", default=",".join(str(v) for v in diag.ANCHOR_WEIGHTS))
    parser.add_argument("--max-samples", type=int, default=0)
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
    weights = [float(item) for item in args.anchor_weights.split(",") if item.strip()]

    rows = harness.load_rows(DATASET, args.samples or None)
    limit = args.max_samples or int(budget["max_samples"])
    rows = harness.require_available(
        rows, DATASET, allow_subset=(suite == "smoke" and not args.samples), limit=limit
    )
    samples = [harness.load_sample(row, DATASET, max_side) for row in rows]

    run_dir = harness.new_run_dir(PROBE, suite, args.out)
    budgets = {
        "anchor_weights": weights,
        "stage1_steps": diag.BUDGETS[suite]["init_steps"],
        "finetune_steps": diag.BUDGETS[suite]["finetune_steps"],
        "pinn_steps": diag.BUDGETS[suite]["alpha_steps"],
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
        ),
    )

    run_rows: list[dict] = []
    for sample in samples:
        print(f"[mode_eval] {sample['sample_id']} stride={sample['stride']}", flush=True)
        config = mode_switch.build_config(suite=suite)
        for mode, arm in (("nebula", "pinn_joint"), ("star", "sgc")):
            run = mode_switch.run_mode(
                sample["observed"],
                mode=mode,
                config=config,
                device=device,
                seed=args.seed,
                show_progress=not args.no_progress,
            )
            run_rows.append(
                _evaluate_row(
                    sample=sample,
                    suite=suite,
                    arm=arm,
                    anchor_weight=0.0,
                    run=run,
                )
            )
        for weight in weights:
            config = mode_switch.build_config(suite=suite, anchor_weight=weight)
            run = mode_switch.run_mode(
                sample["observed"],
                mode="balanced",
                config=config,
                device=device,
                seed=args.seed,
                show_progress=not args.no_progress,
            )
            run_rows.append(
                _evaluate_row(
                    sample=sample,
                    suite=suite,
                    arm=f"balanced@{weight:g}",
                    anchor_weight=weight,
                    run=run,
                )
            )

    deltas = _paired_deltas(run_rows)
    summary = harness.summarise(run_rows, ("arm",), VALUE_KEYS)
    acceptance = _acceptance(run_rows, summary)
    payload = {
        "schema_version": harness.SCHEMA_VERSION,
        "probe": PROBE,
        "suite": suite,
        "dataset": DATASET,
        "note": "non-formal diagnostic; not a paper result",
        "summary": summary,
        "acceptance": acceptance,
    }
    harness.write_rows(run_dir / "runs.csv", run_rows)
    harness.write_rows(run_dir / "paired_deltas.csv", deltas)
    harness.write_rows(run_dir / "summary.csv", summary)
    harness.write_json(run_dir / "summary.json", payload)
    (run_dir / "findings.md").write_text(_markdown(summary, acceptance, suite), encoding="utf-8")

    print()
    print(f"mode_eval ({suite}) -> {utils.project_relative(run_dir)}")
    for row in summary:
        print(
            f"  {row['arm']:>13s} projection={_fmt(row.get('projection_ratio_mean'))} "
            f"bg_mae={_fmt(row.get('bg_mae_mean'))} "
            f"flux={_fmt(row.get('matched_star_flux_error_mean'))} n={row['count']}"
        )
    if acceptance:
        print(f"  balanced acceptance: {acceptance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
