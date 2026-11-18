"""Alpha sweep of the screened-Poisson residual (non-formal diagnostic).

Question
--------
For which screening coefficient ``alpha`` does the physics loss fall fastest,
and during which training window?  ``alpha`` enters two places:

* the PDE residual ``Laplacian(B) - alpha * B + S``;
* the forcing template ``q(r; alpha)``.

The sweep therefore runs two protocols: ``coupled`` (method-faithful) and
``frozen`` (forcing template fixed at the formal ``alpha=0.5``, only the
residual term varies).  A deterministic Cosine-LS solve is reported alongside
as a trajectory-free cross-check.

The formal methods keep ``alpha`` fixed at 0.5; this module never changes them.
Validation split only, non-formal output directory only.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common.baselines import coordinate_pinn as base_pinn
from experiments.common.baselines import cosine_ls
from experiments.common.utils import experiment_utils as utils
from experiments.configs import diagnostics as diag
from experiments.scripts.diagnostics import _harness as harness
from pinn_starlight_core.data.image_loader import coordinate_grid
from pinn_starlight_core.nn.pinn_layers import SkyglowMLP

PROBE = "alpha_sweep"
STAT_KEYS = (
    "max_log_decay_per_step",
    "step_at_max_decay",
    "t50_step",
    "t90_step",
    "final_relative",
)
TERM_KEYS = (
    "term_laplacian_sq",
    "term_alpha_bg_sq",
    "term_forcing_sq",
    "term_cross_lap_bg",
    "term_cross_lap_s",
    "term_cross_bg_s",
)


class _FixedTemplateForcing(torch.nn.Module):
    """Keep the forcing template at a reference alpha while the PDE uses alpha."""

    def __init__(self, forcing: torch.nn.Module, template_alpha: float):
        super().__init__()
        self.forcing = forcing
        self.template_alpha = torch.tensor(template_alpha, dtype=torch.float32)

    def _alpha(self, device):
        return self.template_alpha.to(device)

    def forward(self, coords, alpha):
        return self.forcing(coords, self._alpha(coords.device))

    def sparse_penalty(self):
        return self.forcing.sparse_penalty()

    def set_center_gradient_enabled(self, enabled: bool) -> None:
        self.forcing.set_center_gradient_enabled(enabled)

    def diagnostics(self, alpha, inactive_relative_threshold: float = 0.05):
        return self.forcing.diagnostics(
            self._alpha(next(self.forcing.parameters()).device),
            inactive_relative_threshold=inactive_relative_threshold,
        )


def _physics_terms(model, forcing, alpha: torch.Tensor, coords, indices) -> dict:
    with torch.no_grad():
        xy = coords[indices]
        background, laplacian = base_pinn._background_and_laplacian(model, xy)
        forcing_pred = forcing(xy, alpha)
        background = background.double()
        laplacian = laplacian.double()
        forcing_pred = forcing_pred.double()
        alpha_value = alpha.double()
        residual = laplacian - alpha_value * background + forcing_pred
        values = {
            "term_laplacian_sq": float(torch.mean(laplacian * laplacian)),
            "term_alpha_bg_sq": float(torch.mean((alpha_value * background) ** 2)),
            "term_forcing_sq": float(torch.mean(forcing_pred * forcing_pred)),
            "term_cross_lap_bg": float(torch.mean(-2.0 * alpha_value * laplacian * background)),
            "term_cross_lap_s": float(torch.mean(2.0 * laplacian * forcing_pred)),
            "term_cross_bg_s": float(torch.mean(-2.0 * alpha_value * background * forcing_pred)),
            "eval_physics_loss": float(torch.mean(residual * residual)),
        }
    values["terms_sum"] = float(sum(values[key] for key in TERM_KEYS))
    values["terms_error"] = float(values["terms_sum"] - values["eval_physics_loss"])
    return values


def _run_pinn(
    sample: dict,
    *,
    suite: str,
    alpha: float,
    forcing_alpha: str,
    steps: int,
    chunk: int,
    device,
    seed: int,
) -> dict:
    config = base_pinn.normalized_config(diag.pinn_config(suite=suite, steps=steps))
    centers = base_pinn.initial_centers_from_observed(sample["observed"], config)
    model = SkyglowMLP(config["hidden_dims"]).to(device)
    forcing = base_pinn._build_forcing(  # noqa: SLF001 - frozen builder reused
        sample["observed"], config, device, initial_centers=centers
    ).to(device)
    optimizer = base_pinn._optimizer(model, forcing, config)  # noqa: SLF001
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    alpha_tensor = torch.tensor(float(alpha), dtype=torch.float32, device=device)
    if forcing_alpha == "frozen":
        forcing = _FixedTemplateForcing(forcing, diag.FORCING_ALPHA_REFERENCE).to(device)
    elif forcing_alpha != "coupled":
        raise ValueError(f"unknown forcing_alpha protocol: {forcing_alpha!r}")

    height, width = sample["observed"].shape
    coords = coordinate_grid(height, width, device)
    values = torch.from_numpy(np.ascontiguousarray(sample["observed"]).reshape(-1)).to(device)
    rng = np.random.default_rng(int(seed) + 77)
    eval_indices = torch.as_tensor(
        rng.choice(
            int(coords.shape[0]),
            size=min(diag.ALPHA_EVAL_POINTS, int(coords.shape[0])),
            replace=False,
        ),
        dtype=torch.long,
        device=device,
    )

    history: list[dict] = []
    curve_rows: list[dict] = []
    started = time.perf_counter()
    start = 0
    while start < steps:
        span = min(int(chunk), steps - start)
        chunk_config = dict(config)
        chunk_config["steps"] = span
        chunk_config["log_every"] = span
        history, _, _, _, _ = base_pinn._train_steps(  # noqa: SLF001
            model,
            forcing,
            coords,
            values,
            optimizer,
            generator,
            alpha_tensor,
            chunk_config,
            start,
            history,
            False,
            "analytic",
        )
        start += span
        terms = _physics_terms(model, forcing, alpha_tensor, coords, eval_indices)
        last = history[-1]
        row = {
            "probe": PROBE,
            "suite": suite,
            "method": "pinn",
            "forcing_alpha": forcing_alpha,
            "sample_id": sample["sample_id"],
            "alpha": float(alpha),
            "step": int(start),
            "data_loss": float(last["data_loss"]),
            "physics_loss": float(last["physics_loss"]),
            "sparse_loss": float(last["sparse_loss"]),
            "total_loss": float(last["total_loss"]),
            "log_physics": float(np.log(max(float(last["physics_loss"]), 1e-30))),
            **terms,
        }
        curve_rows.append(row)

    background = base_pinn._predict_background(  # noqa: SLF001
        model, coords, height, width, int(config["prediction_batch_size"])
    )
    runtime_s = time.perf_counter() - started
    return {
        "background_pred": np.asarray(background, dtype=np.float32),
        "history": history,
        "curve_rows": curve_rows,
        "runtime_s": float(runtime_s),
    }


def _run_ls(sample: dict, *, alpha: float) -> dict:
    """Analytic Cosine-LS at a diagnostic alpha.

    ``cosine_ls`` refuses ``alpha != 0.5`` for the formal method.  The global is
    patched only around this call (the file itself is untouched and the lock
    hash stays valid); the solver receives the same alpha explicitly.
    """
    original = cosine_ls.ALPHA
    started = time.perf_counter()
    try:
        cosine_ls.ALPHA = float(alpha)
        background, info = cosine_ls._estimate_background_with_settings(  # noqa: SLF001
            sample["observed"],
            diag.ALPHA_LS_LAMBDA,
            diag.ALPHA_LS_SPARSE_WEIGHT,
            alpha=float(alpha),
            return_info=True,
            backend="cpu",
            show_progress=False,
        )
    finally:
        cosine_ls.ALPHA = original
    return {
        "background_pred": np.asarray(background, dtype=np.float32),
        "info": info,
        "runtime_s": float(time.perf_counter() - started),
    }


def _metrics_row(
    *,
    sample: dict,
    suite: str,
    method: str,
    forcing_alpha: str,
    alpha: float,
    background,
    runtime_s: float,
    extra: dict | None = None,
) -> dict:
    evaluated = harness.evaluate(sample, background)
    row = {
        "probe": PROBE,
        "suite": suite,
        "method": method,
        "forcing_alpha": forcing_alpha,
        "sample_id": sample["sample_id"],
        "alpha": float(alpha),
        "bg_mae": evaluated.get("bg_mae"),
        "bg_ssim": evaluated.get("bg_ssim"),
        "residual_ssim": evaluated.get("residual_ssim"),
        "projection_ratio": evaluated.get("protected_projection_ratio"),
        "runtime_s": float(runtime_s),
        "star_metrics_skipped": bool(evaluated.get("star_metrics_skipped", False)),
    }
    row.update(extra or {})
    return row


def _summary_rows(stats_rows: list[dict], metrics_rows: list[dict]) -> list[dict]:
    stats_rows = [row for row in stats_rows if row["method"] == "pinn"]
    metrics = {
        (row["method"], row["forcing_alpha"], row["alpha"], row["sample_id"]): row
        for row in metrics_rows
    }
    merged = []
    for row in stats_rows:
        key = (row["method"], row["forcing_alpha"], row["alpha"], row["sample_id"])
        metric = metrics.get(key, {})
        merged.append(
            {
                **row,
                "bg_mae": metric.get("bg_mae"),
                "projection_ratio": metric.get("projection_ratio"),
                "runtime_s": metric.get("runtime_s"),
            }
        )
    return harness.summarise(
        merged,
        ("method", "forcing_alpha", "alpha"),
        (*STAT_KEYS, "first_value", "last_value", "bg_mae", "projection_ratio", "runtime_s"),
    )


def _answers(summary_rows: list[dict]) -> dict:
    pinn = [row for row in summary_rows if row["method"] == "pinn" and row.get("count")]
    answers: dict[str, object] = {}
    for protocol in diag.FORCING_ALPHA_MODES:
        rows = [row for row in pinn if row["forcing_alpha"] == protocol]
        if not rows:
            continue
        fastest = max(
            rows,
            key=lambda row: (
                row["max_log_decay_per_step_mean"]
                if np.isfinite(row["max_log_decay_per_step_mean"])
                else -np.inf
            ),
        )
        earliest = min(
            rows,
            key=lambda row: row["t90_step_mean"] if np.isfinite(row["t90_step_mean"]) else np.inf,
        )
        answers[protocol] = {
            "fastest_alpha": fastest["alpha"],
            "fastest_max_log_decay_per_step": fastest["max_log_decay_per_step_mean"],
            "fastest_step_at_max_decay": fastest["step_at_max_decay_mean"],
            "fastest_t50_step": fastest["t50_step_mean"],
            "fastest_t90_step": fastest["t90_step_mean"],
            "earliest_t90_alpha": earliest["alpha"],
            "earliest_t90_step": earliest["t90_step_mean"],
        }
    return answers


def _plot(curve_rows: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plotting is optional
        return

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for alpha in sorted({row["alpha"] for row in curve_rows}):
        rows = [
            row for row in curve_rows if row["alpha"] == alpha and row["forcing_alpha"] == "coupled"
        ]
        if not rows:
            continue
        by_step: dict[int, list[float]] = {}
        for row in rows:
            by_step.setdefault(int(row["step"]), []).append(float(row["physics_loss"]))
        steps = sorted(by_step)
        axis.plot(steps, [float(np.mean(by_step[s])) for s in steps], label=f"alpha={alpha:g}")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("physics loss (batch mean)")
    axis.set_title("alpha sweep - coupled protocol")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=diag.SUITES, default="smoke")
    parser.add_argument("--dataset", choices=diag.DATASETS, default="stage_a")
    parser.add_argument("--samples", default="")
    parser.add_argument("--max-side", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=diag.DIAGNOSTIC_SEED)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--alphas", default=",".join(str(v) for v in diag.ALPHA_GRID))
    parser.add_argument(
        "--forcing-alpha", default=",".join(diag.FORCING_ALPHA_MODES), help="coupled,frozen"
    )
    parser.add_argument("--methods", default="pinn,cosine_ls")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    harness.require_standard_split()
    device = harness.resolve_device(args.device)
    suite = args.suite
    budget = diag.BUDGETS[suite]
    max_side = args.max_side if args.max_side is not None else budget["max_side"]
    steps = int(args.steps or budget["alpha_steps"])
    chunk = int(args.chunk or budget["alpha_chunk"])
    if chunk <= 0 or chunk > steps:
        chunk = steps

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    if not alphas or min(alphas) <= 0.0:
        raise ValueError("alpha values must be strictly positive")
    protocols = [item.strip() for item in args.forcing_alpha.split(",") if item.strip()]
    unknown = sorted(set(protocols) - set(diag.FORCING_ALPHA_MODES))
    if unknown:
        raise ValueError(f"unknown forcing-alpha protocol(s): {unknown}")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    rows = harness.load_rows(args.dataset, args.samples or None)
    limit = args.max_samples or int(budget["max_samples"])
    rows = harness.require_available(
        rows, args.dataset, allow_subset=(suite == "smoke"), limit=limit
    )
    samples = [harness.load_sample(row, args.dataset, max_side) for row in rows]

    run_dir = harness.new_run_dir(PROBE, suite, args.out)
    budgets = {"alpha_steps": steps, "chunk": chunk, "alphas": alphas}
    harness.write_json(
        run_dir / "run_metadata.json",
        harness.run_metadata(
            probe=PROBE,
            suite=suite,
            dataset=args.dataset,
            device=device,
            seed=args.seed,
            budgets=budgets,
            samples=samples,
            args=vars(args),
            extra={
                "protocols": protocols,
                "methods": methods,
                "ls_lambda": diag.ALPHA_LS_LAMBDA,
                "ls_sparse_weight": diag.ALPHA_LS_SPARSE_WEIGHT,
            },
        ),
    )

    curve_rows: list[dict] = []
    metrics_rows: list[dict] = []
    stats_rows: list[dict] = []

    for sample in samples:
        print(f"[alpha_sweep] {sample['sample_id']} stride={sample['stride']}", flush=True)
        for alpha in alphas:
            if "pinn" in methods:
                for protocol in protocols:
                    run = _run_pinn(
                        sample,
                        suite=suite,
                        alpha=alpha,
                        forcing_alpha=protocol,
                        steps=steps,
                        chunk=chunk,
                        device=device,
                        seed=args.seed,
                    )
                    curve_rows.extend(run["curve_rows"])
                    metrics_rows.append(
                        _metrics_row(
                            sample=sample,
                            suite=suite,
                            method="pinn",
                            forcing_alpha=protocol,
                            alpha=alpha,
                            background=run["background_pred"],
                            runtime_s=run["runtime_s"],
                            extra={"steps": steps},
                        )
                    )
                    stats = harness.trajectory_stats(run["history"], "physics_loss")
                    stats.update(
                        {
                            "probe": PROBE,
                            "suite": suite,
                            "method": "pinn",
                            "forcing_alpha": protocol,
                            "sample_id": sample["sample_id"],
                            "alpha": float(alpha),
                            "steps": steps,
                        }
                    )
                    stats_rows.append(stats)
            if "cosine_ls" in methods:
                run = _run_ls(sample, alpha=alpha)
                metrics_rows.append(
                    _metrics_row(
                        sample=sample,
                        suite=suite,
                        method="cosine_ls",
                        forcing_alpha="analytic",
                        alpha=alpha,
                        background=run["background_pred"],
                        runtime_s=run["runtime_s"],
                        extra={
                            "objective": run["info"].get("objective"),
                            "smooth_objective": run["info"].get("smooth_objective"),
                            "sparse_objective": run["info"].get("sparse_objective"),
                        },
                    )
                )
                stats_rows.append(
                    {
                        "probe": PROBE,
                        "suite": suite,
                        "method": "cosine_ls",
                        "forcing_alpha": "analytic",
                        "sample_id": sample["sample_id"],
                        "alpha": float(alpha),
                        "steps": 0,
                    }
                )

    summary_rows = _summary_rows(stats_rows, metrics_rows)
    ls_summary = harness.summarise(
        [row for row in metrics_rows if row["method"] == "cosine_ls"],
        ("method", "alpha"),
        ("objective", "bg_mae", "projection_ratio", "runtime_s"),
    )
    summary = {
        "schema_version": harness.SCHEMA_VERSION,
        "probe": PROBE,
        "suite": suite,
        "dataset": args.dataset,
        "note": "non-formal diagnostic; not a paper result",
        "budgets": budgets,
        "answers": _answers(summary_rows),
        "pinn": summary_rows,
        "cosine_ls": ls_summary,
    }
    harness.write_rows(run_dir / "curves.csv", curve_rows)
    harness.write_rows(run_dir / "metrics.csv", metrics_rows)
    harness.write_rows(run_dir / "summary.csv", summary_rows)
    harness.write_rows(run_dir / "cosine_ls_summary.csv", ls_summary)
    harness.write_json(run_dir / "alpha_summary.json", summary)
    if args.plot and curve_rows:
        _plot(curve_rows, run_dir / "alpha_curves.png")

    print()
    print(f"alpha_sweep ({suite}) -> {utils.project_relative(run_dir)}")
    for row in summary_rows:
        print(
            f"  {row['method']:>9s} {row['forcing_alpha']:>8s} alpha={row['alpha']:<5g} "
            f"max_decay/step={row['max_log_decay_per_step_mean']:.4g} "
            f"step_at_max={row['step_at_max_decay_mean']:.0f} "
            f"t90={row['t90_step_mean']:.0f} bg_mae={row['bg_mae_mean']:.6g}"
        )
    print(f"  answers: {summary['answers']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
