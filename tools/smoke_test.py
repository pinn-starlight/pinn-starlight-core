"""CPU-friendly smoke test for the whole PINN-Starlight experimental stack.

Purpose
-------
A competition code audit may try to *run* this repository.  The formal
experiments need a GPU, ~11 GB of synthetic data and many GPU-hours, so a full
reproduction is not possible on a laptop.  This script instead exercises every
scientific code path end-to-end on a **tiny CPU budget**:

    data -> background estimators -> PINN -> SGC-PINN -> metrics

Nothing here is a scientific result.  Steps are deliberately tiny so that the
whole run finishes in a few minutes on a CPU.  The goal is only to prove that
the code executes, is internally consistent, and produces finite arrays.

Usage
-----
    python -m tools.smoke_test                 # everything, CPU defaults
    python -m tools.smoke_test --list          # list test names
    python -m tools.smoke_test --only pinn,final_pinn
    python -m tools.smoke_test --steps 8 --downsample 24

Exit code is 0 only if every selected test passes.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.common.utils import experiment_utils as utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STAGE_A = "clean_single_002_multi_source_high"
DEFAULT_STAGE_B = "clean_single_002_protected_overlap_high"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _stage_a_row(sample_id: str = DEFAULT_STAGE_A) -> dict:
    rows = utils.load_manifest(utils.SYNTHETIC_DATA_ROOT / "manifest.csv", split="validation")
    for row in rows:
        if row["sample_id"] == sample_id:
            return row
    raise LookupError(f"{sample_id} not found in the Stage-A validation manifest")


def _stage_b_row(sample_id: str = DEFAULT_STAGE_B) -> dict:
    from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST

    rows = utils.load_manifest(STAGE_B_MANIFEST, split="validation")
    for row in rows:
        if row["sample_id"] == sample_id:
            return row
    raise LookupError(f"{sample_id} not found in the Stage-B validation manifest")


def _load_raw(row: dict, key: str, downsample: int) -> np.ndarray:
    """Load a formal synthetic TIFF without clipping, then stride-downsample."""
    array = utils.load_float_tiff(row[key])
    if downsample > 1:
        array = array[::downsample, ::downsample]
    return np.ascontiguousarray(array, dtype=np.float32)


def _load_sample(row: dict, downsample: int, *, stage_b: bool = False) -> dict:
    sample = {
        "sample_id": row["sample_id"],
        "observed": _load_raw(row, "observed", downsample),
        "background_true": _load_raw(row, "background_true", downsample),
        "clean_true": _load_raw(row, "clean_true", downsample),
        "metadata": utils.read_json(utils.to_absolute_path(row["metadata"])),
    }
    if stage_b:
        sample["base_clean"] = _load_raw(row, "base_clean", downsample)
        sample["protected_true"] = _load_raw(row, "protected_true", downsample)
    return sample


def _cached(ctx: dict, key: str, builder):
    """Load expensive samples once and reuse them across tests."""
    if key not in ctx:
        ctx[key] = builder()
    return ctx[key]


def _stage_a(ctx: dict, downsample: int | None = None) -> dict:
    ds = ctx["downsample"] if downsample is None else downsample
    return _cached(ctx, f"stage_a@{ds}", lambda: _load_sample(_stage_a_row(), ds))


def _stage_b(ctx: dict, downsample: int | None = None) -> dict:
    ds = ctx["downsample"] if downsample is None else downsample
    return _cached(ctx, f"stage_b@{ds}", lambda: _load_sample(_stage_b_row(), ds, stage_b=True))


def _assert_finite(name: str, array) -> None:
    array = np.asarray(array)
    if not np.all(np.isfinite(array)):
        raise AssertionError(f"{name} contains NaN/Inf")
    if array.size == 0:
        raise AssertionError(f"{name} is empty")


def _check_prediction(tag: str, result: dict, shape: tuple[int, int]) -> None:
    for key in ("background_pred", "residual_pred"):
        if key not in result:
            raise KeyError(f"{tag} result is missing {key!r}")
        arr = np.asarray(result[key])
        if arr.shape != shape:
            raise AssertionError(f"{tag}.{key} shape {arr.shape} != observed {shape}")
        _assert_finite(f"{tag}.{key}", arr)


def _tiny_pinn_config(steps: int) -> dict:
    return {
        "hidden_dims": [32, 16],
        "steps": int(steps),
        "batch_size": 4096,
        "model_lr": 1e-3,
        "alpha": 0.5,
        "physics_weight": 0.4,
        "sparse_weight": 0.0,
        "center_warmup_steps": min(5, int(steps)),
        "center_refinement_enabled": False,
        "forcing_max_components": 2,
        "forcing_rms_grid_size": 32,
        "prediction_batch_size": 65536,
        "log_every": max(1, int(steps) // 3),
    }


def _tiny_stage3_config(steps: int) -> dict:
    return {
        "steps": int(steps),
        "model_lr": 2e-4,
        "rho": 0.1,
        "lambda_min": 0.0,
        "lambda_max": 0.02,
        "ramp_steps": max(1, int(steps) // 2),
        "conflict_projection": True,
        "sparse_weight": 0.0,
        "neumann_boundary_loss": False,
        "gradient_balance_reference": "projected_regularizer_gradient",
        "laplacian_backend": "analytic",
        "log_every": max(1, int(steps) // 3),
    }


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_imports(ctx):
    """Every module in src/, experiments/, tools/ and research_diagnostics/ imports."""
    import importlib

    modules, failures = [], []
    for base in ("src", "experiments", "tools", "research_diagnostics"):
        for path in (PROJECT_ROOT / base).rglob("*.py"):
            parts = list(path.relative_to(PROJECT_ROOT).parts)
            if parts[0] == "src":
                parts = parts[1:]
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            else:
                parts[-1] = parts[-1][:-3]
            if parts:
                modules.append(".".join(parts))
    for name in sorted(set(modules)):
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    if failures:
        raise AssertionError(
            f"{len(failures)}/{len(set(modules))} modules failed to import:\n  "
            + "\n  ".join(failures)
        )
    return f"{len(set(modules))} modules imported"


def test_analytic_laplacian(ctx):
    """Analytic sigmoid Laplacian must match PyTorch autograd."""
    from experiments.common.baselines import coordinate_pinn as pinn
    from pinn_starlight_core.nn.pinn_layers import SkyglowMLP

    utils.set_seed(90117)
    device = ctx["device"]
    model = SkyglowMLP([16, 8]).to(device)
    xy = torch.tensor(
        [[-0.8, -0.5], [-0.1, 0.2], [0.4, -0.7], [0.75, 0.6]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    analytic_b, analytic_lap = pinn._background_and_laplacian(model, xy)
    direct = torch.sigmoid(model(xy).squeeze(-1))
    grad = torch.autograd.grad(direct, xy, grad_outputs=torch.ones_like(direct), create_graph=True)[
        0
    ]
    parts = []
    for axis in (0, 1):
        second = torch.autograd.grad(
            grad[:, axis],
            xy,
            grad_outputs=torch.ones_like(grad[:, axis]),
            create_graph=False,
            retain_graph=(axis == 0),
        )[0][:, axis]
        parts.append(second)
    auto_lap = parts[0] + parts[1]
    b_err = float(torch.max(torch.abs(analytic_b - direct)).detach())
    lap_err = float(torch.max(torch.abs(analytic_lap - auto_lap)).detach())
    if b_err > 2e-6 or lap_err > 2e-5:
        raise AssertionError(f"b_err={b_err:.3e} lap_err={lap_err:.3e}")
    return f"background err {b_err:.2e}, laplacian err {lap_err:.2e}"


def test_forcing_construction(ctx):
    """Both EffectiveForcingMixture entry points must agree.

    The demo builds the mixture from an ``ImageLoader`` while the formal
    experiments pass a raw array, so a break in one path stays invisible to
    the other.  Assert they produce the same geometry and the same output.
    """
    from pinn_starlight_core.data.image_loader import ImageLoader, rgb_to_gray
    from pinn_starlight_core.nn.physics_model import EffectiveForcingMixture

    device = ctx["device"]
    path = utils.NATIVE_TEST_IMAGE
    if not path.is_file():
        return "skipped (native test image missing)"

    loader = ImageLoader(path, device, downsample=ctx["downsample"])
    from_loader = EffectiveForcingMixture(device, loader=loader, max_components=2)
    from_array = EffectiveForcingMixture(
        device, gray_img=rgb_to_gray(loader.rgb_data), max_components=2
    )

    same_centers = torch.equal(from_loader.initial_centers, from_array.initial_centers)
    if not same_centers:
        raise AssertionError("initial centers differ between loader and array construction")

    alpha = torch.tensor(0.5, device=device)
    coords = torch.tensor([[-0.5, 0.2], [0.1, -0.4], [0.7, 0.7]], device=device)
    with torch.no_grad():
        out_loader = from_loader(coords, alpha)
        out_array = from_array(coords, alpha)
    if not torch.equal(out_loader, out_array):
        raise AssertionError("forcing output differs between loader and array construction")
    if not torch.isfinite(out_loader).all():
        raise AssertionError("forcing output is not finite")
    return f"{from_loader.max_components} components, max |S| {float(out_loader.abs().max()):.3e}"


def test_data_loading(ctx):
    """Stage-A and Stage-B manifests + raw arrays load, and Stage-B truth is consistent."""
    a = _stage_a(ctx)
    b = _stage_b(ctx)
    for name in ("observed", "background_true", "clean_true"):
        _assert_finite(f"stageA.{name}", a[name])
    for name in (
        "observed",
        "background_true",
        "clean_true",
        "base_clean",
        "protected_true",
    ):
        _assert_finite(f"stageB.{name}", b[name])

    # Stage-B construction identity, checked on unclipped arrays:
    #   observed = base_clean + protected_true + background_true
    recon = b["base_clean"] + b["protected_true"] + b["background_true"]
    err = float(np.max(np.abs(recon - b["observed"])))
    if err > 1e-4:
        raise AssertionError(f"Stage-B identity residual too large: {err:.3e}")

    # clean_true = base_clean + protected_true
    clean_err = float(np.max(np.abs(b["clean_true"] - (b["base_clean"] + b["protected_true"]))))
    if clean_err > 1e-4:
        raise AssertionError(f"Stage-B clean identity residual too large: {clean_err:.3e}")

    note = ""
    if float(np.max(a["observed"])) > 1.0:
        note = ", observed exceeds 1 (unclipped, as designed)"
    return (
        f"A{a['observed'].shape} B{b['observed'].shape}, "
        f"identity err {err:.1e}/{clean_err:.1e}{note}"
    )


def test_fft_gaussian(ctx):
    """FFT-Gaussian baseline produces a valid background/residual pair."""
    from experiments.common.baselines import fft_gaussian as fft

    sample = _stage_a(ctx)
    background = fft.estimate_background(sample["observed"], 0.02)
    residual = utils.reconstruct_residual(sample["observed"], background)
    result = {"background_pred": background, "residual_pred": residual}
    _check_prediction("fft_gaussian", result, sample["observed"].shape)
    return f"bg mean {float(np.mean(background)):.4f}"


def test_fourier_ls(ctx):
    """DCT-II screened-Poisson (Cosine-LS) solver runs on CPU."""
    from experiments.common.baselines import cosine_ls as fourier_ls

    sample = _stage_a(ctx)
    background, info = fourier_ls.estimate_background(
        sample["observed"],
        1e-3,
        1e-4,
        return_info=True,
        backend="cpu",
        show_progress=False,
    )
    residual = utils.reconstruct_residual(sample["observed"], background)
    result = {"background_pred": background, "residual_pred": residual}
    _check_prediction("fourier_ls", result, sample["observed"].shape)
    return f"bg mean {float(np.mean(background)):.4f}, info keys {len(info)}"


def test_pinn(ctx):
    """Original coordinate PINN (joint data+physics) trains and predicts."""
    from experiments.common.baselines import coordinate_pinn as pinn

    sample = _stage_a(ctx)
    result = pinn.train_background(
        sample["observed"],
        config=_tiny_pinn_config(ctx["steps"]),
        device=ctx["device"],
        seed=20260728,
        show_progress=False,
        return_state=True,
        laplacian_backend="analytic",
    )
    # Formal Method contract: train_background returns B_hat only; the
    # experiment layer reconstructs the residual from the observation.
    prediction = {
        "background_pred": result["background_pred"],
        "residual_pred": utils.reconstruct_residual(sample["observed"], result["background_pred"]),
    }
    _check_prediction("pinn", prediction, sample["observed"].shape)
    hist = result.get("history") or []
    return f"{len(hist)} logged steps, bg mean {float(np.mean(result['background_pred'])):.4f}"


def test_final_pinn(ctx):
    """SGC-PINN: data-only stage-1 -> fixed forcing fit -> coordinated finetune."""
    from experiments.common.baselines import coordinate_pinn as pinn
    from experiments.common.baselines import sgc_pinn as final_pinn

    sample = _stage_a(ctx)
    observed = sample["observed"]
    # fit_fixed_forcing / finetune expect an already-normalised Stage-1 config,
    # exactly as the formal runners do via pinn.normalized_config(...).
    stage1_cfg = pinn.normalized_config(_tiny_pinn_config(ctx["steps"]))
    stage2_cfg = {
        "mode": "fixed_geometry_nonnegative_least_squares",
        "prefit_points": 4096,
        "fit_fraction": 0.8,
        "geometry_refinement": False,
        "uses_true_background": False,
    }
    stage3_cfg = _tiny_stage3_config(ctx["steps"])

    initial_centers = pinn.initial_centers_from_observed(observed, stage1_cfg)
    stage1 = pinn.train_background(
        observed,
        config=stage1_cfg,
        device=ctx["device"],
        seed=20260728,
        show_progress=False,
        return_state=True,
        laplacian_backend="analytic",
        initial_centers=initial_centers,
    )
    forcing_state, forcing_diag = final_pinn.fit_fixed_forcing(
        observed=observed,
        model_state=stage1["state"]["model_state"],
        stage1_config=stage1_cfg,
        stage2_config=stage2_cfg,
        initial_centers=initial_centers,
        device=ctx["device"],
        seed=20260728 + 1009,
    )
    refined = final_pinn.finetune(
        observed=observed,
        model_state=stage1["state"]["model_state"],
        forcing_state=forcing_state,
        stage1_config=stage1_cfg,
        stage3_config=stage3_cfg,
        initial_centers=initial_centers,
        device=ctx["device"],
        seed=20260728 + 2003,
        use_physics=True,
        show_progress=False,
    )
    prediction = {
        "background_pred": refined["background_pred"],
        "residual_pred": utils.reconstruct_residual(observed, refined["background_pred"]),
    }
    _check_prediction("final_pinn", prediction, observed.shape)
    return (
        f"forcing diag keys {len(forcing_diag)}, "
        f"conflict frac {refined.get('gradient_conflict_fraction', float('nan')):.3f}, "
        f"bg mean {float(np.mean(refined['background_pred'])):.4f}"
    )


def test_metrics_stage_a(ctx):
    """Stage-A metric stack (MAE/SSIM/PSNR + star photometry) evaluates a prediction."""
    from experiments.common.utils import metrics

    # Metrics need real stellar PSFs, so evaluate at high resolution.
    sample = _stage_a(ctx, ctx["metrics_downsample"])
    from experiments.common.baselines import fft_gaussian as fft

    background = fft.estimate_background(sample["observed"], 0.02)
    prediction = {
        "background_pred": background,
        "residual_pred": utils.reconstruct_residual(sample["observed"], background),
    }
    result = metrics.evaluate_synthetic(sample, prediction)
    core = (
        "bg_mae",
        "bg_rmse",
        "bg_psnr",
        "bg_ssim",
        "residual_mae",
        "residual_rmse",
        "residual_psnr",
        "residual_ssim",
        "star_f1",
    )
    for key in core:
        if key not in result:
            raise AssertionError(f"metric {key} missing")
        if not np.isfinite(result[key]):
            raise AssertionError(f"metric {key} is not finite: {result[key]}")
    # Match-dependent metrics are legitimately NaN when nothing matches.
    undefined = sorted(
        k
        for k, v in result.items()
        if isinstance(v, (int, float)) and v is not None and not np.isfinite(v)
    )
    note = f", undefined={undefined}" if undefined else ""
    return (
        f"bg_mae {result['bg_mae']:.4g}, residual_ssim "
        f"{result['residual_ssim']:.4f}, star_f1 {result['star_f1']:.3f}{note}"
    )


def test_metrics_stage_b(ctx):
    """Stage-B metric stack adds faint-star recall and protected-structure terms."""
    from experiments.common.utils import stage_b_metrics

    sample = _stage_b(ctx, ctx["metrics_downsample"])
    from experiments.common.baselines import fft_gaussian as fft

    background = fft.estimate_background(sample["observed"], 0.02)
    prediction = {
        "background_pred": background,
        "residual_pred": utils.reconstruct_residual(sample["observed"], background),
    }
    result = stage_b_metrics.evaluate_stage_b(sample, prediction)
    for key in (
        "bg_mae",
        "faint_star_recall",
        "protected_region_mae",
        "protected_projection_ratio",
    ):
        if key not in result:
            raise AssertionError(f"Stage-B metric {key} missing")
        if not np.isfinite(result[key]):
            raise AssertionError(f"Stage-B metric {key} is not finite: {result[key]}")
    return (
        f"bg_mae {result['bg_mae']:.4g}, faint_recall "
        f"{result['faint_star_recall']:.3f}, proj_ratio "
        f"{result['protected_projection_ratio']:+.4f}"
    )


def test_metrics_real(ctx):
    """Real-image descriptive metrics run without ground truth."""
    from experiments.common.utils import metrics

    sample = _stage_a(ctx, ctx["metrics_downsample"])
    from experiments.common.baselines import fft_gaussian as fft

    background = fft.estimate_background(sample["observed"], 0.02)
    residual = utils.reconstruct_residual(sample["observed"], background)
    result = metrics.evaluate_real(sample["observed"], background, residual)
    for key, value in result.items():
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise AssertionError(f"real metric {key} is not finite: {value}")
    return f"{len(result)} descriptive metrics"


def test_generators(ctx):
    """Synthetic Stage-A / Stage-B background generators produce valid truth fields."""
    from experiments.common.data import stage_b_backgrounds, synthetic_backgrounds

    produced = []
    for background_type in synthetic_backgrounds.BACKGROUND_TYPES:
        field, truth = synthetic_backgrounds.generate_background(
            64, 96, background_type, intensity=0.35, scale=0.5, seed=20260728
        )
        field = np.asarray(field, dtype=np.float32)
        _assert_finite(f"synthetic_backgrounds.{background_type}", field)
        if not isinstance(truth, dict):
            raise AssertionError(f"{background_type} generation_truth is not a dict")
        produced.append(background_type)

    for scene_type in stage_b_backgrounds.SCENE_TYPES:
        scene = stage_b_backgrounds.generate_stage_b_scene(64, 96, scene_type, "high", 20260728)
        if not isinstance(scene, (tuple, list)) or len(scene) < 2:
            raise AssertionError(f"{scene_type} scene has unexpected structure")
        for item in scene:
            if isinstance(item, np.ndarray):
                _assert_finite(f"stage_b_backgrounds.{scene_type}", item)
        produced.append(scene_type)

    return f"{len(produced)} generators: {', '.join(produced)}"


def test_data_inventory(ctx):
    """Report which formal inputs are present; always informative, never fatal."""
    required = {
        "collections manifest": utils.COLLECTION_MANIFEST,
        "Stage-A manifest": utils.SYNTHETIC_DATA_ROOT / "manifest.csv",
        "Stage-B manifest": utils.SYNTHETIC_DATA_ROOT.parent / "synthetic_stage_b" / "manifest.csv",
        "E0 native test image": utils.NATIVE_TEST_IMAGE,
        "Global configuration lock": utils.GLOBAL_CONFIGURATION_LOCK,
        "Stage-B configuration lock": utils.BASE_OUTPUT_ROOT
        / "stage_b/configuration_lock/stage_b_global_configuration_lock.json",
        "E4 real images dir": utils.REAL_HIGH_POLLUTION_ROOT,
        "camera reference candidates": utils.DATA_ROOT / "collections/clean_candidates",
    }
    present, missing = [], []
    for label, path in required.items():
        (present if Path(path).exists() else missing).append(label)

    # The smoke test itself only needs the two manifests and one sample each.
    for label in ("Stage-A manifest", "Stage-B manifest"):
        if label in missing:
            raise AssertionError(f"smoke test cannot run without {label}")

    detail = f"{len(present)}/{len(required)} formal inputs present"
    if missing:
        detail += f"; missing: {', '.join(missing)}"
    return detail


TESTS = [
    ("imports", test_imports),
    ("inventory", test_data_inventory),
    ("laplacian", test_analytic_laplacian),
    ("data", test_data_loading),
    ("fft_gaussian", test_fft_gaussian),
    ("fourier_ls", test_fourier_ls),
    ("forcing", test_forcing_construction),
    ("pinn", test_pinn),
    ("final_pinn", test_final_pinn),
    ("metrics_a", test_metrics_stage_a),
    ("metrics_b", test_metrics_stage_b),
    ("metrics_real", test_metrics_real),
    ("generators", test_generators),
]


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
def _preflight_data() -> None:
    """Fail fast with an actionable message when the smoke dataset is absent.

    Nothing here is tracked by git: the arrays are extracted from the server
    transfer archives.  A fresh clone therefore has no data, and the honest
    fix is one command rather than a confusing traceback.
    """
    from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST

    required = {
        "Stage-A manifest": utils.SYNTHETIC_DATA_ROOT / "manifest.csv",
        "Stage-B manifest": STAGE_B_MANIFEST,
        "Global configuration lock": utils.GLOBAL_CONFIGURATION_LOCK,
    }
    missing = [f"  {label}: {path}" for label, path in required.items() if not Path(path).is_file()]
    if not missing:
        return

    print("=" * 78)
    print("SMOKE DATA MISSING")
    print("=" * 78)
    print("Not found:")
    print("\n".join(missing))
    print()
    print("This dataset is NOT tracked by git; it is extracted from the server")
    print("transfer archives in artifacts/transfers/stage11_transfer_archives/.")
    print()
    print("Fix with:")
    print("    bash tools/fetch_smoke_data.sh")
    print()
    print("If that directory is absent on this machine, only the data-free tests")
    print("can run:")
    print("    python -m tools.smoke_test --only imports,laplacian,generators")
    raise SystemExit(2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12, help="PINN optimisation steps")
    parser.add_argument("--downsample", type=int, default=16, help="training image downsample")
    parser.add_argument(
        "--metrics-downsample",
        type=int,
        default=1,
        help="downsample for metrics tests (1 = full res; star matching needs this)",
    )
    parser.add_argument("--only", default="", help="comma separated test names")
    parser.add_argument("--list", action="store_true", help="list test names and exit")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.list:
        for name, fn in TESTS:
            print(f"{name:14s} {fn.__doc__.splitlines()[0]}")
        return 0

    selected = [n.strip() for n in args.only.split(",") if n.strip()]
    data_free = {"imports", "laplacian", "generators", "inventory"}
    if not (set(selected) and set(selected) <= data_free):
        _preflight_data()
    if selected:
        unknown = sorted(set(selected) - {n for n, _ in TESTS})
        if unknown:
            parser.error(f"unknown test(s): {', '.join(unknown)}")
        tests = [(n, f) for n, f in TESTS if n in selected]
    else:
        tests = TESTS

    device = torch.device(args.device)
    ctx = {
        "device": device,
        "steps": args.steps,
        "downsample": args.downsample,
        "metrics_downsample": args.metrics_downsample,
    }

    print("=" * 78)
    print("PINN-Starlight CPU smoke test (NOT a scientific result)")
    print("=" * 78)
    print(f"device            : {device}")
    print(f"pinn steps        : {args.steps}")
    print(f"train downsample  : {args.downsample}")
    print(f"metrics downsample: {args.metrics_downsample}")
    print(f"tests             : {len(tests)}")
    print("=" * 78)

    results, failures = [], []
    for name, fn in tests:
        started = time.perf_counter()
        try:
            detail = fn(ctx)
            elapsed = time.perf_counter() - started
            results.append((name, "PASS", elapsed, detail))
            print(f"[PASS] {name:14s} {elapsed:7.2f}s  {detail}")
        except Exception as exc:  # noqa: BLE001 - a smoke test must not abort early
            elapsed = time.perf_counter() - started
            tb = traceback.format_exc(limit=6)
            results.append((name, "FAIL", elapsed, f"{type(exc).__name__}: {exc}"))
            failures.append((name, tb))
            print(f"[FAIL] {name:14s} {elapsed:7.2f}s  {type(exc).__name__}: {exc}")

    print("=" * 78)
    passed = sum(1 for r in results if r[1] == "PASS")
    total_time = sum(r[2] for r in results)
    print(f"RESULT: {passed}/{len(results)} passed in {total_time:.1f}s")
    if failures:
        print()
        for name, tb in failures:
            print(f"----- {name} traceback -----")
            print(tb)
        return 1
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
