"""Frozen exploratory validation of conservative PINN decomposition.

This module is deliberately NOT a new formal Stage11 experiment runner.

Frozen scientific boundary
--------------------------
1. Stage-A and Stage-B VALIDATION rows only. Test rows are never loaded.
2. One fixed seed (20260728).
3. Four predeclared candidates + one same-budget current-method reference.
4. No parameter grid, no Pareto-front claim, no automatic winner.
5. All candidates in one run are executed on one machine/environment.
6. Runtime is logged for audit only and must not be used to select a method.
7. Formal locks, E2/E3/E4, Stage-B formal outputs and CV5 outputs are never
   overwritten.
8. The code refuses to run if the audited Stage11 scientific source files no
   longer match the source package against which this patch was written.

The experiment asks a narrow mechanism question:
    Does a conservative physics-first decomposition preserve low-frequency
    astronomical structure better than data-first initialization, while keeping
    background/photometry quality reasonably close?

It does NOT claim to solve the general multi-objective/Pareto problem.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import nnls
from torch.nn import functional as F
from tqdm import tqdm

import pinn_starlight_core.nn.pinn_loss as pinn_loss
from experiments.common.baselines import coordinate_pinn as base_pinn
from experiments.common.baselines import sgc_pinn as formal_followup
from experiments.common.baselines import cosine_ls as dct_solver
from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.common.utils import stage_b_metrics
from experiments.configs.formal_experiment import FINAL_PINN_FOLLOWUP_CONFIG
from experiments.scripts.final_pinn.common import forced_reference_metrics
from pinn_starlight_core.data.image_loader import coordinate_grid
from pinn_starlight_core.nn.pinn_layers import SkyglowMLP


PATCH_DIR = Path(__file__).resolve().parent
PATCH_MANIFEST = PATCH_DIR / "pinn_preservation_frozen_manifest.json"
A_MANIFEST = utils.SYNTHETIC_DATA_ROOT / "manifest.csv"
A_LOCK = utils.BASE_OUTPUT_ROOT / "configuration_lock/global_configuration_lock.json"
B_LOCK = (
    utils.BASE_OUTPUT_ROOT
    / "stage_b/configuration_lock/stage_b_global_configuration_lock.json"
)
OUTPUT_BASE = (
    utils.BASE_OUTPUT_ROOT
    / "diagnostics/pinn_preservation_exploratory_frozen_v2"
)

REFERENCE = "v0_current_datafirst_projected"
CANDIDATES = (
    "v1_physics_first",
    "v2_physics_first_anchor",
    "v3_dct_init",
    "v4_budgeted_conflict",
)

FIXED_SEED = 20260728

FULL_INIT_STEPS = 20000
FULL_DISTILL_STEPS = 20000
FULL_REFINE_STEPS = 5000

SMOKE_INIT_STEPS = 30
SMOKE_DISTILL_STEPS = 30
SMOKE_REFINE_STEPS = 20

PHYSICS_FIRST_DATA_WEIGHT = 0.10
PHYSICS_FIRST_PHYSICS_WEIGHT = 1.0
PHYSICS_FIRST_SPARSE_WEIGHT = 1e-3
ROBUST_BETA = 0.02
V2_ANCHOR_WEIGHT = 0.25
V4_CONFLICT_BUDGET = 0.20


def main(*, suite: str, device_name: str | None, force: bool, save_arrays: bool):
    suite = str(suite).strip().lower()
    if suite not in {"smoke", "full"}:
        raise ValueError("suite must be smoke or full")

    # This module is intentionally standard-split only. Unsetting the split env
    # in the launcher prevents an accidental CV5 context, and this assertion is
    # a second independent guard.
    if utils.SPLIT_MODE != "standard":
        raise RuntimeError(
            f"exploratory preservation probe requires standard split; got {utils.SPLIT_MODE!r}"
        )

    patch_manifest = _verify_patch_and_sources()
    _verify_formal_prerequisites()

    device = _resolve_device(device_name)
    _analytic_laplacian_self_check(device)

    if suite == "smoke":
        init_steps = SMOKE_INIT_STEPS
        distill_steps = SMOKE_DISTILL_STEPS
        refine_steps = SMOKE_REFINE_STEPS
    else:
        init_steps = FULL_INIT_STEPS
        distill_steps = FULL_DISTILL_STEPS
        refine_steps = FULL_REFINE_STEPS

    output_root = OUTPUT_BASE / suite
    if force and output_root.exists():
        shutil.rmtree(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"{output_root} is non-empty. Use --force only to replace this "
            "diagnostic output; formal Stage11 outputs are never touched."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    rows_a = utils.load_manifest(A_MANIFEST, split="validation")
    rows_b = utils.load_manifest(STAGE_B_MANIFEST, split="validation")
    _assert_validation_only(rows_a, "Stage-A")
    _assert_validation_only(rows_b, "Stage-B")

    if suite == "smoke":
        rows_a = [_choose_smoke_row(rows_a)]
        rows_b = [_choose_smoke_row(rows_b)]

    lock_a = utils.read_json(A_LOCK)
    lock_b = utils.read_json(B_LOCK)
    dct_cfg_a = dict(lock_a["methods"]["fourier_ls"])
    dct_cfg_b = dict(lock_b["methods"]["fourier_ls"])

    followup = deepcopy(FINAL_PINN_FOLLOWUP_CONFIG)
    data_cfg = dict(base_pinn.normalized_config(followup["stage1"]))
    data_cfg["steps"] = int(init_steps)
    data_cfg["physics_weight"] = 0.0
    data_cfg["sparse_weight"] = 0.0
    stage2_cfg = dict(followup["stage2"])
    stage3_cfg = dict(followup["stage3"])
    stage3_cfg["steps"] = int(refine_steps)

    run_metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "scientific_status": "posthoc_exploratory_mechanism_validation",
        "formal_result_replacement": False,
        "suite": suite,
        "split": "validation_only",
        "test_arrays_loaded": False,
        "seed": FIXED_SEED,
        "runtime_is_selection_metric": False,
        "pareto_optimization_claim": False,
        "parameter_grid_search": False,
        "device": str(device),
        "budget": {
            "init_steps": init_steps,
            "dct_distill_steps": distill_steps,
            "refine_steps": refine_steps,
        },
        "fixed_hyperparameters": {
            "physics_first_data_weight": PHYSICS_FIRST_DATA_WEIGHT,
            "physics_first_physics_weight": PHYSICS_FIRST_PHYSICS_WEIGHT,
            "physics_first_sparse_weight": PHYSICS_FIRST_SPARSE_WEIGHT,
            "robust_beta": ROBUST_BETA,
            "v2_anchor_weight": V2_ANCHOR_WEIGHT,
            "v4_conflict_budget": V4_CONFLICT_BUDGET,
        },
        "reference": REFERENCE,
        "candidates": list(CANDIDATES),
        "stage_a_validation_ids": [r["sample_id"] for r in rows_a],
        "stage_b_validation_ids": [r["sample_id"] for r in rows_b],
        "patch_manifest": patch_manifest,
    }
    _write_json(output_root / "run_metadata.json", run_metadata)
    _write_json(output_root / "platform.json", _platform_record(device))

    final_rows: list[dict] = []
    init_rows: list[dict] = []

    jobs = [("a", r) for r in rows_a] + [("b", r) for r in rows_b]
    print("=" * 92)
    print("Frozen exploratory PINN preservation validation")
    print("VALIDATION ONLY — Stage-A/B test rows are not opened")
    print("No Pareto claim; no parameter search; no automatic winner")
    print(f"suite   : {suite}")
    print(f"device  : {device}")
    print(f"seed    : {FIXED_SEED}")
    print(f"A rows  : {len(rows_a)}")
    print(f"B rows  : {len(rows_b)}")
    print(f"output  : {output_root}")
    print("=" * 92)

    for job_index, (stage, row) in enumerate(jobs, 1):
        sample = _load_sample(stage, row)
        print()
        print(
            f"[{job_index}/{len(jobs)}] Stage-{stage.upper()} "
            f"{sample['sample_id']} | {sample['background_type']} | "
            f"{sample['intensity_level']}"
        )

        initial_centers = base_pinn.initial_centers_from_observed(
            sample["observed"], data_cfg
        )

        # ==============================================================
        # Shared branch D: current data-first initialisation.
        # V0 and V4 start from EXACTLY the same model state and forcing fit.
        # ==============================================================
        data_init = base_pinn.train_background(
            sample["observed"],
            config=data_cfg,
            device=device,
            seed=FIXED_SEED,
            show_progress=True,
            return_state=True,
            laplacian_backend="analytic",
            initial_centers=initial_centers,
        )
        data_state = data_init["state"]["model_state"]
        init_rows.append(
            _make_init_row(
                stage,
                sample,
                "data_first",
                data_init["background_pred"],
                float(data_init["runtime_s"]),
            )
        )

        data_forcing_state, data_forcing_diag = formal_followup.fit_fixed_forcing(
            observed=sample["observed"],
            model_state=data_state,
            stage1_config=data_cfg,
            stage2_config=stage2_cfg,
            initial_centers=initial_centers,
            device=device,
            seed=FIXED_SEED + 1009,
        )

        # Same-budget reference = current final method execution logic.
        v0 = formal_followup.finetune(
            observed=sample["observed"],
            model_state=data_state,
            forcing_state=data_forcing_state,
            stage1_config=data_cfg,
            stage3_config=stage3_cfg,
            initial_centers=initial_centers,
            device=device,
            seed=FIXED_SEED + 2003,
            use_physics=True,
            show_progress=True,
        )
        final_rows.append(
            _save_final(
                output_root, stage, sample, REFERENCE, v0,
                init_runtime_s=float(data_init["runtime_s"]),
                forcing_diag=data_forcing_diag,
                save_arrays=save_arrays,
            )
        )

        v4 = _custom_refine(
            observed=sample["observed"],
            model_state=data_state,
            forcing_state=data_forcing_state,
            base_config=data_cfg,
            stage3_config=stage3_cfg,
            initial_centers=initial_centers,
            device=device,
            seed=FIXED_SEED + 2003,
            mode="budgeted_conflict",
            anchor_background=None,
            show_progress=True,
        )
        final_rows.append(
            _save_final(
                output_root, stage, sample, "v4_budgeted_conflict", v4,
                init_runtime_s=float(data_init["runtime_s"]),
                forcing_diag=data_forcing_diag,
                save_arrays=save_arrays,
            )
        )

        # ==============================================================
        # Shared branch P: conservative physics-first initialisation.
        # V1 and V2 share this entire state and forcing identification.
        # ==============================================================
        physics_init = _train_physics_first(
            observed=sample["observed"],
            base_config=data_cfg,
            initial_centers=initial_centers,
            device=device,
            seed=FIXED_SEED,
            steps=init_steps,
            show_progress=True,
        )
        physics_state = physics_init["model_state"]
        init_rows.append(
            _make_init_row(
                stage,
                sample,
                "physics_first",
                physics_init["background_pred"],
                float(physics_init["runtime_s"]),
            )
        )

        physics_forcing_state, physics_forcing_diag = _fit_forcing_keep_geometry(
            observed=sample["observed"],
            model_state=physics_state,
            learned_forcing_state=physics_init["forcing_state"],
            base_config=data_cfg,
            stage2_config=stage2_cfg,
            initial_centers=initial_centers,
            device=device,
            seed=FIXED_SEED + 1009,
        )

        v1 = formal_followup.finetune(
            observed=sample["observed"],
            model_state=physics_state,
            forcing_state=physics_forcing_state,
            stage1_config=data_cfg,
            stage3_config=stage3_cfg,
            initial_centers=initial_centers,
            device=device,
            seed=FIXED_SEED + 2003,
            use_physics=True,
            show_progress=True,
        )
        final_rows.append(
            _save_final(
                output_root, stage, sample, "v1_physics_first", v1,
                init_runtime_s=float(physics_init["runtime_s"]),
                forcing_diag=physics_forcing_diag,
                save_arrays=save_arrays,
            )
        )

        v2 = _custom_refine(
            observed=sample["observed"],
            model_state=physics_state,
            forcing_state=physics_forcing_state,
            base_config=data_cfg,
            stage3_config=stage3_cfg,
            initial_centers=initial_centers,
            device=device,
            seed=FIXED_SEED + 2003,
            mode="projected_with_anchor",
            anchor_background=physics_init["background_pred"],
            show_progress=True,
        )
        final_rows.append(
            _save_final(
                output_root, stage, sample, "v2_physics_first_anchor", v2,
                init_runtime_s=float(physics_init["runtime_s"]),
                forcing_diag=physics_forcing_diag,
                save_arrays=save_arrays,
            )
        )

        # ==============================================================
        # Branch C: selected screened-Poisson DCT-II initialisation.
        # ==============================================================
        dct_cfg = dct_cfg_a if stage == "a" else dct_cfg_b
        dct_started = time.perf_counter()
        dct_background = dct_solver.estimate_background(
            sample["observed"],
            lambda_ls=float(dct_cfg["lambda_ls"]),
            sparse_weight=float(dct_cfg["sparse_weight"]),
            return_info=False,
            show_progress=False,
            backend=("cuda" if device.type == "cuda" else "cpu"),
        )
        dct_solver_runtime = time.perf_counter() - dct_started

        dct_init = _distill_background(
            target_background=dct_background,
            base_config=data_cfg,
            device=device,
            seed=FIXED_SEED,
            steps=distill_steps,
            show_progress=True,
        )
        dct_state = dct_init["model_state"]
        init_rows.append(
            _make_init_row(
                stage,
                sample,
                "dct_init",
                dct_init["background_pred"],
                float(dct_solver_runtime + dct_init["runtime_s"]),
            )
        )

        dct_forcing_state, dct_forcing_diag = formal_followup.fit_fixed_forcing(
            observed=sample["observed"],
            model_state=dct_state,
            stage1_config=data_cfg,
            stage2_config=stage2_cfg,
            initial_centers=initial_centers,
            device=device,
            seed=FIXED_SEED + 1009,
        )
        v3 = formal_followup.finetune(
            observed=sample["observed"],
            model_state=dct_state,
            forcing_state=dct_forcing_state,
            stage1_config=data_cfg,
            stage3_config=stage3_cfg,
            initial_centers=initial_centers,
            device=device,
            seed=FIXED_SEED + 2003,
            use_physics=True,
            show_progress=True,
        )
        final_rows.append(
            _save_final(
                output_root, stage, sample, "v3_dct_init", v3,
                init_runtime_s=float(dct_solver_runtime + dct_init["runtime_s"]),
                forcing_diag=dct_forcing_diag,
                save_arrays=save_arrays,
            )
        )

        # Flush after every sample so an interruption still leaves an auditable
        # partial diagnostic record.
        _write_all_tables(output_root, final_rows, init_rows)

        del (
            data_init, v0, v4, physics_init, v1, v2,
            dct_background, dct_init, v3
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_all_tables(output_root, final_rows, init_rows)
    print()
    print("=" * 92)
    print("Exploratory validation complete.")
    print("No automatic method winner has been declared.")
    print("Please interpret candidate_summary.csv + paired_vs_reference.csv +")
    print("initialization_summary.csv together.")
    print("=" * 92)


def _verify_patch_and_sources():
    if not PATCH_MANIFEST.is_file():
        raise FileNotFoundError(PATCH_MANIFEST)
    patch = json.loads(PATCH_MANIFEST.read_text(encoding="utf-8"))
    if patch.get("test_split_allowed") is not False:
        raise RuntimeError("patch manifest does not freeze test_split_allowed=False")
    expected = dict(patch.get("audited_source_sha256") or {})
    if not expected:
        raise RuntimeError("patch manifest has no audited source hashes")
    mismatches = []
    for rel, expected_hash in expected.items():
        path = utils.PROJECT_ROOT / rel
        if not path.is_file():
            mismatches.append(f"missing: {rel}")
            continue
        actual = _sha256(path)
        if actual != expected_hash:
            mismatches.append(
                f"hash mismatch: {rel}\n"
                f"  expected {expected_hash}\n"
                f"  actual   {actual}"
            )
    if mismatches:
        raise RuntimeError(
            "Stage11 scientific source differs from the version audited for this "
            "patch. Do not guess or continue.\n" + "\n".join(mismatches)
        )
    return patch


def _verify_formal_prerequisites():
    # Stage-A verifier checks the current formal config, manifest fingerprint and
    # frozen method configuration without opening test image arrays.
    utils.load_global_configuration_lock(
        A_LOCK, verify=True, require_formal_ready=True
    )
    if not B_LOCK.is_file():
        raise FileNotFoundError(
            "Stage-B Global Lock is required before this diagnostic: " + str(B_LOCK)
        )
    b_lock = utils.read_json(B_LOCK)
    if b_lock.get("lock_state") != "FROZEN" or not bool(b_lock.get("formal_test_ready")):
        raise RuntimeError("Stage-B Global Lock is not frozen/formal-test-ready")
    if "fourier_ls" not in (b_lock.get("methods") or {}):
        raise RuntimeError("Stage-B lock is missing the selected Fourier-LS config")


def _choose_smoke_row(rows):
    high = [
        r for r in rows
        if str(r.get("intensity_level", "")).strip().lower() == "high"
    ]
    return (high or list(rows))[0]


def _assert_validation_only(rows, label):
    if not rows:
        raise RuntimeError(f"{label} validation split is empty")
    bad = [r.get("sample_id") for r in rows if r.get("split") != "validation"]
    if bad:
        raise RuntimeError(
            f"{label} contains non-validation rows; refusing to continue: {bad}"
        )


def _load_sample(stage, row):
    # Common loader does load truth because truth is needed later for evaluation.
    # Crucially, training receives ONLY sample["observed"]; the truth arrays are
    # never passed into candidate optimisation functions.
    sample = utils.load_synthetic_sample(row)
    if sample["split"] != "validation":
        raise RuntimeError("non-validation sample reached loader")
    if stage == "b":
        sample["base_clean"] = utils.load_float_tiff(row["base_clean"])
        sample["protected_true"] = utils.load_float_tiff(row["protected_true"])
        sample["scene_type"] = row.get("scene_type", row["background_type"])
    return sample


def _train_physics_first(
    *,
    observed,
    base_config,
    initial_centers,
    device,
    seed,
    steps,
    show_progress,
):
    """Conservative physics-dominated initial decomposition.

    The weak Smooth-L1 observation term merely anchors overall brightness.
    Physics + sparse forcing carry the decomposition bias.  No truth array is
    available inside this function.
    """
    utils.set_seed(int(seed))
    utils.reset_peak_vram(device)
    observed = utils.load_observed(observed)

    height, width = observed.shape
    coords = coordinate_grid(height, width, device)
    values = torch.from_numpy(observed.reshape(-1)).to(device)

    model = SkyglowMLP(base_config["hidden_dims"]).to(device)
    forcing = base_pinn._build_forcing(
        observed,
        base_config,
        device,
        initial_centers=initial_centers,
    ).to(device)

    optimizer = torch.optim.Adam(
        [
            {"params": model.parameters(), "lr": float(base_config["model_lr"])},
            {"params": forcing.parameters(), "lr": float(base_config["forcing_lr"])},
        ]
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    alpha = torch.tensor(
        float(base_config["alpha"]), dtype=torch.float32, device=device
    )
    batch_size = min(int(base_config["batch_size"]), int(coords.shape[0]))
    log_every = max(1, int(base_config["log_every"]))
    warmup = min(
        int(base_config["center_warmup_steps"]),
        max(0, int(steps) // 3),
    )

    history = []
    started = time.perf_counter()
    iterator = range(1, int(steps) + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="physics-first init", leave=False)

    for step in iterator:
        forcing.set_center_gradient_enabled(
            bool(base_config["center_refinement_enabled"]) and step > warmup
        )
        indices = torch.randint(
            0, coords.shape[0], (batch_size,),
            device=device, generator=generator
        )
        xy = coords[indices]
        target = values[indices]

        background, laplacian = base_pinn._background_and_laplacian(model, xy)
        robust_data = F.smooth_l1_loss(
            background, target, beta=ROBUST_BETA, reduction="mean"
        )
        forcing_pred = forcing(xy, alpha)
        physics = pinn_loss.mse_physics_from_laplacian(
            background, laplacian, forcing_pred, alpha
        )
        sparse = forcing.sparse_penalty()

        total = (
            PHYSICS_FIRST_DATA_WEIGHT * robust_data
            + PHYSICS_FIRST_PHYSICS_WEIGHT * physics
            + PHYSICS_FIRST_SPARSE_WEIGHT * sparse
        )

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == int(steps):
            row = {
                "step": int(step),
                "total_loss": float(total.detach().item()),
                "robust_data_loss": float(robust_data.detach().item()),
                "physics_loss": float(physics.detach().item()),
                "sparse_loss": float(sparse.detach().item()),
                "data_weight": PHYSICS_FIRST_DATA_WEIGHT,
                "physics_weight": PHYSICS_FIRST_PHYSICS_WEIGHT,
                "sparse_weight": PHYSICS_FIRST_SPARSE_WEIGHT,
            }
            history.append(row)
            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    robust=f"{row['robust_data_loss']:.2e}",
                    physics=f"{row['physics_loss']:.2e}",
                )

    runtime_s = time.perf_counter() - started
    background_pred = base_pinn._predict_background(
        model, coords, height, width, int(base_config["prediction_batch_size"])
    )
    return {
        "background_pred": np.asarray(background_pred, dtype=np.float32),
        "model_state": _cpu_state(model.state_dict()),
        "forcing_state": _cpu_state(forcing.state_dict()),
        "history": history,
        "runtime_s": float(runtime_s),
        "peak_vram_mb": utils.peak_vram_mb(device),
    }


def _fit_forcing_keep_geometry(
    *,
    observed,
    model_state,
    learned_forcing_state,
    base_config,
    stage2_config,
    initial_centers,
    device,
    seed,
):
    """Refit non-negative forcing amplitudes while retaining learned geometry."""
    height, width = observed.shape
    all_coords = coordinate_grid(height, width, device)
    n_total = int(all_coords.shape[0])
    n_pick = min(int(stage2_config["prefit_points"]), n_total)

    rng = np.random.default_rng(int(seed))
    selected_np = rng.choice(n_total, size=n_pick, replace=False)
    rng.shuffle(selected_np)
    n_fit = int(round(float(stage2_config["fit_fraction"]) * n_pick))
    n_fit = min(max(n_fit, 3), n_pick - 1)

    selected = torch.as_tensor(selected_np, dtype=torch.long, device=device)
    coords = all_coords[selected]

    model = SkyglowMLP(base_config["hidden_dims"]).to(device)
    model.load_state_dict(model_state)
    model.eval()
    alpha = torch.tensor(
        float(base_config["alpha"]), dtype=torch.float32, device=device
    )

    q_parts = []
    with torch.no_grad():
        for start in range(0, coords.shape[0], 8192):
            xy = coords[start:start + 8192]
            background, laplacian = base_pinn._background_and_laplacian(model, xy)
            q_parts.append((alpha * background - laplacian).detach())
    q_all = torch.cat(q_parts, dim=0)

    forcing = base_pinn._build_forcing(
        observed, base_config, device, initial_centers=initial_centers
    ).to(device)
    forcing.load_state_dict(learned_forcing_state)
    forcing.set_center_gradient_enabled(False)

    with torch.no_grad():
        components = forcing._unnormalized_components(coords, alpha)
        basis = components / forcing.component_rms(alpha).unsqueeze(0)

    a_fit = basis[:n_fit].detach().cpu().double().numpy()
    y_fit = q_all[:n_fit].detach().cpu().double().numpy()
    a_check = basis[n_fit:].detach().cpu().double().numpy()
    y_check = q_all[n_fit:].detach().cpu().double().numpy()

    weights, residual_norm = nnls(a_fit, y_fit)
    formal_followup._set_forcing_weights(forcing, weights)

    fit_pred = a_fit @ weights
    check_pred = a_check @ weights
    diagnostics = {
        "mode": "physics_first_geometry_then_nonnegative_amplitude_refit",
        "geometry_source": "physics_first_initialization",
        "prefit_points": int(n_pick),
        "fit_points": int(n_fit),
        "diagnostic_check_points": int(n_pick - n_fit),
        "weights": [float(x) for x in weights],
        "nnls_residual_norm": float(residual_norm),
        "fit_rmse": _rmse(y_fit, fit_pred),
        "diagnostic_check_rmse": _rmse(y_check, check_pred),
        "fit_relative_rmse": _relative_rmse(y_fit, fit_pred),
        "diagnostic_check_relative_rmse": _relative_rmse(y_check, check_pred),
        "diagnostic_check_correlation": _safe_corr(y_check, check_pred),
        "uses_true_background": False,
        "components": forcing.diagnostics(
            alpha,
            inactive_relative_threshold=float(
                base_config["inactive_relative_threshold"]
            ),
        ),
    }
    return _cpu_state(forcing.state_dict()), diagnostics


def _distill_background(
    *,
    target_background,
    base_config,
    device,
    seed,
    steps,
    show_progress,
):
    """Distil a selected DCT-II screened-Poisson background into the same MLP."""
    target = np.asarray(target_background, dtype=np.float32)
    if target.ndim != 2 or not np.isfinite(target).all():
        raise ValueError("DCT target must be a finite 2-D array")

    utils.set_seed(int(seed))
    utils.reset_peak_vram(device)
    height, width = target.shape
    coords = coordinate_grid(height, width, device)
    values = torch.from_numpy(target.reshape(-1)).to(device)

    model = SkyglowMLP(base_config["hidden_dims"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(base_config["model_lr"])
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    batch_size = min(int(base_config["batch_size"]), int(coords.shape[0]))
    log_every = max(1, int(base_config["log_every"]))

    history = []
    started = time.perf_counter()
    iterator = range(1, int(steps) + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="DCT -> MLP distill", leave=False)

    for step in iterator:
        indices = torch.randint(
            0, coords.shape[0], (batch_size,),
            device=device, generator=generator
        )
        xy = coords[indices]
        target_batch = values[indices]
        background = torch.sigmoid(model(xy).squeeze(-1))
        loss = pinn_loss.mse_data(target_batch, background)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == int(steps):
            value = float(loss.detach().item())
            history.append({"step": int(step), "distillation_mse": value})
            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(mse=f"{value:.2e}")

    runtime_s = time.perf_counter() - started
    background_pred = base_pinn._predict_background(
        model, coords, height, width, int(base_config["prediction_batch_size"])
    )
    return {
        "background_pred": np.asarray(background_pred, dtype=np.float32),
        "model_state": _cpu_state(model.state_dict()),
        "history": history,
        "runtime_s": float(runtime_s),
        "peak_vram_mb": utils.peak_vram_mb(device),
    }


def _custom_refine(
    *,
    observed,
    model_state,
    forcing_state,
    base_config,
    stage3_config,
    initial_centers,
    device,
    seed,
    mode,
    anchor_background,
    show_progress,
):
    if mode not in {"projected_with_anchor", "budgeted_conflict"}:
        raise ValueError(mode)

    utils.set_seed(int(seed))
    utils.reset_peak_vram(device)
    observed = utils.load_observed(observed)
    height, width = observed.shape
    coords = coordinate_grid(height, width, device)
    values = torch.from_numpy(observed.reshape(-1)).to(device)

    anchor_values = None
    if mode == "projected_with_anchor":
        if anchor_background is None:
            raise ValueError("V2 requires anchor_background")
        anchor = np.asarray(anchor_background, dtype=np.float32)
        if anchor.shape != observed.shape:
            raise ValueError("anchor shape mismatch")
        anchor_values = torch.from_numpy(anchor.reshape(-1)).to(device)

    model = SkyglowMLP(base_config["hidden_dims"]).to(device)
    model.load_state_dict(model_state)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]

    forcing = base_pinn._build_forcing(
        observed, base_config, device, initial_centers=initial_centers
    ).to(device)
    forcing.load_state_dict(forcing_state)
    forcing.set_center_gradient_enabled(False)
    forcing.eval()
    for p in forcing.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(
        params, lr=float(stage3_config["model_lr"])
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    alpha = torch.tensor(
        float(base_config["alpha"]), dtype=torch.float32, device=device
    )
    steps = int(stage3_config["steps"])
    batch_size = min(int(base_config["batch_size"]), int(coords.shape[0]))
    log_every = max(1, int(stage3_config["log_every"]))
    rho = float(stage3_config["rho"])
    lambda_min = float(stage3_config["lambda_min"])
    lambda_max = float(stage3_config["lambda_max"])
    ramp_steps = max(1, int(stage3_config["ramp_steps"]))

    history = []
    logged = conflicts = capped = 0
    lambdas = []
    ratios = []
    cosines = []

    started = time.perf_counter()
    iterator = range(1, steps + 1)
    if show_progress:
        label = (
            "V2 anchored refinement"
            if mode == "projected_with_anchor"
            else "V4 budgeted conflict"
        )
        iterator = tqdm(iterator, desc=label, leave=False)

    final_data = final_phys = final_anchor = float("nan")

    for step in iterator:
        indices = torch.randint(
            0, coords.shape[0], (batch_size,),
            device=device, generator=generator
        )
        xy = coords[indices]
        target = values[indices]

        background, laplacian = base_pinn._background_and_laplacian(model, xy)
        data_loss = pinn_loss.mse_data(target, background)

        if mode == "projected_with_anchor":
            anchor_loss = pinn_loss.mse_data(anchor_values[indices], background)
            primary_loss = data_loss + V2_ANCHOR_WEIGHT * anchor_loss
        else:
            anchor_loss = torch.zeros((), dtype=torch.float32, device=device)
            primary_loss = data_loss

        forcing_pred = forcing(xy, alpha)
        physics_loss = pinn_loss.mse_physics_from_laplacian(
            background, laplacian, forcing_pred, alpha
        )

        primary_grads = torch.autograd.grad(
            primary_loss, params, retain_graph=True, create_graph=False
        )
        physics_grads = torch.autograd.grad(
            physics_loss, params, retain_graph=False, create_graph=False
        )

        primary_norm_sq = _sum_sq(primary_grads)
        physics_norm_sq = _sum_sq(physics_grads)
        dot = _dot(primary_grads, physics_grads)
        eps = torch.tensor(1e-24, dtype=torch.float64, device=device)
        primary_norm = torch.sqrt(primary_norm_sq + eps)
        physics_norm = torch.sqrt(physics_norm_sq + eps)
        raw_cos = dot / (primary_norm * physics_norm + eps)
        conflict = bool((dot < 0.0).detach().item())

        if mode == "projected_with_anchor":
            coeff = torch.minimum(
                dot / (primary_norm_sq + eps),
                torch.zeros((), dtype=torch.float64, device=device),
            )
            used_physics_grads = [
                gp - coeff.to(gp.dtype) * gd
                for gd, gp in zip(primary_grads, physics_grads)
            ]
            used_norm = torch.sqrt(_sum_sq(used_physics_grads) + eps)
            lambda_eff = rho * primary_norm / (used_norm + eps)
            lambda_eff = torch.clamp(
                lambda_eff, min=lambda_min, max=lambda_max
            )
            lambda_use = lambda_eff * min(1.0, float(step) / ramp_steps)
            cap_applied = False
        else:
            used_physics_grads = physics_grads
            used_norm = physics_norm
            lambda_eff = rho * primary_norm / (used_norm + eps)
            lambda_eff = torch.clamp(
                lambda_eff, min=lambda_min, max=lambda_max
            )
            lambda_use = lambda_eff * min(1.0, float(step) / ramp_steps)
            cap_applied = False
            if conflict:
                # Guarantees:
                # gd · (gd + lambda gp) >= (1-budget) ||gd||^2
                cap_value = (
                    V4_CONFLICT_BUDGET * primary_norm_sq / (-dot + eps)
                )
                cap_value = torch.clamp(cap_value, min=0.0)
                before = lambda_use
                lambda_use = torch.minimum(lambda_use, cap_value)
                cap_applied = bool((lambda_use < before).detach().item())

        optimizer.zero_grad(set_to_none=True)
        scale = lambda_use.to(params[0].dtype)
        for p, gp, gr in zip(params, primary_grads, used_physics_grads):
            p.grad = (gp + scale * gr).detach()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            pn = float(primary_norm.detach().item())
            un = float(used_norm.detach().item())
            lam = float(lambda_use.detach().item())
            cosine = float(raw_cos.detach().item())
            ratio = lam * un / max(pn, 1e-30)

            final_data = float(data_loss.detach().item())
            final_phys = float(physics_loss.detach().item())
            final_anchor = float(anchor_loss.detach().item())
            history.append(
                {
                    "step": int(step),
                    "mode": mode,
                    "data_loss": final_data,
                    "anchor_loss": final_anchor,
                    "anchor_weight": (
                        V2_ANCHOR_WEIGHT
                        if mode == "projected_with_anchor" else 0.0
                    ),
                    "physics_loss": final_phys,
                    "primary_grad_norm": pn,
                    "raw_physics_grad_norm": float(
                        physics_norm.detach().item()
                    ),
                    "used_physics_grad_norm": un,
                    "raw_grad_cosine": cosine,
                    "lambda_use": lam,
                    "scaled_physics_to_primary_ratio": ratio,
                    "gradient_conflict": conflict,
                    "conflict_budget": (
                        V4_CONFLICT_BUDGET
                        if mode == "budgeted_conflict" else float("nan")
                    ),
                    "budget_cap_applied": cap_applied,
                }
            )
            logged += 1
            conflicts += int(conflict)
            capped += int(cap_applied)
            lambdas.append(lam)
            ratios.append(ratio)
            cosines.append(cosine)

            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(
                    data=f"{final_data:.2e}",
                    phys=f"{final_phys:.2e}",
                    cos=f"{cosine:.2f}",
                )

    runtime_s = time.perf_counter() - started
    background_pred = base_pinn._predict_background(
        model, coords, height, width, int(base_config["prediction_batch_size"])
    )
    return {
        "background_pred": np.asarray(background_pred, dtype=np.float32),
        "history": history,
        "stage3_runtime_s": float(runtime_s),
        "peak_vram_mb": utils.peak_vram_mb(device),
        "gradient_conflict_fraction": (
            float(conflicts / logged) if logged else float("nan")
        ),
        "budget_cap_fraction": (
            float(capped / logged) if logged else 0.0
        ),
        "mean_lambda_use": (
            float(np.mean(lambdas)) if lambdas else float("nan")
        ),
        "mean_scaled_physics_to_primary_ratio": (
            float(np.mean(ratios)) if ratios else float("nan")
        ),
        "mean_raw_grad_cosine": (
            float(np.mean(cosines)) if cosines else float("nan")
        ),
        "final_data_loss": final_data,
        "final_physics_loss": final_phys,
        "final_anchor_loss": final_anchor,
    }


def _make_init_row(stage, sample, name, background_pred, runtime_s):
    scores = _evaluate(stage, sample, background_pred)
    return {
        "stage": stage,
        "split": "validation",
        "initialization": name,
        "seed": FIXED_SEED,
        "sample_id": sample["sample_id"],
        "base_id": sample["base_id"],
        "background_type": sample["background_type"],
        "intensity_level": sample["intensity_level"],
        "runtime_s_audit_only": float(runtime_s),
        **scores,
    }


def _save_final(
    output_root,
    stage,
    sample,
    method,
    result,
    *,
    init_runtime_s,
    forcing_diag,
    save_arrays,
):
    scores = _evaluate(stage, sample, result["background_pred"])
    row = {
        "stage": stage,
        "split": "validation",
        "method": method,
        "seed": FIXED_SEED,
        "sample_id": sample["sample_id"],
        "base_id": sample["base_id"],
        "background_type": sample["background_type"],
        "intensity_level": sample["intensity_level"],
        "init_runtime_s_audit_only": float(init_runtime_s),
        "stage3_runtime_s_audit_only": float(
            result.get("stage3_runtime_s", float("nan"))
        ),
        "runtime_is_selection_metric": False,
        **{
            k: v
            for k, v in result.items()
            if k in {
                "gradient_conflict_fraction",
                "budget_cap_fraction",
                "mean_lambda_use",
                "mean_scaled_physics_to_primary_ratio",
                "mean_raw_grad_cosine",
                "final_data_loss",
                "final_physics_loss",
                "final_anchor_loss",
            }
        },
        **scores,
    }

    sample_dir = (
        output_root / f"stage_{stage}" / method / sample["sample_id"]
    )
    sample_dir.mkdir(parents=True, exist_ok=True)
    _write_json(sample_dir / "final_metrics.json", row)
    _write_json(sample_dir / "forcing_diagnostics.json", forcing_diag)
    _write_rows(
        sample_dir / "training_history.csv",
        list(result.get("history") or []),
    )

    if save_arrays:
        residual = utils.reconstruct_residual(
            sample["observed"], result["background_pred"]
        )
        utils.save_prediction_arrays(
            sample_dir,
            sample["observed"],
            result["background_pred"],
            residual,
        )
    return row


def _evaluate(stage, sample, background_pred):
    background = np.asarray(background_pred, dtype=np.float32)
    residual = utils.reconstruct_residual(sample["observed"], background)
    prediction = {
        "background_pred": background,
        "residual_pred": residual,
    }
    if stage == "a":
        result = metrics.evaluate_synthetic(sample, prediction)
    else:
        result = stage_b_metrics.evaluate_stage_b(sample, prediction)
    result.update(forced_reference_metrics(sample, residual))
    return result


def _write_all_tables(output_root, final_rows, init_rows):
    _write_rows(output_root / "metrics.csv", final_rows)
    _write_rows(output_root / "initialization_metrics.csv", init_rows)

    final_summary = _group_summary(
        final_rows, group_keys=("stage", "method")
    )
    init_summary = _group_summary(
        init_rows, group_keys=("stage", "initialization")
    )
    paired = _paired_vs_reference(final_rows)

    _write_rows(output_root / "candidate_summary.csv", final_summary)
    _write_rows(output_root / "initialization_summary.csv", init_summary)
    _write_rows(output_root / "paired_vs_reference.csv", paired)
    _write_mechanism_report(
        output_root, final_summary, init_summary, paired
    )


def _group_summary(rows, *, group_keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    excluded = {
        "seed",
        "runtime_is_selection_metric",
    }
    out = []
    for key, group in sorted(groups.items()):
        item = {
            k: v for k, v in zip(group_keys, key)
        }
        item["sample_count"] = len(group)
        numeric_keys = sorted(
            {
                k for row in group for k, v in row.items()
                if k not in excluded and isinstance(v, (int, float))
            }
        )
        for metric in numeric_keys:
            vals = []
            for row in group:
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    vals.append(value)
            if vals:
                item[f"mean_{metric}"] = float(np.mean(vals))
                item[f"median_{metric}"] = float(np.median(vals))
        out.append(item)
    return out


def _paired_vs_reference(rows):
    by_key = {
        (r["method"], r["stage"], r["sample_id"]): r
        for r in rows
    }
    lower_better = {
        "bg_mae": True,
        "bg_rmse": True,
        "bg_psnr": False,
        "bg_ssim": False,
        "residual_ssim": False,
        "star_f1": False,
        "matched_star_mean_relative_flux_error": True,
        "forced_reference_mean_relative_flux_error": True,
        "forced_reference_p95_relative_flux_error": True,
        "faint_star_recall": False,
        "protected_region_mae": True,
        "protected_projection_abs_error": True,
    }
    metrics_to_compare = list(lower_better)
    out = []
    stages = sorted({r["stage"] for r in rows})

    for stage in stages:
        sample_ids = sorted(
            {r["sample_id"] for r in rows if r["stage"] == stage}
        )
        for candidate in CANDIDATES:
            for metric in metrics_to_compare:
                deltas = []
                wins = losses = ties = 0
                for sample_id in sample_ids:
                    a = by_key.get((candidate, stage, sample_id))
                    b = by_key.get((REFERENCE, stage, sample_id))
                    if not a or not b:
                        continue
                    try:
                        av = float(a[metric])
                        bv = float(b[metric])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if not np.isfinite(av) or not np.isfinite(bv):
                        continue
                    delta = av - bv
                    deltas.append(delta)
                    tol = max(1e-12, 1e-8 * max(abs(av), abs(bv), 1.0))
                    if abs(delta) <= tol:
                        ties += 1
                    elif (delta < 0 if lower_better[metric] else delta > 0):
                        wins += 1
                    else:
                        losses += 1
                if deltas:
                    out.append(
                        {
                            "stage": stage,
                            "candidate": candidate,
                            "baseline": REFERENCE,
                            "metric": metric,
                            "better_direction": (
                                "lower" if lower_better[metric] else "higher"
                            ),
                            "valid_pairs": len(deltas),
                            "wins": wins,
                            "losses": losses,
                            "ties": ties,
                            "mean_delta_candidate_minus_reference": float(
                                np.mean(deltas)
                            ),
                            "median_delta_candidate_minus_reference": float(
                                np.median(deltas)
                            ),
                        }
                    )

        # Protected projection is special: the scientifically meaningful target
        # is 1, so compare abs error above; report raw ratios descriptively only.
    return out


def _write_mechanism_report(output_root, final_summary, init_summary, paired):
    lines = [
        "PINN preservation exploratory mechanism report",
        "================================================",
        "",
        "STATUS",
        "- post-hoc exploratory validation",
        "- validation split only",
        "- one fixed seed",
        "- no parameter grid",
        "- no Pareto-front claim",
        "- no automatic winner",
        "- runtime is audit-only",
        "",
        "PRIMARY MECHANISM QUESTION",
        "Does conservative physics-first initialization retain more Stage-B",
        "protected low-frequency structure before refinement than data-first",
        "initialization, and can that advantage survive refinement without a",
        "large loss of Stage-A background/photometry quality?",
        "",
        "INITIALIZATION SUMMARY",
    ]

    for row in init_summary:
        if row.get("stage") != "b":
            continue
        lines.append(
            "- Stage-B "
            + str(row["initialization"])
            + ": protected_projection_ratio="
            + _fmt(row.get("mean_protected_projection_ratio"))
            + ", projection_abs_error="
            + _fmt(row.get("mean_protected_projection_abs_error"))
            + ", bg_mae="
            + _fmt(row.get("mean_bg_mae"))
        )

    lines += ["", "FINAL CANDIDATE SUMMARY"]
    for row in final_summary:
        lines.append(
            "- Stage-"
            + str(row["stage"]).upper()
            + " "
            + str(row["method"])
            + ": bg_mae="
            + _fmt(row.get("mean_bg_mae"))
            + ", forced_flux="
            + _fmt(
                row.get(
                    "mean_forced_reference_mean_relative_flux_error"
                )
            )
            + ", protected_projection_ratio="
            + _fmt(row.get("mean_protected_projection_ratio"))
        )

    lines += [
        "",
        "INTERPRETATION LIMIT",
        "These numbers may support or reject the initialization-order hypothesis.",
        "They do not establish a Pareto optimum and do not replace the frozen",
        "Stage11 formal test/CV5 evidence. A later A+B confirmation, if run,",
        "must be a separately frozen step after this validation-only exploration.",
    ]

    (output_root / "mechanism_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _platform_record(device):
    record = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_requested": str(device),
        "stage11_machine_role": os.environ.get("STAGE11_MACHINE_ROLE"),
        "stage11_pixi_environment": os.environ.get(
            "STAGE11_PIXI_ENVIRONMENT"
        ),
        "runtime_used_for_candidate_selection": False,
    }
    if device.type == "cuda":
        index = 0 if device.index is None else int(device.index)
        record.update(
            {
                "gpu_name": torch.cuda.get_device_name(index),
                "gpu_capability": list(
                    torch.cuda.get_device_capability(index)
                ),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return record


def _resolve_device(value):
    device = torch.device(
        value if value is not None
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _analytic_laplacian_self_check(device):
    utils.set_seed(90117)
    model = SkyglowMLP([16, 8]).to(device)
    xy = torch.tensor(
        [[-0.8, -0.5], [-0.1, 0.2], [0.4, -0.7], [0.75, 0.6]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    analytic_b, analytic_lap = base_pinn._background_and_laplacian(model, xy)
    direct = torch.sigmoid(model(xy).squeeze(-1))
    grad = torch.autograd.grad(
        direct, xy, grad_outputs=torch.ones_like(direct), create_graph=True
    )[0]
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
    b_err = float(
        torch.max(torch.abs(analytic_b - direct)).detach().cpu()
    )
    lap_err = float(
        torch.max(torch.abs(analytic_lap - auto_lap)).detach().cpu()
    )
    if b_err > 2e-6 or lap_err > 2e-5:
        raise RuntimeError(
            "analytic Laplacian self-check failed: "
            f"background={b_err:.3e}, laplacian={lap_err:.3e}"
        )


def _sum_sq(grads):
    total = torch.zeros(
        (), dtype=torch.float64, device=grads[0].device
    )
    for g in grads:
        total = total + torch.sum(g.detach().double().square())
    return total


def _dot(a, b):
    total = torch.zeros(
        (), dtype=torch.float64, device=a[0].device
    )
    for ga, gb in zip(a, b):
        total = total + torch.sum(
            ga.detach().double() * gb.detach().double()
        )
    return total


def _cpu_state(value):
    if isinstance(value, dict):
        return {k: _cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_state(v) for v in value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rmse(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _relative_rmse(y, p):
    denom = float(
        np.sqrt(np.mean(np.asarray(y, dtype=np.float64) ** 2))
    )
    return _rmse(y, p) / max(denom, 1e-12)


def _safe_corr(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if np.std(y) <= 1e-12 or np.std(p) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(y, p)[0, 1])


def _fmt(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{value:.6g}" if np.isfinite(value) else "n/a"


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, ensure_ascii=False, indent=2, allow_nan=True
        ) + "\n",
        encoding="utf-8",
    )


def _write_rows(path, rows):
    path = Path(path)
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("smoke", "full"), default="full"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace only this diagnostic suite's output directory",
    )
    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="optionally save diagnostic prediction arrays; metrics are always saved",
    )
    args = parser.parse_args()
    main(
        suite=args.suite,
        device_name=args.device,
        force=args.force,
        save_arrays=args.save_arrays,
    )
