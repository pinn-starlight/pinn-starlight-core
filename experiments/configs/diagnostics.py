"""Configuration for the non-formal mechanism probes.

These settings belong to ``experiments/scripts/diagnostics/`` only.  They are
deliberately kept out of ``formal_experiment.py`` / ``stage_b_experiment.py``:
the frozen Stage-A/B protocols, locks, seeds and budgets must stay untouched.
Changing a value here never changes a formal result.

The probes answer three questions:

* ``init_memory``: how much of the physics-first initialization advantage in
  ``protected_projection_ratio`` survives the later refinement, and how strong
  an anchor is needed to retain it?
* ``alpha_sweep``: for which ``alpha`` (and during which training window) does
  the screened-Poisson residual fall fastest?
* ``mode_eval``: what do the ``star`` / ``nebula`` / ``balanced`` operating
  points cost in background quality, star photometry and diffuse-structure
  preservation?

Nothing in this module is a paper result.  The diagnostic budgets are declared
before the runs and are intentionally smaller than the formal ones.
"""

from __future__ import annotations

from copy import deepcopy

from experiments.configs.formal_experiment import (
    FINAL_PINN_FOLLOWUP_CONFIG,
    FORMAL_SEEDS,
    PINN_CONFIG,
)
from experiments.configs.stage_b_experiment import STAGE_B_CONFIG

DIAGNOSTIC_SCHEMA_VERSION = 1

# One fixed seed, matching the frozen exploratory preservation probe.
DIAGNOSTIC_SEED = FORMAL_SEEDS[0]

# Stage-B evaluation threshold, taken from the formal Stage-B config so the
# diagnostic projection metric cannot silently drift from the paper one.
PROTECTED_MASK_THRESHOLD_FRACTION = float(
    STAGE_B_CONFIG["evaluation"]["protected_mask_threshold_fraction"]
)

# --------------------------------------------------------------------------
# budgets
# --------------------------------------------------------------------------
# ``smoke`` exists to prove the plumbing on a CPU in seconds.  ``full`` is the
# declared diagnostic budget: it matches the frozen Stage-11 follow-up budgets
# (20k data-only / 20k physics-first initialization, 5k coordinated finetune).
BUDGETS = {
    "smoke": {
        "init_steps": 40,
        "finetune_steps": 24,
        "curve_steps": (0, 12, 24),
        "alpha_steps": 40,
        "alpha_chunk": 20,
        "max_side": 512,
        "max_samples": 1,
    },
    "full": {
        "init_steps": 20000,
        "finetune_steps": 5000,
        "curve_steps": (0, 1000, 5000),
        "alpha_steps": 10000,
        "alpha_chunk": 100,
        "max_side": None,
        "max_samples": 8,
    },
}

SUITES = tuple(BUDGETS)
DATASETS = ("stage_a", "stage_b")

# --------------------------------------------------------------------------
# initialization recipes (shared by init_memory and the balanced mode)
# --------------------------------------------------------------------------
# ``physics_first`` reproduces the frozen probe's 0.1 : 1.0 data : physics
# ratio with the unmodified formal training loop: physics_weight is relative to
# a unit data term, so 10.0 means data weight 0.1.  sparse_weight is scaled by
# the same factor (1e-2 relative to 10.0 == 1e-3 relative to 1.0).
INIT_RECIPES = {
    "data_first": {"physics_weight": 0.0, "sparse_weight": 0.0},
    "physics_first": {"physics_weight": 10.0, "sparse_weight": 1e-2},
}
INIT_NAMES = tuple(INIT_RECIPES)

FINETUNE_ARMS = ("none", "data_only", "sgc", "anchored")
ANCHOR_WEIGHTS = (0.25, 0.5, 0.75, 1.0)

# Small budget knobs for the smoke suite; the full suite keeps the frozen
# Stage-11 follow-up architecture / optimizer settings.
SMOKE_MODEL = {
    "hidden_dims": [32, 16],
    "batch_size": 4096,
    "model_lr": 1e-3,
    "forcing_max_components": 2,
    "forcing_rms_grid_size": 32,
    "prediction_batch_size": 65536,
}


def _apply_smoke_overrides(config: dict, *, steps: int) -> dict:
    config = dict(config)
    config.update(deepcopy(SMOKE_MODEL))
    config["steps"] = int(steps)
    config["center_warmup_steps"] = min(5, int(steps))
    config["center_refinement_enabled"] = False
    config["log_every"] = max(1, int(steps) // 4)
    return config


def stage1_config(*, suite: str, init: str = "data_first", steps: int | None = None) -> dict:
    """Stage-1 config for one initialization recipe.

    The returned mapping is unnormalised on purpose; callers pass it through
    ``coordinate_pinn.normalized_config`` exactly like the formal runners do.
    """
    if suite not in BUDGETS:
        raise ValueError(f"unknown suite: {suite!r}; expected one of {SUITES}")
    if init not in INIT_RECIPES:
        raise ValueError(f"unknown init: {init!r}; expected one of {INIT_NAMES}")
    steps = int(BUDGETS[suite]["init_steps"] if steps is None else steps)
    config = deepcopy(FINAL_PINN_FOLLOWUP_CONFIG["stage1"])
    if suite == "smoke":
        config = _apply_smoke_overrides(config, steps=steps)
    else:
        config["steps"] = steps
    config.update(INIT_RECIPES[init])
    return config


def stage2_config(*, suite: str) -> dict:
    if suite not in BUDGETS:
        raise ValueError(f"unknown suite: {suite!r}; expected one of {SUITES}")
    config = deepcopy(FINAL_PINN_FOLLOWUP_CONFIG["stage2"])
    if suite == "smoke":
        config["prefit_points"] = 4096
    return config


def stage3_config(*, suite: str, steps: int | None = None) -> dict:
    if suite not in BUDGETS:
        raise ValueError(f"unknown suite: {suite!r}; expected one of {SUITES}")
    steps = int(BUDGETS[suite]["finetune_steps"] if steps is None else steps)
    config = deepcopy(FINAL_PINN_FOLLOWUP_CONFIG["stage3"])
    config["steps"] = steps
    config["ramp_steps"] = max(1, min(int(config["ramp_steps"]), steps // 2 or 1))
    config["log_every"] = max(1, steps // 4)
    return config


def pinn_config(*, suite: str, steps: int | None = None) -> dict:
    """Joint data + physics config used by the ``nebula`` operating point."""
    if suite not in BUDGETS:
        raise ValueError(f"unknown suite: {suite!r}; expected one of {SUITES}")
    steps = int(BUDGETS[suite]["alpha_steps"] if steps is None else steps)
    config = deepcopy(PINN_CONFIG["training"])
    if suite == "smoke":
        config = _apply_smoke_overrides(config, steps=steps)
        config["physics_weight"] = float(PINN_CONFIG["training"]["physics_weight"])
        config["sparse_weight"] = float(PINN_CONFIG["training"]["sparse_weight"])
    else:
        config["steps"] = steps
        config["log_every"] = 100
    return config


# --------------------------------------------------------------------------
# alpha sweep
# --------------------------------------------------------------------------
ALPHA_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
FORCING_ALPHA_MODES = ("coupled", "frozen")
FORCING_ALPHA_REFERENCE = 0.5
ALPHA_EVAL_POINTS = 4096
# Declared diagnostic constants: the formal selection artifact is not read here
# (and is not present on this machine).  They are fixed for every alpha so the
# comparison is self-consistent, not paper-comparable.
ALPHA_LS_LAMBDA = 0.3
ALPHA_LS_SPARSE_WEIGHT = 1e-3

# --------------------------------------------------------------------------
# mode switch
# --------------------------------------------------------------------------
MODE_NAMES = ("star", "nebula", "balanced")
MODE_DEFAULTS = {
    "star": {},
    "nebula": {},
    "balanced": {
        "init": "physics_first",
        "anchor_weight": 0.5,
    },
}

# Pre-declared acceptance rule for the experimental balanced mode (validation
# only; evaluated by mode_eval, never used to select a formal method).
BALANCED_ACCEPTANCE = {
    "min_projection_ratio": 0.15,
    "max_bg_mae_ratio_vs_sgc": 1.25,
}
