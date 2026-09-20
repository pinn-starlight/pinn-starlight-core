"""Stage10-B follow-up benchmark configuration.

Stage10-A remains frozen and untouched.

Stage10-B is a post-Stage10-A follow-up stress test motivated by the finding
that the Stage10-A synthetic task separates low-frequency skyglow from stellar
structure unusually cleanly. To keep the B comparison fair, each method may use
only Stage-B train/validation data for its normal pre-test selection/training.
Stage-B test arrays are not opened until a separate Stage-B Global Lock exists.

The candidate spaces, training budgets, architectures, numerical solvers and
execution policies are inherited unchanged from Stage10-A. Only the synthetic
scene distribution changes.
"""
from __future__ import annotations

from copy import deepcopy

from experiments.configs.split_protocols import active_base_split

from experiments.configs.formal_experiment import (
    FINAL_PINN_FOLLOWUP_CONFIG,
    FORMAL_EXECUTION_CONFIG,
    FORMAL_METHODS,
    FORMAL_SEEDS,
    selection_dependency_config,
)
from experiments.common.data.stage_b_backgrounds import (
    INTENSITY_LEVELS,
    PROTECTED_PEAK,
    SCENE_TYPES,
)

STAGE_B_SCHEMA_VERSION = 4
STAGE_B_PROTOCOL_VERSION = 4
STAGE_B_SEED = 20260909

STAGE_B_BASE_SPLIT = active_base_split()

STAGE_B_CONFIG = {
    "schema_version": STAGE_B_SCHEMA_VERSION,
    "protocol_version": STAGE_B_PROTOCOL_VERSION,
    "protocol_name": "adapted_frequency_overlap_followup_benchmark",
    "seed": STAGE_B_SEED,
    "methods": list(FORMAL_METHODS),
    "base_split": dict(STAGE_B_BASE_SPLIT),
    "scene_types": list(SCENE_TYPES),
    "intensity_levels": dict(INTENSITY_LEVELS),
    "protected_peak_by_scene": dict(PROTECTED_PEAK),
    "observation_model": {
        "formula": "observed = base_clean + protected_true + background_true",
        "clean_true_formula": "clean_true = base_clean + protected_true",
        "background_true_semantics": "synthetic_light_pollution_to_remove",
        "protected_true_semantics": (
            "synthetic_diffuse_low_frequency_structure_that_belongs_to_the_clean_target"
        ),
        "additional_synthetic_noise": False,
        "scientific_clipping": False,
        "multiplicative_rescaling": False,
    },
    "comparison_policy": {
        "same_camera_base_split_as_stage10_a": True,
        "same_method_implementations_as_stage10_a": True,
        "same_candidate_spaces_and_training_budgets_as_stage10_a": True,
        "stage_b_train_validation_may_be_used_before_b_lock": True,
        "stage_b_test_may_be_used_before_b_lock": False,
        "reselect_gaussian_on_stage_b_validation": True,
        "reselect_fourier_ls_on_stage_b_validation": True,
        "retrain_unet_on_stage_b_train_validation": True,
        "reselect_pinn_on_stage_b_validation": True,
        "reason": (
            "compare each method under the harder Stage-B distribution without "
            "giving any method post-test tuning or an avoidable distribution-shift handicap"
        ),
    },
    "evaluation": {
        "faint_star_fraction": 1.0 / 3.0,
        "protected_mask_threshold_fraction": 0.25,
        "minimum_required_spectral_overlap": 0.85,
        "primary_new_metrics": [
            "faint_star_recall",
            "protected_region_mae",
            "protected_projection_abs_error",
        ],
    },
    "execution": deepcopy(FORMAL_EXECUTION_CONFIG),
    "final_pinn_followup": deepcopy(FINAL_PINN_FOLLOWUP_CONFIG),
}


def stage_b_config_snapshot() -> dict:
    return deepcopy(STAGE_B_CONFIG)


def stage_b_selection_config(method: str) -> dict:
    """Use exactly the same candidate spaces and budgets as Stage10-A."""
    return deepcopy(selection_dependency_config(method))


STAGE_B_TEST_SEEDS = tuple(int(v) for v in FORMAL_SEEDS)
