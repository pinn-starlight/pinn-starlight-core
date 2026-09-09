"""Single source of truth for the Stage-10 formal experiment configuration.

Stage 10 changes the *benchmark semantics* and the physics-matched forcing model.  The camera-acquired reference is first
flattened with a fixed method-independent low-frequency background estimator;
synthetic skyglow is then added without any extra Gaussian/Poisson noise.

PINN and Fourier-LS share the same observed-only candidate-centre rule and the
same sparse signed effective-forcing family.  They still solve different
optimization problems.
"""

from __future__ import annotations

from copy import deepcopy
import math

FORMAL_CONFIG_SCHEMA_VERSION = 3
FORMAL_SEEDS = (20260728, 20260729, 20260730)
FORMAL_METHODS = ("fft_gaussian", "fourier_ls", "unet_small", "pinn")

# Formal Stage10-A execution policy.  These settings freeze numerical/backend
# choices that should not depend on environment variables or GPU count.
# `unet_small` remains the legacy internal identifier; the paper-facing label
# is the standard-capacity single-channel U-Net baseline.
FORMAL_EXECUTION_CONFIG = {
    "benchmark_gpu_policy": "single_visible_gpu",
    "benchmark_device": "cuda:0",
    "pinn_laplacian_backend": "autograd",
    "fourier_ls_backend": "cuda",
    "fourier_ls_numeric_dtype": "float64/complex128",
    "unet_training_memory_format": "nchw_contiguous",
    "unet_optimizer": "Adam",
    "unet_fused_adam": False,
    "unet_tiled_batch_size": 1,
}

SYNTHETIC_DATA_CONFIG = {
    "protocol_version": 4,
    "native_background": {
        "method": "photutils.Background2D_sigma_clipped_median",
        "mesh_divisor": 16,
        "mesh_min": 64,
        "mesh_max": 256,
        "sigma_clip": 3.0,
        "sigma_clip_maxiters": 5,
        "filter_size": 3,
        # Audit is intentionally independent of every evaluated method.
        "star_threshold": 0.03,
        "star_fwhm": 3.0,
        "audit_max_stars": 300,
        "audit_sky_median_abs_max": 0.005,
        "audit_centroid_shift_median_max": 0.10,
        "audit_flux_error_median_max": 0.01,
        "audit_flux_error_p95_max": 0.05,
        "strict_audit": True,
    },
    "observation_model": {
        "formula": "observed = clean_true + background_true",
        "clean_true_semantics": (
            "camera-acquired reference after fixed native low-frequency background "
            "subtraction; camera-scale fluctuations remain except for the deliberately "
            "removed low-frequency estimate"
        ),
        "background_true_semantics": "synthetically_added_skyglow_component",
        "additional_synthetic_noise": False,
        "skyglow_induced_shot_noise_simulated": False,
        "task_scope": "light_pollution_background_estimation_and_removal_not_denoising",
        "scientific_clipping": False,
        "multiplicative_rescaling_after_flattening": False,
    },
}

_FORCING_COMMON = {
    "forcing_max_components": 3,
    "forcing_sigma_bounds": [0.05, 0.55],
    "forcing_theta_bounds": [-0.5 * math.pi, 0.5 * math.pi],
    "forcing_winsor_percentile": 99.5,
    "forcing_low_pass_sigma_fraction": 0.04,
    "forcing_min_center_distance": 0.30,
    "forcing_center_refinement": 0.08,
    "forcing_initial_weights": [1.0, 0.05, 0.05],
    "forcing_initial_sigma": 0.425,
    "forcing_rms_grid_size": 64,
    "forcing_rms_epsilon": 1e-8,
    "inactive_relative_threshold": 0.05,
}

PINN_CONFIG = {
    "selection": {
        "seed": FORMAL_SEEDS[0],
        "selection_order": ["network", "physics_weight", "sparse_weight", "steps"],
        "network_candidates": [[256, 128], [256, 64], [128, 128], [128, 64]],
        "physics_weight_candidates": [0.4, 0.5, 0.8, 1.0, 1.5],
        "sparse_weight_candidates": [1e-4, 1e-3, 1e-2],
        "continue_steps": 1000,
        "max_steps": 50000,
        "stability_tolerance": 0.005,
        "stability_patience": 3,
        "selection_metric": (
            "validation_bg_mae_then_residual_ssim_then_flux_error_then_parameter_count"
        ),
    },
    "training": {
        "hidden_dims": [128, 128],
        "physics_weight": 0.5,
        "sparse_weight": 1e-3,
        "steps": 10000,
        "batch_size": 8192,
        "model_lr": 1e-3,
        "forcing_lr": 1e-3,
        "alpha": 0.5,
        # 10% of the nominal 10k Stage-10 fit.  This is absolute so continuation
        # runs do not accidentally re-freeze centres after resume.
        "center_warmup_steps": 1000,
        "center_refinement_enabled": True,
        **_FORCING_COMMON,
        "prediction_batch_size": 65536,
        "log_every": 100,
    },
    "test": {
        "seeds": list(FORMAL_SEEDS),
    },
    "ablation": {
        "variants": [
            "data_only",
            "single_component",
            "multi_fixed_centers",
            "multi_bounded_centers",
        ],
        "allowed_changed_keys": [
            "physics_weight",
            "sparse_weight",
            "forcing_max_components",
            "center_refinement_enabled",
        ],
    },
}

GAUSSIAN_CONFIG = {
    "selection": {
        "sigma_candidates": [0.02, 0.04, 0.08, 0.16],
        "selection_metric": "validation_background_mse",
    },
    "method": {
        "deterministic": True,
    },
}

FOURIER_LS_CONFIG = {
    "selection": {
        "lambda_ls_candidates": [0.01, 0.1, 0.3, 0.4, 0.5],
        "lower_boundary_extension": 0.001,
        "upper_boundary_extension": 1.0,
        "sparse_weight_candidates": [1e-4, 1e-3, 1e-2],
        # To keep selection practical, lambda is selected first with the middle
        # sparse value, then sparse weight is selected at the chosen lambda.
        "lambda_stage_sparse_weight": 1e-3,
        "selection_metric": "validation_background_mae",
    },
    "method": {
        "alpha": 0.5,
        "max_components": 3,
        "center_initialization": {
            "rule": (
                "observed-only P99.5 winsorization + Gaussian low-pass + greedy NMS"
            ),
            "winsor_percentile": _FORCING_COMMON["forcing_winsor_percentile"],
            "low_pass_sigma_fraction": _FORCING_COMMON[
                "forcing_low_pass_sigma_fraction"
            ],
            "min_center_distance": _FORCING_COMMON["forcing_min_center_distance"],
            "center_refinement": _FORCING_COMMON["forcing_center_refinement"],
        },
        "forcing": {
            "family": "signed_effective_forcing_q_times_elliptical_gaussian",
            "component_normalization": "unit_rms_on_fixed_canonical_grid",
            "rms_grid_size": _FORCING_COMMON["forcing_rms_grid_size"],
            "rms_epsilon": _FORCING_COMMON["forcing_rms_epsilon"],
            "weights_nonnegative": True,
            "inactive_relative_threshold": _FORCING_COMMON[
                "inactive_relative_threshold"
            ],
        },
        "geometry": {
            "parameters_per_component": [
                "center_x",
                "center_y",
                "sigma_x",
                "sigma_y",
                "theta",
            ],
            "sigma_bounds": _FORCING_COMMON["forcing_sigma_bounds"],
            "theta_bounds": _FORCING_COMMON["forcing_theta_bounds"],
            "initial_sigma": 0.425,
            "initial_theta": 0.0,
            "estimated_per_image_from_observed_only": True,
        },
        "optimizer": {
            "name": "bounded_profile_block_coordinate_Powell",
            "max_sweeps": 2,
            "relative_objective_early_stop": 1e-4,
            # Engineering budget frozen before the new formal run; these values
            # are not tuned on the held-out test split.
            "component_maxiter": 25,
            "xtol": 1e-3,
            "ftol": 1e-4,
            "weight_profile": "enumerate_all_2^K_nonnegative_active_sets",
        },
        "boundary_condition": "periodic",
        "coordinate_system": "normalized_xy_minus1_to_1",
        "scientific_output_clipping": False,
        "deterministic": True,
    },
}

UNET_CONFIG = {
    "training": {
        "patch_size": 256,
        "epochs": 12,
        "steps_per_epoch": 100,
        "batch_size": 4,
        "base_channels": 64,
        "learning_rate": 1e-3,
        "patience": 3,
        "min_delta_relative": 0.002,
        "amp_requested": True,
        "seeds": list(FORMAL_SEEDS),
    },
    "inference": {
        "mode": "tiled",
        "tile_size": 256,
        "overlap": 64,
        "policy": "fixed tiled inference",
    },
    "selection": {
        "checkpoint_metric": "validation_loss",
        "e4_checkpoint_rule": "minimum validation_loss across formal seeds",
    },
}

FORMAL_EXPERIMENT_CONFIG = {
    "schema_version": FORMAL_CONFIG_SCHEMA_VERSION,
    "methods": list(FORMAL_METHODS),
    "execution": FORMAL_EXECUTION_CONFIG,
    "synthetic_data": SYNTHETIC_DATA_CONFIG,
    "pinn": PINN_CONFIG,
    "fft_gaussian": GAUSSIAN_CONFIG,
    "fourier_ls": FOURIER_LS_CONFIG,
    "unet_small": UNET_CONFIG,
}


def formal_config_snapshot() -> dict:
    """Return a mutation-safe snapshot of every formal experiment setting."""
    return deepcopy(FORMAL_EXPERIMENT_CONFIG)


def selection_dependency_config(method: str) -> dict:
    """Return exactly the config subset that can change one selector's result."""
    # All selectors depend on the Stage-10 synthetic-data semantics because the
    # validation arrays change when native-background flattening changes.
    data_dependency = {"synthetic_data": SYNTHETIC_DATA_CONFIG}
    if method == "pinn":
        return deepcopy(
            {
                **data_dependency,
                "selection": PINN_CONFIG["selection"],
                "training": PINN_CONFIG["training"],
            }
        )
    if method == "fft_gaussian":
        return deepcopy({**data_dependency, **GAUSSIAN_CONFIG})
    if method == "fourier_ls":
        return deepcopy({**data_dependency, **FOURIER_LS_CONFIG})
    if method == "unet_small":
        return deepcopy({**data_dependency, **UNET_CONFIG})
    raise ValueError(f"Unknown formal method: {method}")
