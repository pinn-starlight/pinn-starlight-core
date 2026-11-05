"""Verify that symbol citations in docs/CODE_MAP.md still point at real code.

The code map is used during a live code audit: every "file → function(行号)"
claim has to be true.  This checker re-derives each cited symbol from the map
itself, so editing code without updating the map is caught here instead of in
front of a judge.

Guarded sources (single source of truth, edit here when the map changes):
  * ``CITED_SYMBOLS``     -- (relative path, symbol) pairs quoted in the map
  * ``CITED_LINE_RANGES`` -- (relative path, symbol, expected start/end line)

Usage
-----
    python -m tools.verify_code_map            # check everything
    python -m tools.verify_code_map --list     # list what is checked

Exit code is 0 only when every citation resolves.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_MAP = PROJECT_ROOT / "docs" / "CODE_MAP.md"

# Symbols the map names explicitly.  Module-level constants are included here
# on purpose: the AST walker below accepts Assign/AnnAssign as well as defs.
CITED_SYMBOLS: dict[str, tuple[str, ...]] = {
    "src/pinn_starlight_core/nn/pinn_loss.py": (
        "mse_data", "mse_physics", "laplacian",
        "mse_physics_from_laplacian", "sparse_effective_forcing",
    ),
    "src/pinn_starlight_core/nn/pinn_layers.py": ("SkyglowMLP", "skyglow_linear"),
    "src/pinn_starlight_core/nn/physics_model.py": ("Alpha", "EffectiveForcingMixture"),
    "src/pinn_starlight_core/physics/forcing.py": (
        "point_source_term_numpy", "candidate_centers_from_observed",
        "build_forcing_component_numpy", "forcing_negative_energy_fraction",
        "unit_rms_scale_numpy",
    ),
    "src/pinn_starlight_core/data/image_loader.py": ("ImageLoader", "coordinate_grid"),
    "src/pinn_starlight_core/main.py": ("single_train", "main"),
    "experiments/common/baselines/coordinate_pinn.py": (
        "PINNConfig", "DEFAULT_CONFIG", "train_background",
        "initial_centers_from_observed", "_build_forcing", "_train_steps",
        "resolve_laplacian_backend", "_background_and_laplacian",
        "normalized_config", "_optimizer", "_predict_background", "_history_row",
    ),
    "experiments/common/baselines/sgc_pinn.py": (
        "CONTROL_NAME", "METHOD_NAME", "normalized_followup_config", "run_paired",
        "fit_fixed_forcing", "finetune", "_set_forcing_weights", "_sum_sq", "_dot",
    ),
    "experiments/common/baselines/cosine_ls.py": (
        "estimate_background", "_estimate_background_with_settings",
        "_component_bounds", "_prepare_workspace", "_gram_from_states",
        "_profile_from_gram", "_reconstruct_background",
        "_screened_operator_symbol", "_apply_screened_operator",
        "_dct_1d_torch", "_idct_1d_torch", "validate_frozen_config",
    ),
    "experiments/common/baselines/fft_gaussian.py": ("estimate_background",),
    "experiments/common/baselines/unet_small.py": (
        "_ConvBlock", "UNetSmall", "train", "predict", "predict_with_model",
        "load_checkpoint_model", "_predict_background", "_predict_tiled",
        "_validation_loss",
    ),
    "experiments/configs/formal_experiment.py": (
        "FORMAL_SEEDS", "FORMAL_METHODS", "FORMAL_EXECUTION_CONFIG",
        "SYNTHETIC_DATA_CONFIG", "_FORCING_COMMON", "PINN_CONFIG",
        "GAUSSIAN_CONFIG", "FOURIER_LS_CONFIG", "FINAL_PINN_FOLLOWUP_CONFIG",
        "UNET_CONFIG", "FORMAL_EXPERIMENT_CONFIG",
        "formal_config_snapshot", "selection_dependency_config",
    ),
    "experiments/configs/split_protocols.py": (
        "BASE_IDS", "STANDARD_BASE_SPLIT", "CV5_SEQUENCE", "CV5_FOLDS",
        "validate_protocols", "active_base_split",
    ),
    "experiments/configs/stage_b_experiment.py": ("STAGE_B_CONFIG",),
    "experiments/common/data/generate_synthetic.py": (
        "load_clean_image", "build_star_reference", "synthesize_from_clean",
        "save_sample", "_prepare_native_reference", "build_dataset", "_stable_seed",
    ),
    "experiments/common/data/synthetic_backgrounds.py": ("BACKGROUND_TYPES", "generate_background"),
    "experiments/common/data/native_background.py": (
        "estimate_native_background", "flatten_reference", "audit_flattening",
    ),
    "experiments/common/data/stage_b_backgrounds.py": (
        "SCENE_TYPES", "generate_stage_b_scene", "_protected_structure",
        "spectral_overlap_coefficient",
    ),
    "experiments/common/data/generate_stage_b.py": ("build_dataset", "_scene_star_reference"),
    "experiments/common/utils/scores.py": ("mae", "rmse", "psnr", "ssim"),
    "experiments/common/utils/metrics.py": (
        "build_detection_image", "extract_stars", "measure_raw_aperture_flux",
        "star_metrics", "evaluate_synthetic", "evaluate_real", "_match_stars",
    ),
    "experiments/common/utils/stage_b_metrics.py": (
        "evaluate_stage_b", "_faint_star_metrics", "_protected_structure_metrics",
    ),
    "experiments/common/utils/experiment_utils.py": (
        "set_seed", "load_float_tiff", "load_observed", "reconstruct_residual",
        "load_global_configuration_lock", "save_run_metadata", "_resolve_data_root",
    ),
    "experiments/common/utils/selection_artifacts.py": (
        "SELECTION_SOURCE_DEPENDENCIES", "FORMAL_SOURCE_DEPENDENCIES",
        "build_selection_artifact", "verify_selection_artifact",
    ),
    "experiments/common/utils/stage_b_provenance.py": (
        "build_selection_artifact", "verify_selection_artifact",
        "source_snapshot", "verify_source_snapshot",
    ),
    "experiments/common/utils/stage_b_anchor.py": ("load_stage_a_anchor",),
    "experiments/scripts/preflight/run.py": ("main",),
    "experiments/scripts/selection/gaussian.py": ("validation_mse_scores", "main"),
    "experiments/scripts/selection/fourier_ls.py": (
        "select_hyperparameters", "_evaluate_candidate", "_selection_key",
    ),
    "experiments/scripts/selection/unet.py": ("main",),
    "experiments/scripts/selection/pinn.py": (
        "run_candidate", "_select_stage", "_continue_until_stable",
        "_precompute_initial_centers", "_selection_key",
    ),
    "experiments/scripts/config_lock/run.py": ("main", "_verify_unet_checkpoints"),
    "experiments/scripts/benchmark.py": ("run_method", "_formal_device", "main"),
    "experiments/scripts/ablation.py": ("build_variant_config", "_center_error"),
    "experiments/scripts/real_images/run.py": ("main", "_load_global_lock_for_e4", "_run_final_pinn_only"),
    "experiments/scripts/final_pinn/common.py": (
        "forced_reference_metrics", "paired_comparisons", "write_csv",
    ),
    "experiments/scripts/final_pinn/stage_a.py": ("main",),
    "experiments/scripts/final_pinn/stage_b.py": ("main",),
    "experiments/scripts/final_pinn/aggregate_cv5.py": ("main",),
    "experiments/scripts/protocol/run.py": ("_phase_modules", "run_one"),
    "experiments/scripts/protocol/prepare_cv5.py": ("prepare_stage",),
    "experiments/scripts/protocol/aggregate_cv5.py": ("aggregate",),
    "experiments/scripts/verify_cosine_ls/run.py": ("main",),
    "experiments/scripts/stage_b/formal_run.py": ("_load_lock", "main"),
    "experiments/scripts/stage_b/lock.py": ("FORMAL_SOURCE_FILES", "main"),
    "experiments/scripts/stage_b/select_pinn.py": ("main",),
    "run_stage11.py": (
        "GPUInfo", "_query_nvidia_smi", "_select_gpus", "_validate_runtime_profile",
        "_single_gpu_env", "_gpu_health_probe", "ensure_dataset",
        "run_a", "run_b", "run_cv5", "run_e4", "_run_one_cv_fold", "main",
    ),
    "tools/smoke_test.py": (
        "test_imports", "test_analytic_laplacian", "test_data_loading",
        "test_fft_gaussian", "test_fourier_ls", "test_pinn", "test_final_pinn",
        "test_metrics_stage_a", "test_metrics_stage_b", "test_metrics_real",
        "test_generators", "test_data_inventory", "TESTS",
    ),
    "tools/run_one_image.py": (
        "load_synthetic", "load_image", "evaluate_synthetic", "evaluate_real",
        "draw_figure", "main",
    ),
    "tools/code_check_demo.py": (
        "load_sample", "run_methods", "evaluate", "draw_figure", "main",
    ),
}

# "file :: symbol  ->  (start, end)" inclusive 1-based line numbers cited in the map.
# Only claims the map states as explicit ranges are pinned here.
CITED_LINE_RANGES: tuple[tuple[str, str, int, int], ...] = (
    ("src/pinn_starlight_core/nn/pinn_loss.py", "mse_physics", 15, 17),
    ("src/pinn_starlight_core/nn/pinn_loss.py", "laplacian", 33, 55),
    ("src/pinn_starlight_core/nn/pinn_layers.py", "SkyglowMLP", 14, 35),
    ("src/pinn_starlight_core/nn/physics_model.py", "Alpha", 13, 30),
    ("src/pinn_starlight_core/nn/physics_model.py", "EffectiveForcingMixture", 61, 299),
    ("src/pinn_starlight_core/physics/forcing.py", "point_source_term_numpy", 25, 38),
    ("src/pinn_starlight_core/physics/forcing.py", "candidate_centers_from_observed", 41, 104),
    ("src/pinn_starlight_core/physics/forcing.py", "forcing_negative_energy_fraction", 208, 215),
    ("experiments/common/baselines/coordinate_pinn.py", "train_background", 85, 204),
    ("experiments/common/baselines/coordinate_pinn.py", "normalized_config", 421, 479),
    ("experiments/common/baselines/coordinate_pinn.py", "_background_and_laplacian", 371, 419),
    ("experiments/common/baselines/sgc_pinn.py", "run_paired", 80, 188),
    ("experiments/common/baselines/sgc_pinn.py", "fit_fixed_forcing", 191, 282),
    ("experiments/common/baselines/sgc_pinn.py", "finetune", 285, 533),
    ("experiments/common/baselines/cosine_ls.py", "estimate_background", 65, 92),
    ("experiments/common/baselines/cosine_ls.py",
     "_estimate_background_with_settings", 95, 368),
    ("experiments/common/baselines/cosine_ls.py", "_profile_from_gram", 918, 949),
    ("experiments/common/baselines/cosine_ls.py", "_screened_operator_symbol", 987, 1000),
    ("experiments/common/baselines/fft_gaussian.py", "estimate_background", 12, 38),
    ("experiments/common/baselines/unet_small.py", "UNetSmall", 50, 102),
    ("experiments/common/utils/metrics.py", "evaluate_synthetic", 325, 386),
    ("experiments/common/utils/experiment_utils.py", "load_global_configuration_lock", 285, 390),
    ("experiments/scripts/benchmark.py", "_formal_device", 101, 109),
    ("experiments/scripts/selection/pinn.py", "_select_stage", 216, 258),
    ("experiments/scripts/selection/pinn.py", "_continue_until_stable", 261, 332),
    ("experiments/scripts/config_lock/run.py", "main", 52, 173),
    ("experiments/scripts/protocol/prepare_cv5.py", "prepare_stage", 68, 179),
    ("experiments/configs/split_protocols.py", "validate_protocols", 75, 99),
    ("tools/smoke_test.py", "test_analytic_laplacian", 191, 225),
    ("experiments/scripts/stage_b/preflight.py", "main", 36, 144),
    ("experiments/scripts/stage_b/lock.py", "main", 50, 162),
    ("experiments/scripts/stage_b/formal_run.py", "_load_lock", 25, 36),
    ("experiments/scripts/stage_b/formal_run.py", "main", 202, 332),
    ("experiments/scripts/stage_b/select_pinn.py", "main", 126, 243),
    ("experiments/scripts/final_pinn/stage_b.py", "main", 35, 161),
    ("experiments/common/utils/stage_b_anchor.py", "load_stage_a_anchor", 28, 82),
    ("experiments/scripts/protocol/aggregate_cv5.py", "aggregate", 79, 174),
    ("experiments/scripts/verify_cosine_ls/run.py", "main", 13, 75),
    ("experiments/common/data/generate_stage_b.py", "build_dataset", 102, 240),
)


def _defined_names(tree: ast.Module) -> dict[str, tuple[int, int]]:
    """Map every top-level def/class/assignment name to its source span."""
    spans: dict[str, tuple[int, int]] = {}
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            end = getattr(node, "end_lineno", node.lineno)
            for name in names:
                spans[name] = (start, end)
    return spans


def _check_symbols():
    problems = []
    total = 0
    for relative, names in CITED_SYMBOLS.items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            problems.append(f"missing file: {relative}")
            continue
        spans = _defined_names(ast.parse(path.read_text(encoding="utf-8")))
        for name in names:
            total += 1
            if name not in spans:
                problems.append(f"{relative}::{name} (symbol not found)")
    return total, problems


def _check_lines():
    problems = []
    total = 0
    for relative, name, start, end in CITED_LINE_RANGES:
        total += 1
        path = PROJECT_ROOT / relative
        if not path.is_file():
            problems.append(f"missing file: {relative}")
            continue
        spans = _defined_names(ast.parse(path.read_text(encoding="utf-8")))
        if name not in spans:
            problems.append(f"{relative}::{name} (symbol not found)")
            continue
        actual_start, actual_end = spans[name]
        if (actual_start, actual_end) != (start, end):
            problems.append(
                f"{relative}::{name} cited as {start}-{end} "
                f"but actually {actual_start}-{actual_end}"
            )
    return total, problems


def _check_code_map_exists():
    if not CODE_MAP.is_file():
        return [f"code map missing: {CODE_MAP}"]
    text = CODE_MAP.read_text(encoding="utf-8")
    problems = []
    for relative in CITED_SYMBOLS:
        if relative not in text:
            problems.append(f"code map never mentions {relative}")
    return problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list checked citations")
    args = parser.parse_args(argv)

    if args.list:
        for relative, names in CITED_SYMBOLS.items():
            for name in names:
                print(f"{relative}::{name}")
        for relative, name, start, end in CITED_LINE_RANGES:
            print(f"{relative}::{name}  [{start}-{end}]")
        return 0

    symbol_total, symbol_problems = _check_symbols()
    line_total, line_problems = _check_lines()
    map_problems = _check_code_map_exists()
    problems = symbol_problems + line_problems + map_problems

    print("=" * 70)
    print("CODE MAP CITATION CHECK")
    print("=" * 70)
    print(f"files                : {len(CITED_SYMBOLS)}")
    print(f"symbols checked      : {symbol_total}")
    print(f"line ranges checked  : {line_total}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for item in problems:
            print(f"  - {item}")
        print("\nUpdate docs/CODE_MAP.md (and this guard) to match the code.")
        return 1
    print("\nRESULT: all citations resolve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
