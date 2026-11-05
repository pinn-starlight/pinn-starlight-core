"""Stage10-B PINN validation selector.

The search order, candidate spaces, continuation rule and selection metric are
identical to Stage10-A E1. Only the validation images are Stage-B images.

Stage-B has a different clean-target semantic: clean_true includes the protected
low-frequency structure. Therefore the Stage10-A evaluator's hard-coded
`catalog_source` guard must not be reused verbatim. We reuse the A E1 optimizer
and selection machinery, but install a Stage-B-specific evaluator only for this
process. Stage10-A source files are not modified.
"""
from __future__ import annotations

import argparse
import numpy as np

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.data.generate_stage_b import STAGE_B_MANIFEST
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics as common_metrics
from experiments.common.utils import scores
from experiments.common.utils import stage_b_provenance as provenance
from experiments.configs.stage_b_experiment import stage_b_selection_config
from experiments.scripts.selection import pinn as pinn_selector

OUTPUT_ROOT = utils.OUTPUT_ROOT / "stage_b/selection/pinn"
ARTIFACT = OUTPUT_ROOT / "pinn_selected.json"
SOURCE_FILES = (
    "experiments/scripts/stage_b/select_pinn.py",
    "experiments/scripts/selection/pinn.py",
    "experiments/configs/stage_b_experiment.py",
    "experiments/common/data/generate_stage_b.py",
    "experiments/common/data/stage_b_backgrounds.py",
    "experiments/common/baselines/coordinate_pinn.py",
    "experiments/common/utils/metrics.py",
    "experiments/common/utils/experiment_utils.py",
    "src/pinn_starlight_core/data/image_loader.py",
    "src/pinn_starlight_core/nn/physics_model.py",
    "src/pinn_starlight_core/physics/forcing.py",
    "src/pinn_starlight_core/nn/pinn_layers.py",
    "src/pinn_starlight_core/nn/pinn_loss.py",
)


def _evaluate_stage_b_for_pinn_selection(sample: dict, prediction: dict) -> dict:
    """A-compatible selection metrics with Stage-B star-reference semantics.

    Stage-B reference star coordinates are frozen from the native-background-
    flattened camera reference, while local-background-subtracted aperture flux
    is measured on the Stage-B clean target (base + protected structure).
    This is intentional and differs from Stage10-A, so we validate that contract
    explicitly instead of pretending the reference is an A sample.
    """
    background_true = sample["background_true"]
    clean_true = sample["clean_true"]
    background_pred = prediction["background_pred"]
    residual_pred = prediction["residual_pred"]

    result = {
        "bg_mae": scores.mae(background_pred, background_true),
        "bg_rmse": scores.rmse(background_pred, background_true),
        "bg_psnr": scores.psnr(background_pred, background_true),
        "bg_ssim": scores.ssim(background_pred, background_true),
        "residual_mae": scores.mae(residual_pred, clean_true),
        "residual_rmse": scores.rmse(residual_pred, clean_true),
        "residual_psnr": scores.psnr(residual_pred, clean_true),
        "residual_ssim": scores.ssim(residual_pred, clean_true),
    }

    reference_info = sample.get("metadata", {}).get("star_reference", {})
    if int(reference_info.get("evaluation_schema_version", 0)) != common_metrics.EVALUATION_SCHEMA_VERSION:
        raise ValueError(
            "Stage-B star_reference schema mismatch; regenerate Stage-B data "
            "with the current generator"
        )

    # Stage-B-specific provenance contract.
    if reference_info.get("catalog_source") != "stage_b_base_clean_coordinates":
        raise ValueError(
            "Stage-B PINN selector expected star coordinates frozen from base_clean"
        )
    if reference_info.get("coordinate_source") != "native_background_flattened_camera_reference":
        raise ValueError(
            "Stage-B star coordinates must come from the native-background-"
            "flattened camera reference"
        )
    if reference_info.get("photometry_source") != "stage_b_clean_true_base_plus_protected_structure":
        raise ValueError(
            "Stage-B reference photometry must be measured on the Stage-B clean target"
        )

    detection_info = reference_info.get("detection_image", {})
    if detection_info.get("reference_mode") != common_metrics.REFERENCE_DETECTION_MODE:
        raise ValueError("Stage-B star_reference reference detection mode mismatch")
    expected_reference_rule = common_metrics.DETECTION_IMAGE_RULES[
        common_metrics.REFERENCE_DETECTION_MODE
    ]
    if detection_info.get("reference_rule") != expected_reference_rule:
        raise ValueError("Stage-B star_reference reference detection rule mismatch")

    photometry = reference_info.get("photometry", {})
    expected_photometry = common_metrics.default_photometry_config()
    for key, expected in expected_photometry.items():
        if photometry.get(key) != expected:
            raise ValueError(f"Stage-B star_reference photometry mismatch: {key}")

    reference_stars = reference_info.get("stars")
    if reference_stars is None:
        raise ValueError("Stage-B sample is missing frozen star_reference.stars")

    result.update(
        common_metrics.star_metrics(
            residual_pred,
            threshold=float(reference_info["threshold"]),
            fwhm=float(reference_info["fwhm"]),
            matching_radius=float(reference_info["matching_radius"]),
            reference_stars=reference_stars,
            reference_count_capped=bool(reference_info.get("count_capped", False)),
            photometry=photometry,
            detection_mode=common_metrics.FORMAL_DETECTION_MODE,
        )
    )
    return result


def main(*, force=False, device_name=None, laplacian_backend="autograd"):
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    validation_rows = utils.load_manifest(STAGE_B_MANIFEST, split="validation")
    if not validation_rows:
        raise ValueError("Stage-B validation split is empty")

    formal_config = stage_b_selection_config("pinn")
    selection = formal_config["selection"]
    base_config = dict(formal_config["training"])
    seed = int(selection["seed"])
    device = pinn_selector._resolve_device(device_name)
    laplacian_backend = pinn.resolve_laplacian_backend(laplacian_backend)
    utils.set_seed(seed)

    utils.save_run_metadata(output_root, {
        "experiment": "Stage10-B PINN validation selection",
        "manifest": utils.project_relative(STAGE_B_MANIFEST),
        "selection_config": formal_config,
        "validation_samples": [row["sample_id"] for row in validation_rows],
        "compute_device": str(device),
        "pinn_laplacian_backend": laplacian_backend,
        "evaluation_adapter": "stage_b_clean_target_star_reference",
    })

    initial_centers = pinn_selector._precompute_initial_centers(validation_rows, base_config)
    candidate_table = []

    # Stage-B reuses the Stage-A selection machinery, but its star-reference
    # semantics differ (coordinates frozen from base_clean, flux measured on the
    # Stage-B clean target).  The evaluator is therefore passed in explicitly
    # rather than monkey-patched into the shared metrics module.
    evaluator = _evaluate_stage_b_for_pinn_selection

    selected_config, _ = pinn_selector._select_stage(
        "T1_network",
        base_config,
        [{"hidden_dims": list(candidate)} for candidate in selection["network_candidates"]],
        validation_rows,
        output_root,
        seed,
        device,
        candidate_table,
        keep_states=False,
        laplacian_backend=laplacian_backend,
        initial_centers_by_sample=initial_centers,
        evaluator=evaluator,
    )
    selected_config, _ = pinn_selector._select_stage(
        "T2_physics_weight",
        selected_config,
        [{"physics_weight": value} for value in selection["physics_weight_candidates"]],
        validation_rows,
        output_root,
        seed,
        device,
        candidate_table,
        keep_states=False,
        laplacian_backend=laplacian_backend,
        initial_centers_by_sample=initial_centers,
        evaluator=evaluator,
    )
    selected_config, selected_result = pinn_selector._select_stage(
        "T3_sparse_weight",
        selected_config,
        [{"sparse_weight": value} for value in selection["sparse_weight_candidates"]],
        validation_rows,
        output_root,
        seed,
        device,
        candidate_table,
        keep_states=True,
        laplacian_backend=laplacian_backend,
        initial_centers_by_sample=initial_centers,
        evaluator=evaluator,
    )
    selected_config, selected_result = pinn_selector._continue_until_stable(
        selected_config,
        selected_result,
        validation_rows,
        output_root,
        device,
        candidate_table,
        selection,
        seed,
        laplacian_backend,
        initial_centers,
        evaluator=evaluator,
    )

    selected_config = dict(selected_config)
    selected_config["steps"] = int(max(row["step"] for row in selected_result["rows"]))
    selected_config = dict(pinn.normalized_config(selected_config))
    selected = {
        "config": selected_config,
        "selection_seed": seed,
        "selection_metric": selection["selection_metric"],
        "selection_score": selected_result["aggregate"],
    }

    artifact = provenance.build_selection_artifact(
        method="pinn",
        selection_config=formal_config,
        selected=selected,
        manifest=STAGE_B_MANIFEST,
        rows_by_split={"validation": validation_rows},
        source_files=SOURCE_FILES,
        metadata={
            "compute_device": str(device),
            "pinn_laplacian_backend": laplacian_backend,
            "evaluation_adapter": "stage_b_clean_target_star_reference",
        },
    )
    utils.save_rows_csv(output_root / "candidate_results.csv", candidate_table)
    utils.write_json(ARTIFACT, artifact)
    print(
        "Stage-B PINN selected:",
        f"hidden={selected_config['hidden_dims']}",
        f"physics={selected_config['physics_weight']}",
        f"sparse={selected_config['sparse_weight']}",
        f"steps={selected_config['steps']}",
        ARTIFACT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--laplacian-backend",
        choices=("autograd", "analytic"),
        default="autograd",
    )
    args = parser.parse_args()
    main(
        force=args.force,
        device_name=args.device,
        laplacian_backend=args.laplacian_backend,
    )
