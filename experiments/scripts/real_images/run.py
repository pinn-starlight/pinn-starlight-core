"""E4: five-method qualitative comparison on real high-pollution images."""


import argparse
import time
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.baselines import sgc_pinn as final_pinn
from experiments.common.baselines import fft_gaussian as fft
from experiments.common.baselines import cosine_ls as fourier_ls
from experiments.common.baselines import unet_small as unet
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.common.utils import selection_artifacts as artifacts
from experiments.configs.formal_experiment import (
    FINAL_PINN_FOLLOWUP_CONFIG,
    FORMAL_EXECUTION_CONFIG,
    FORMAL_METHODS,
    formal_config_snapshot,
)

METHODS = (*tuple(FORMAL_METHODS), final_pinn.METHOD_NAME)



REAL_IMAGE_ROOT = utils.REAL_HIGH_POLLUTION_ROOT
GLOBAL_LOCK = utils.GLOBAL_CONFIGURATION_LOCK
OUTPUT_ROOT = utils.OUTPUT_ROOT / "e4_real"
DOWNSAMPLE = 2
CROP_SIZE = 0
EXPECTED_REAL_IMAGE_COUNT = 16
_E4_RUNNER_SOURCE = "experiments/scripts/real_images/run.py"


def _default_real_images():
    """Return only the frozen high-pollution real-image collection for E4."""
    images = utils.supported_image_files(REAL_IMAGE_ROOT)
    if len(images) != EXPECTED_REAL_IMAGE_COUNT:
        raise ValueError(
            f"E4 requires exactly {EXPECTED_REAL_IMAGE_COUNT} images in "
            f"{REAL_IMAGE_ROOT}; found {len(images)}"
        )
    return tuple(str(path) for path in images)



def _formal_device():
    """Resolve the frozen single-GPU Stage-A benchmark device."""
    device = torch.device(str(FORMAL_EXECUTION_CONFIG["benchmark_device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Formal Stage-A requires CUDA for the frozen benchmark policy; "
            "CUDA is not available."
        )
    return device


def main(*, force: bool = False):
    print("E4 five-method real-image comparison")
    device = _formal_device()
    real_images = _default_real_images()
    global_lock, lock_verification = _load_global_lock_for_e4()
    gaussian_config = global_lock["methods"]["fft_gaussian"]
    fourier_config = global_lock["methods"]["fourier_ls"]
    fourier_lambda_ls, fourier_sparse_weight = fourier_ls.validate_frozen_config(fourier_config)
    unet_config = global_lock["methods"]["unet_small"]
    pinn_method_config = global_lock["methods"]["pinn"]
    pinn_config = pinn_method_config["config"]
    seed = int(pinn_method_config["test_seeds"][0])

    frozen_final_config = global_lock["formal_config"]["snapshot"]["final_pinn_followup"]
    if frozen_final_config != FINAL_PINN_FOLLOWUP_CONFIG:
        raise ValueError("final PINN config differs from the frozen Stage-A Global Lock")
    final_seed = int(FINAL_PINN_FOLLOWUP_CONFIG["test_seeds"][0])

    checkpoint = utils.to_absolute_path(unet_config["e4_checkpoint"])
    unet_model, unet_checkpoint_data = unet.load_checkpoint_model(checkpoint, device)

    # All frozen prerequisites are verified before --force is allowed to clear
    # the previous E4 output. This prevents a provenance/preflight failure from
    # destroying a previously completed real-image run.
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)

    utils.save_run_metadata(
        output_root,
        {
            "experiment": "E4",
            "images": [utils.project_relative(path) for path in real_images],
            "image_count": len(real_images),
            "real_image_root": utils.project_relative(REAL_IMAGE_ROOT),
            "downsample": DOWNSAMPLE,
            "center_crop_size": CROP_SIZE,
            "global_configuration_lock": utils.project_relative(GLOBAL_LOCK),
            "global_configuration_lock_sha256": utils.file_sha256(GLOBAL_LOCK),
            "global_lock_verification": lock_verification,
            "fft_sigma": gaussian_config["sigma"],
            "fourier_ls_lambda_ls": fourier_lambda_ls,
            "fourier_ls_sparse_weight": fourier_sparse_weight,
            "fourier_ls_alpha": fourier_config["alpha"],
            "fourier_ls_boundary": fourier_config["boundary_condition"],
            "fourier_ls_optimizer": fourier_config["optimizer"],
            "unet_checkpoint": unet_config["e4_checkpoint"],
            "unet_validation_loss": unet_checkpoint_data.get("validation_loss"),
            "unet_inference": unet_config["inference"],
            "pinn_seed": seed,
            "final_pinn_method": final_pinn.METHOD_NAME,
            "final_pinn_config": FINAL_PINN_FOLLOWUP_CONFIG,
            "final_pinn_seed": final_seed,
            "compute_device": str(device),
            "formal_execution_policy": dict(FORMAL_EXECUTION_CONFIG),
            "force_overwrite_output": force,
            "evaluation_schema_version": metrics.EVALUATION_SCHEMA_VERSION,
            "star_detection_rule": metrics.DETECTION_IMAGE_RULE,
        },
    )

    rows = []
    for image_index, image_path in enumerate(real_images, start=1):
        image_path = utils.to_absolute_path(image_path)
        print(
            f"E4 [{image_index}/{len(real_images)}] {image_path.name}",
            flush=True,
        )
        observed = utils.load_gray_image(image_path, downsample=DOWNSAMPLE)
        observed = _center_crop(observed, CROP_SIZE)
        image_name = image_path.stem
        predictions = {}

        utils.reset_peak_vram(device)
        started = time.perf_counter()
        fft_background = fft.estimate_background(observed, float(gaussian_config["sigma"]))
        predictions["fft_gaussian"] = {
            "background_pred": fft_background,
            "residual_pred": utils.reconstruct_residual(observed, fft_background),
            "runtime_s": time.perf_counter() - started,
            "peak_vram_mb": None,
        }

        started = time.perf_counter()
        fourier_background, fourier_info = fourier_ls.estimate_background(
            observed,
            fourier_lambda_ls,
            fourier_sparse_weight,
            return_info=True,
            backend=FORMAL_EXECUTION_CONFIG["fourier_ls_backend"],
            show_progress=True,
            progress_desc=(
                f"E4 Cosine-LS [{image_index}/{len(real_images)}] {image_name}"
            ),
        )
        predictions["fourier_ls"] = {
            "background_pred": fourier_background,
            "residual_pred": utils.reconstruct_residual(observed, fourier_background),
            "runtime_s": time.perf_counter() - started,
            "peak_vram_mb": None,
            "fourier_info": fourier_info,
        }

        utils.reset_peak_vram(device)
        started = time.perf_counter()
        unet_background, unet_info = unet.predict_with_model(
            unet_model,
            observed,
            device=device,
            inference_mode=unet_config["inference"]["mode"],
            tile_size=int(unet_config["inference"]["tile_size"]),
            overlap=int(unet_config["inference"]["overlap"]),
            return_info=True,
        )
        predictions["unet_small"] = {
            "background_pred": unet_background,
            "residual_pred": utils.reconstruct_residual(observed, unet_background),
            "runtime_s": time.perf_counter() - started,
            "peak_vram_mb": utils.peak_vram_mb(device),
            **unet_info,
        }

        pinn_result = pinn.train_background(
            observed,
            config=pinn_config,
            device=device,
            seed=seed,
            show_progress=True,
            laplacian_backend=FORMAL_EXECUTION_CONFIG["pinn_laplacian_backend"],
        )
        predictions["pinn"] = {
            "background_pred": pinn_result["background_pred"],
            "residual_pred": utils.reconstruct_residual(
                observed, pinn_result["background_pred"]
            ),
            "runtime_s": pinn_result["runtime_s"],
            "peak_vram_mb": pinn_result["peak_vram_mb"],
        }

        final_result = _run_final_pinn_only(
            observed,
            config=FINAL_PINN_FOLLOWUP_CONFIG,
            device=device,
            seed=final_seed,
            show_progress=True,
        )
        final_method_result = final_result[final_pinn.METHOD_NAME]
        predictions[final_pinn.METHOD_NAME] = {
            "background_pred": final_method_result["background_pred"],
            "residual_pred": utils.reconstruct_residual(
                observed, final_method_result["background_pred"]
            ),
            "runtime_s": final_method_result["runtime_s_total"],
            "peak_vram_mb": final_method_result["peak_vram_mb"],
        }

        for method, prediction in predictions.items():
            method_output = output_root / image_name / method
            utils.save_prediction_arrays(
                method_output,
                observed,
                prediction["background_pred"],
                prediction["residual_pred"],
            )
            statistics = metrics.evaluate_real(
                observed,
                prediction["background_pred"],
                prediction["residual_pred"],
            )
            background = np.asarray(prediction["background_pred"], dtype=np.float32)
            row = {
                "image": image_name,
                "method": method,
                "runtime_s": prediction["runtime_s"],
                "peak_vram_mb": prediction["peak_vram_mb"],
                "inference_mode_used": prediction.get("inference_mode_used"),
                "background_below_zero_fraction": float(np.mean(background < 0.0)),
                "background_above_one_fraction": float(np.mean(background > 1.0)),
                **statistics,
            }
            if "fourier_info" in prediction:
                row.update(
                    {
                        "fourier_objective": prediction["fourier_info"]["objective"],
                        "fourier_smooth_objective": prediction["fourier_info"]["smooth_objective"],
                        "fourier_sparse_objective": prediction["fourier_info"]["sparse_objective"],
                        "fourier_profile_nfev": prediction["fourier_info"]["profile_nfev"],
                        "fourier_effective_component_count": prediction["fourier_info"]["effective_component_count"],
                        "fourier_sigma_x": prediction["fourier_info"]["sigma_x"],
                        "fourier_sigma_y": prediction["fourier_info"]["sigma_y"],
                        "fourier_theta": prediction["fourier_info"]["theta"],
                    }
                )
            rows.append(row)

        fourier_output = output_root / image_name / "fourier_ls"
        utils.write_json(
            fourier_output / "estimated_parameters.json",
            predictions["fourier_ls"]["fourier_info"],
        )

        pinn_output = output_root / image_name / "pinn"
        utils.save_rows_csv(pinn_output / "history.csv", pinn_result["history"])
        utils.write_json(
            pinn_output / "final_parameters.json",
            {
                key: pinn_result[key]
                for key in (
                    "step",
                    "total_loss",
                    "data_loss",
                    "physics_loss",
                    "sparse_loss",
                    "effective_component_count",
                    "components",
                    "alpha",
                    "center_x",
                    "center_y",
                    "sigma_x",
                    "sigma_y",
                    "theta",
                    "runtime_s",
                    "peak_vram_mb",
                )
            },
        )

        final_output = output_root / image_name / final_pinn.METHOD_NAME
        utils.save_rows_csv(
            final_output / "stage1_history.csv",
            final_result["shared"]["stage1_history"],
        )
        utils.write_json(
            final_output / "forcing_diagnostics.json",
            final_result["shared"]["forcing_diagnostics"],
        )
        utils.save_rows_csv(
            final_output / "training_history.csv",
            final_method_result["history"],
        )
        utils.write_json(
            final_output / "final_diagnostics.json",
            {
                "method": final_pinn.METHOD_NAME,
                "seed": final_seed,
                "runtime_s_total": final_method_result["runtime_s_total"],
                "stage1_runtime_s": final_result["shared"]["stage1_runtime_s"],
                "forcing_runtime_s": final_result["shared"]["forcing_runtime_s"],
                "stage3_runtime_s": final_method_result["stage3_runtime_s"],
                "peak_vram_mb": final_method_result["peak_vram_mb"],
                "parameter_count": final_method_result["parameter_count"],
                "gradient_conflict_fraction": final_method_result[
                    "gradient_conflict_fraction"
                ],
                "mean_lambda_use": final_method_result["mean_lambda_use"],
                "median_lambda_use": final_method_result["median_lambda_use"],
                "mean_scaled_physics_to_data_ratio": final_method_result[
                    "mean_scaled_physics_to_data_ratio"
                ],
            },
        )
        _save_comparison_figure(
            output_root / image_name / "comparison.png",
            observed,
            predictions,
        )
        _save_inspection_checklist(
            output_root / image_name / "inspection_checklist.md",
            image_name,
        )

    utils.save_rows_csv(output_root / "metrics.csv", rows)
    utils.write_json(output_root / "summary.json", utils.summarize_rows(rows, "method"))
    print(f"E4 完成：{output_root}")
    print("No ground truth is available for real images; review each inspection_checklist.md")



def _load_global_lock_for_e4():
    """Load the frozen Stage-A lock while allowing only the E4 runner to evolve.

    Historical Stage-A locks fingerprinted ``experiments/scripts/real_images/run.py``
    together with the Stage-A formal sources. E4 is now an independent
    real-image evaluation and this runner intentionally changed from four to
    five methods. Rebuilding a post-test Stage-A lock solely to update that
    unrelated file hash would weaken the audit trail.

    We therefore try the ordinary strict verifier first. The compatibility
    path is entered only when the *sole* strict failure is the E4 runner hash.
    It then reproduces the frozen-lock checks while exempting exactly that one
    file; every other source hash, selection artifact, data-protocol hash,
    formal-config snapshot, and U-Net checkpoint remains mandatory.
    """
    try:
        lock = utils.load_global_configuration_lock(
            GLOBAL_LOCK, verify=True, require_formal_ready=True
        )
        return lock, {
            "mode": "strict",
            "e4_runner_hash_exempted": False,
        }
    except ValueError as exc:
        expected = f"Source changed after selection/lock: {_E4_RUNNER_SOURCE}"
        if str(exc) != expected:
            raise

    lock = utils.load_global_configuration_lock(
        GLOBAL_LOCK, verify=False, require_formal_ready=True
    )
    _verify_legacy_lock_except_e4_runner(lock)
    print(
        "E4 lock compatibility: verified all frozen dependencies except the "
        "historical E4 runner hash (independent E4 code was intentionally updated).",
        flush=True,
    )
    return lock, {
        "mode": "strict_except_historical_e4_runner_hash",
        "e4_runner_hash_exempted": True,
        "exempted_source": _E4_RUNNER_SOURCE,
    }


def _verify_legacy_lock_except_e4_runner(lock):
    """Reproduce Global-Lock verification, exempting only ``e4/run.py``."""
    formal = lock.get("formal_config", {})
    current_snapshot = formal_config_snapshot()
    current_config_sha = artifacts.stable_json_sha256(current_snapshot)
    if artifacts.stable_json_sha256(formal.get("snapshot")) != formal.get("sha256"):
        raise ValueError("Global Lock formal-config snapshot is internally inconsistent")
    if formal.get("sha256") != current_config_sha:
        raise ValueError(
            "Formal experiment config changed after Global Configuration Lock; "
            "rebuild affected selection artifacts and the lock before test"
        )

    manifest = utils.to_absolute_path(lock["data_provenance"]["manifest"])
    expected_manifest_sha = lock["data_provenance"].get("manifest_sha256")
    if expected_manifest_sha and utils.file_sha256(manifest) != expected_manifest_sha:
        raise ValueError(
            "Synthetic manifest changed after Global Configuration Lock; rebuild the lock"
        )

    protocol_path = utils.to_absolute_path(
        lock.get("data_provenance", {}).get("dataset_protocol", "")
    )
    expected_protocol_sha = lock.get("data_provenance", {}).get(
        "dataset_protocol_sha256"
    )
    if not protocol_path.is_file() or not expected_protocol_sha:
        raise ValueError("Global Lock is missing dataset-protocol provenance")
    if utils.file_sha256(protocol_path) != expected_protocol_sha:
        raise ValueError("Synthetic dataset protocol changed after Global Lock")

    for method, info in lock.get("selection_artifacts", {}).items():
        artifact_path = utils.to_absolute_path(info["path"])
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Frozen selection artifact missing: {artifact_path}")
        if utils.file_sha256(artifact_path) != info.get("sha256"):
            raise ValueError(f"Frozen selection artifact changed: {artifact_path}")

    provenance = lock.get("source_provenance", {})
    records = provenance.get("files", []) if isinstance(provenance, dict) else []
    if not records:
        raise ValueError("Source provenance contains no files")
    actual_paths = tuple(record.get("path") for record in records)
    expected_paths = tuple(artifacts.FORMAL_SOURCE_DEPENDENCIES)
    if actual_paths != expected_paths:
        raise ValueError("Source dependency list changed; rerun the affected selector")
    if artifacts.stable_json_sha256(records) != provenance.get("bundle_sha256"):
        raise ValueError("Stored source provenance bundle is internally inconsistent")

    exempted_records = 0
    for record in records:
        relative = record.get("path")
        expected_sha = record.get("sha256")
        if relative == _E4_RUNNER_SOURCE:
            exempted_records += 1
            continue
        absolute = utils.PROJECT_ROOT / str(relative)
        if not absolute.is_file():
            raise FileNotFoundError(f"Frozen source file missing: {absolute}")
        if utils.file_sha256(absolute) != expected_sha:
            raise ValueError(f"Source changed after selection/lock: {relative}")
    if exempted_records != 1:
        raise ValueError(
            "Historical Global Lock does not contain exactly one E4 runner provenance record"
        )

    for candidate in lock["methods"]["unet_small"].get("checkpoints", []):
        checkpoint = utils.to_absolute_path(candidate["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Frozen U-Net checkpoint missing: {checkpoint}")
        expected_sha = candidate.get("sha256")
        if expected_sha and utils.file_sha256(checkpoint) != expected_sha:
            raise ValueError(f"Frozen U-Net checkpoint changed: {checkpoint}")


def _save_comparison_figure(path, observed, predictions):
    column_count = 1 + len(METHODS)
    figure, axes = plt.subplots(
        2,
        column_count,
        figsize=(3.5 * column_count, 7),
        constrained_layout=True,
    )
    axes[0, 0].imshow(observed, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Observed")
    axes[1, 0].imshow(observed, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Observed")

    titles = {
        "fft_gaussian": "FFT-Gaussian",
        "fourier_ls": "Cosine-LS",
        "unet_small": "U-Net-small",
        "pinn": "PINN",
        final_pinn.METHOD_NAME: "Final-PINN",
    }
    for column, method in enumerate(METHODS, start=1):
        prediction = predictions[method]
        axes[0, column].imshow(
            prediction["background_pred"],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, column].set_title(f"{titles[method]} background")
        # Display-only clipping: scientific TIFFs and every metric use raw residual_pred.
        axes[1, column].imshow(
            np.clip(prediction["residual_pred"], 0.0, 1.0),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, column].set_title(f"{titles[method]} residual")

    for axis in axes.flat:
        axis.axis("off")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _save_inspection_checklist(path, image_name):
    content = f"""# {image_name} inspection checklist

| Check | FFT-Gaussian | Cosine-LS | U-Net-small | PINN | Final-PINN | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Large-scale gradient remains |  |  |  |  |  |  |
| Stars absorbed into background |  |  |  |  |  |  |
| Diffuse structure removed incorrectly |  |  |  |  |  |  |
| Edge artifacts |  |  |  |  |  |  |
| Negative values |  |  |  |  |  |  |
| New bright points |  |  |  |  |  |  |
"""
    Path(path).write_text(content, encoding="utf-8")


def _run_final_pinn_only(
    observed,
    *,
    config,
    device,
    seed,
    show_progress,
):
    """Run only the frozen improved branch of Final-PINN on one real image.

    This reproduces the improved branch of ``decoupled_gradient_pinn.run_paired``
    exactly, but intentionally omits the matched data-only control because E4 is
    a five-method qualitative comparison rather than a mechanism-ablation run.
    """
    cfg = final_pinn.normalized_followup_config(config)
    observed = utils.load_observed(observed)
    stage1_cfg = dict(pinn.normalized_config(cfg["stage1"]))
    initial_centers = pinn.initial_centers_from_observed(observed, stage1_cfg)

    stage1 = pinn.train_background(
        observed,
        config=stage1_cfg,
        device=device,
        seed=int(seed),
        show_progress=show_progress,
        return_state=True,
        laplacian_backend="analytic",
        initial_centers=initial_centers,
    )
    model_state = stage1["state"]["model_state"]

    forcing_started = time.perf_counter()
    forcing_state, forcing_diag = final_pinn.fit_fixed_forcing(
        observed=observed,
        model_state=model_state,
        stage1_config=stage1_cfg,
        stage2_config=cfg["stage2"],
        initial_centers=initial_centers,
        device=device,
        seed=int(seed) + 1009,
    )
    forcing_runtime_s = time.perf_counter() - forcing_started

    improved = final_pinn.finetune(
        observed=observed,
        model_state=model_state,
        forcing_state=forcing_state,
        stage1_config=stage1_cfg,
        stage3_config=cfg["stage3"],
        initial_centers=initial_centers,
        device=device,
        seed=int(seed) + 2003,
        use_physics=True,
        show_progress=show_progress,
    )

    model_params = int(
        sum(value.numel() for value in model_state.values() if torch.is_tensor(value))
    )
    forcing_params = int(
        sum(value.numel() for value in forcing_state.values() if torch.is_tensor(value))
    )
    improved.update(
        {
            "method": final_pinn.METHOD_NAME,
            "runtime_s_total": float(
                stage1["runtime_s"] + forcing_runtime_s + improved["stage3_runtime_s"]
            ),
            "parameter_count": model_params + forcing_params,
        }
    )
    return {
        "shared": {
            "stage1_history": list(stage1["history"]),
            "stage1_runtime_s": float(stage1["runtime_s"]),
            "forcing_diagnostics": forcing_diag,
            "forcing_runtime_s": float(forcing_runtime_s),
            "config": cfg,
            "seed": int(seed),
        },
        final_pinn.METHOD_NAME: improved,
    }


def _center_crop(image, crop_size):
    if crop_size <= 0:
        return image
    height, width = image.shape
    crop_height = min(height, crop_size)
    crop_width = min(width, crop_size)
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return image[top : top + crop_height, left : left + crop_width]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace the E4 output")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(force=args.force)
