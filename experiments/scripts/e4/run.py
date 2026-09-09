"""E4: Stage-10 qualitative comparison on real high-pollution images."""


import argparse
import time
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.baselines import fft_gaussian as fft
from experiments.common.baselines import fft_screened_poisson as fourier_ls
from experiments.common.baselines import unet_small as unet
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics
from experiments.configs.formal_experiment import FORMAL_EXECUTION_CONFIG, FORMAL_METHODS

METHODS = tuple(FORMAL_METHODS)



REAL_IMAGE_ROOT = utils.REAL_HIGH_POLLUTION_ROOT
GLOBAL_LOCK = utils.GLOBAL_CONFIGURATION_LOCK
OUTPUT_ROOT = utils.OUTPUT_ROOT / "e4_real"
DOWNSAMPLE = 2
CROP_SIZE = 0
EXPECTED_REAL_IMAGE_COUNT = 16


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
    """Resolve the frozen single-GPU Stage10-A benchmark device."""
    device = torch.device(str(FORMAL_EXECUTION_CONFIG["benchmark_device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Formal Stage10-A requires CUDA for the frozen benchmark policy; "
            "CUDA is not available."
        )
    return device


def main(*, force: bool = False):
    print("E4 Stage-10 real-image comparison")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=force)
    device = _formal_device()
    real_images = _default_real_images()
    global_lock = utils.load_global_configuration_lock(
        GLOBAL_LOCK, verify=True, require_formal_ready=True
    )
    gaussian_config = global_lock["methods"]["fft_gaussian"]
    fourier_config = global_lock["methods"]["fourier_ls"]
    fourier_lambda_ls, fourier_sparse_weight = fourier_ls.validate_frozen_config(fourier_config)
    unet_config = global_lock["methods"]["unet_small"]
    pinn_method_config = global_lock["methods"]["pinn"]
    pinn_config = pinn_method_config["config"]
    seed = int(pinn_method_config["test_seeds"][0])
    checkpoint = utils.to_absolute_path(unet_config["e4_checkpoint"])
    unet_model, unet_checkpoint_data = unet.load_checkpoint_model(checkpoint, device)

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
                f"E4 Fourier-LS [{image_index}/{len(real_images)}] {image_name}"
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


def _save_comparison_figure(path, observed, predictions):
    figure, axes = plt.subplots(2, 5, figsize=(17.5, 7), constrained_layout=True)
    axes[0, 0].imshow(observed, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Observed")
    axes[1, 0].imshow(observed, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Observed")

    titles = {
        "fft_gaussian": "FFT-Gaussian",
        "fourier_ls": "Fourier-LS",
        "unet_small": "U-Net-small",
        "pinn": "PINN",
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

| Check | FFT-Gaussian | Fourier-LS | U-Net-small | PINN | Notes |
| --- | --- | --- | --- | --- | --- |
| Large-scale gradient remains |  |  |  |  |  |
| Stars absorbed into background |  |  |  |  |  |
| Diffuse structure removed incorrectly |  |  |  |  |  |
| Edge artifacts |  |  |  |  |  |
| Negative values |  |  |  |  |  |
| New bright points |  |  |  |  |  |
"""
    Path(path).write_text(content, encoding="utf-8")


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
