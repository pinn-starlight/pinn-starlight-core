"""E4 真实图片对比."""


import json
import time
from pathlib import Path
import numpy as np
import torch
from matplotlib import pyplot as plt
from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.baselines import fft_gaussian as fft
from experiments.common.baselines import unet_small as unet
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics


FORCE_OVERWRITE_OUTPUT = False

NATIVE_REAL_IMAGE = str(utils.PROJECT_ROOT / "data/test/native_test.tif")
COLLECTION_ROOT = utils.PROJECT_ROOT / "data/collections"
COLLECTION_MANIFEST = utils.PROJECT_ROOT / "data/collections/manifest.csv"
METHODS = ("fft_gaussian", "unet_small", "pinn")
PINN_CONFIG = utils.OUTPUT_ROOT / "e1_tuning/locked_pinn_config.json"
E2_CONFIG = utils.OUTPUT_ROOT / "e2_benchmark/locked_e2_config.json"
OUTPUT_ROOT = utils.OUTPUT_ROOT / "e4_real"
SEED = utils.SEEDS[0]
DOWNSAMPLE = 2
CROP_SIZE = 0


def _default_real_images():
    """
        返回真实图
    """
    if not Path(NATIVE_REAL_IMAGE).is_file():
        raise FileNotFoundError(f"E4 主图不存在：{NATIVE_REAL_IMAGE}")
    if not COLLECTION_MANIFEST.is_file():
        raise FileNotFoundError(
            "E4 需要数据清单，但未找到 "
            f"{COLLECTION_MANIFEST}；请同步 data/collections"
        )

    rows = utils.load_manifest(COLLECTION_MANIFEST)
    candidates = sorted(
        (row for row in rows if row.get("usage") == "e4_candidate"),
        key=lambda row: row["image_id"],
    )
    if not candidates:
        raise ValueError(
            f"No usage=e4_candidate images in {COLLECTION_MANIFEST}"
        )

    images = [NATIVE_REAL_IMAGE]
    missing = []
    for row in candidates:
        current_path = Path(row["current_path"])
        if current_path.is_absolute():
            image_path = current_path
        elif current_path.parts[:2] == ("data", "collections"):
            image_path = utils.to_absolute_path(current_path)
        else:
            image_path = COLLECTION_ROOT / current_path
        if not image_path.is_file():
            missing.append(f"{row['image_id']}: {image_path}")
        images.append(str(image_path))
    if missing:
        raise FileNotFoundError(
            "E4 清单中有不存在的图像：\n" + "\n".join(missing)
        )
    return tuple(dict.fromkeys(images))


REAL_IMAGES = _default_real_images()


def main():
    print("E4")
    output_root = utils.prepare_output_root(OUTPUT_ROOT, force=FORCE_OVERWRITE_OUTPUT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_config = json.loads(Path(PINN_CONFIG).read_text(encoding="utf-8"))
    e2_config = json.loads(Path(E2_CONFIG).read_text(encoding="utf-8"))
    checkpoint = utils.to_absolute_path(e2_config["unet_checkpoint"])
    unet_model, unet_checkpoint_data = unet.load_checkpoint_model(checkpoint, device)

    utils.save_run_metadata(
        output_root,
        {
            "experiment": "E4",
            "images": [utils.project_relative(path) for path in REAL_IMAGES],
            "image_count": len(REAL_IMAGES),
            "candidate_manifest": utils.project_relative(COLLECTION_MANIFEST),
            "downsample": DOWNSAMPLE,
            "center_crop_size": CROP_SIZE,
            "pinn_config": utils.project_relative(PINN_CONFIG),
            "e2_config": utils.project_relative(E2_CONFIG),
            "fft_sigma": e2_config["fft_sigma"],
            "unet_checkpoint": e2_config["unet_checkpoint"],
            "unet_validation_loss": unet_checkpoint_data.get("validation_loss"),
            "pinn_seed": SEED,
            "force_overwrite_output": FORCE_OVERWRITE_OUTPUT,
        },
    )

    rows = []
    for image_index, image_path in enumerate(REAL_IMAGES, start=1):
        image_path = utils.to_absolute_path(image_path)
        print(
            f"E4 [{image_index}/{len(REAL_IMAGES)}] {image_path.name}",
            flush=True,
        )
        observed = utils.load_gray_image(image_path, downsample=DOWNSAMPLE)
        observed = _center_crop(observed, CROP_SIZE)
        image_name = image_path.stem
        predictions = {}

        utils.reset_peak_vram(device)
        started = time.perf_counter()
        fft_background = fft.estimate_background(observed, float(e2_config["fft_sigma"]))
        predictions["fft_gaussian"] = {
            "background_pred": fft_background,
            "residual_pred": observed - fft_background,
            "runtime_s": time.perf_counter() - started,
            "peak_vram_mb": None,
        }

        utils.reset_peak_vram(device)
        started = time.perf_counter()
        unet_background = unet.predict_with_model(
            unet_model,
            observed,
            device=device,
            tile_size=int(e2_config["unet_tile_size"]),
            overlap=int(e2_config["unet_overlap"]),
        )
        predictions["unet_small"] = {
            "background_pred": unet_background,
            "residual_pred": observed - unet_background,
            "runtime_s": time.perf_counter() - started,
            "peak_vram_mb": utils.peak_vram_mb(device),
        }

        pinn_result = pinn.train_background(
            observed,
            config=pinn_config,
            device=device,
            seed=SEED,
            show_progress=False
        )
        predictions["pinn"] = {
            "background_pred": pinn_result["background_pred"],
            "residual_pred": pinn_result["residual_pred"],
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
            rows.append(
                {
                    "image": image_name,
                    "method": method,
                    "runtime_s": prediction["runtime_s"],
                    "peak_vram_mb": prediction["peak_vram_mb"],
                    **statistics,
                }
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
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    axes[0, 0].imshow(observed, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 0].set_title("Observed")
    axes[1, 0].imshow(observed, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("Observed")

    titles = {
        "fft_gaussian": "FFT-Gaussian",
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

| Check | FFT-Gaussian | U-Net-small | PINN | Notes |
| --- | --- | --- | --- | --- |
| Large-scale gradient remains |  |  |  |  |
| Stars absorbed into background |  |  |  |  |
| Diffuse structure removed incorrectly |  |  |  |  |
| Edge artifacts |  |  |  |  |
| Negative values |  |  |  |  |
| New bright points |  |  |  |  |
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

if __name__ == '__main__':
    main()
