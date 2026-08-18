"""E4：固定真实星空图上的三方法定性对比。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from experiments.common.baselines import coordinate_pinn as pinn
from experiments.common.baselines import fft_gaussian as fft
from experiments.common.baselines import unet_small as unet
from experiments.common.utils import experiment_utils as utils
from experiments.common.utils import metrics

REAL_IMAGES = (str(utils.PROJECT_ROOT / "data/test/native_test.tif"),)
METHODS = ("fft_gaussian", "unet_small", "pinn")


def main():
    args = _parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pinn_config = json.loads(Path(args.pinn_config).read_text(encoding="utf-8"))
    e2_config = json.loads(Path(args.e2_config).read_text(encoding="utf-8"))
    checkpoint = utils.resolve_path(e2_config["unet_checkpoint"])
    unet_model, _ = unet.load_checkpoint_model(checkpoint, device)

    utils.save_run_metadata(
        output_root,
        {
            "experiment": "E4",
            "images": [utils.project_relative(path) for path in args.images],
            "downsample": args.downsample,
            "center_crop_size": args.crop_size,
            "pinn_config": utils.project_relative(args.pinn_config),
            "e2_config": utils.project_relative(args.e2_config),
            "fft_sigma": e2_config["fft_sigma"],
            "unet_checkpoint": e2_config["unet_checkpoint"],
            "pinn_seed": args.seed,
        },
    )

    rows = []
    for image_path in args.images:
        image_path = utils.resolve_path(image_path)
        observed = utils.load_gray_image(image_path, downsample=args.downsample)
        observed = _center_crop(observed, args.crop_size)
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
            seed=args.seed,
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
    print("真实图没有背景真值，请填写每张图的 inspection_checklist.md。")


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
    content = f"""# {image_name} 人工检查

本页只记录定性观察，不把真实图描述性统计当作有真值指标。

| 检查项 | FFT-Gaussian | U-Net-small | PINN | 备注 |
| --- | --- | --- | --- | --- |
| 大尺度梯度是否残留 |  |  |  |  |
| 星点是否被背景吸收 |  |  |  |  |
| 银河/薄云/地景是否被误判 |  |  |  |  |
| 是否有边缘伪影 |  |  |  |  |
| 是否出现负值 |  |  |  |  |
| 是否产生原图不存在的亮点 |  |  |  |  |

结论：
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
    parser = argparse.ArgumentParser(description="E4：固定真实图三方法对比")
    parser.add_argument("--images", nargs="+", default=list(REAL_IMAGES))
    parser.add_argument(
        "--pinn-config",
        default=str(utils.OUTPUT_ROOT / "e1_tuning/locked_pinn_config.json"),
    )
    parser.add_argument(
        "--e2-config",
        default=str(utils.OUTPUT_ROOT / "e2_synthetic/locked_e2_config.json"),
    )
    parser.add_argument("--output-root", default=str(utils.OUTPUT_ROOT / "e4_real"))
    parser.add_argument("--seed", type=int, default=utils.SEEDS[0])
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    main()
