"""Standalone coordinate-PINN demo.

This module is a self-contained smoke run over local images; it is **not** part
of the Stage-11 formal pipeline (see ``experiments/`` for that).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import optim
from tqdm import tqdm

from pinn_starlight_core.data.image_loader import ImageLoader
from pinn_starlight_core.nn import physics_model
from pinn_starlight_core.nn import pinn_layers as layers
from pinn_starlight_core.nn import pinn_loss as losses

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "test"
IMAGE_SUFFIXES = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".nef",
    ".cr2",
    ".dng",
    ".arw",
}


@dataclass(frozen=True)
class DemoConfig:
    """Hyperparameters of the standalone demo run.

    Defaults reproduce the Stage-10 demo settings; they are not tuned for the
    formal experiments and must not be reused as a benchmark configuration.
    """

    steps: int = 1_000
    batch_size: int = 1_024
    physics_weight: float = 0.4
    model_lr: float = 1e-3
    city_lr: float = 1e-3
    alpha_lr: float = 1e-4
    alpha_init: float = 0.55
    alpha_min: float = 0.4
    alpha_max: float = 0.6
    max_components: int = 3
    sigma_max: float = 0.55
    initial_sigma: float = 0.425


DEFAULT_DEMO_CONFIG = DemoConfig()


def single_train(input_file, device, config: DemoConfig = DEFAULT_DEMO_CONFIG):
    """Fit one image and return the final losses plus forcing diagnostics."""
    loader = ImageLoader(input_file, device)
    coords, values, _, _ = loader.get_gray_data()

    model = layers.SkyglowMLP().to(device)
    city_source = physics_model.EffectiveForcingMixture(
        device,
        loader=loader,
        max_components=config.max_components,
        sigma_max=config.sigma_max,
        initial_sigma=config.initial_sigma,
    ).to(device)
    alpha_module = physics_model.Alpha(
        init=config.alpha_init,
        alpha_min=config.alpha_min,
        alpha_max=config.alpha_max,
    ).to(device)
    optimizer = optim.Adam(
        [
            {"params": model.parameters(), "lr": config.model_lr},
            {"params": city_source.parameters(), "lr": config.city_lr},
            {"params": alpha_module.parameters(), "lr": config.alpha_lr},
        ]
    )

    final_total_loss = 0.0
    final_data_loss = 0.0
    final_physics_loss = 0.0

    for _ in tqdm(range(config.steps), file=sys.stdout):
        index = torch.randint(0, coords.shape[0], (config.batch_size,), device=device)
        batch_xy = coords[index].clone().requires_grad_(True)
        batch_observed = values[index]

        alpha = alpha_module()
        background_pred = model(batch_xy).squeeze(-1)
        city_pred = city_source(batch_xy, alpha)
        data_loss = losses.mse_data(batch_observed, background_pred)
        physics_loss = losses.mse_physics(
            background_pred,
            city_pred,
            alpha,
            batch_xy,
        )
        total_loss = data_loss + config.physics_weight * physics_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        final_total_loss = total_loss.detach().item()
        final_data_loss = data_loss.detach().item()
        final_physics_loss = physics_loss.detach().item()

    components = city_source.diagnostics(alpha_module().detach())
    strongest = max(components, key=lambda item: item["weight"])
    return {
        "total_loss": final_total_loss,
        "data_loss": final_data_loss,
        "physics_loss": final_physics_loss,
        "alpha": alpha_module().detach().item(),
        "center_x": strongest["center_x"],
        "center_y": strongest["center_y"],
        "sigma_x": strongest["sigma_x"],
        "sigma_y": strongest["sigma_y"],
        "theta": strongest["theta"],
        "components": components,
    }


def print_summary(input_file, result, steps):
    print(f"\n{input_file.name} | step {steps}")
    print(
        f"loss={result['total_loss']:.8f} | "
        f"data={result['data_loss']:.8f} | "
        f"physics={result['physics_loss']:.8f}"
    )
    print(
        f"alpha={result['alpha']:.4f} | "
        f"center=({result['center_x']:.4f}, {result['center_y']:.4f}) | "
        f"sigma=({result['sigma_x']:.4f}, {result['sigma_y']:.4f}) | "
        f"theta={result['theta']:.4f}"
    )


def _collect_images(input_dir: Path):
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}\n"
            "Pass --input-dir PATH to point at a folder of images."
        )
    input_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not input_files:
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        raise FileNotFoundError(
            f"No input images found in: {input_dir}\nSupported formats: {supported}"
        )
    return input_files


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="pinn-starlight-core",
        description="Standalone coordinate-PINN demo over the images in a folder.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"folder of input images (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DemoConfig.steps,
        help=f"optimization steps per image (default: {DemoConfig.steps})",
    )
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_files = _collect_images(Path(args.input_dir))
    config = DemoConfig(steps=args.steps)

    print(f"demo input: {Path(args.input_dir)} ({len(input_files)} image(s), device={device})")
    for input_file in input_files:
        result = single_train(input_file, device, config)
        print_summary(input_file, result, config.steps)


if __name__ == "__main__":
    main()
