from __future__ import annotations

import sys
from pathlib import Path

import torch
from tqdm import tqdm
from torch import optim

from pinn_starlight_core.data.image_loader import ImageLoader
from pinn_starlight_core.nn import physics_model
from pinn_starlight_core.nn import pinn_layers as layers
from pinn_starlight_core.nn import pinn_loss as losses

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = Path(f"{PROJECT_ROOT}/data/test")
STEPS = 1000
BATCH_SIZE = 1_024
PHYSICS_WEIGHT = 0.4
MODEL_LR = 1e-3
ICITY_LR = 1e-3
ALPHA_LR = 1e-4
ALPHA_INIT = 0.55
ALPHA_MIN = 0.4
ALPHA_MAX = 0.6
KERNEL_SIZE = 31


def single_train(input_file, device):
    loader = ImageLoader(str(input_file), device)
    coords, values, _, _ = loader.get_gray_data()

    model = layers.SkyglowMLP().to(device)
    city_source = physics_model.Icity(device, KERNEL_SIZE, loader).to(device)
    alpha_module = physics_model.Alpha(
        init=ALPHA_INIT,
        alpha_min=ALPHA_MIN,
        alpha_max=ALPHA_MAX,
    ).to(device)
    optimizer = optim.Adam(
        [
            {"params": model.parameters(), "lr": MODEL_LR},
            {"params": city_source.parameters(), "lr": ICITY_LR},
            {"params": alpha_module.parameters(), "lr": ALPHA_LR},
        ]
    )

    final_total_loss = 0.0
    final_data_loss = 0.0
    final_physics_loss = 0.0

    for _ in tqdm(range(STEPS), file=sys.stdout):
        index = torch.randint(0, coords.shape[0], (BATCH_SIZE,), device=device)
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
        total_loss = data_loss + PHYSICS_WEIGHT * physics_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        final_total_loss = total_loss.detach().item()
        final_data_loss = data_loss.detach().item()
        final_physics_loss = physics_loss.detach().item()

    sigma_x, sigma_y = city_source.get_sigma()
    return {
        "total_loss": final_total_loss,
        "data_loss": final_data_loss,
        "physics_loss": final_physics_loss,
        "alpha": alpha_module().detach().item(),
        "center_x": city_source.x.detach().item(),
        "center_y": city_source.y.detach().item(),
        "sigma_x": sigma_x.detach().item(),
        "sigma_y": sigma_y.detach().item(),
        "theta": city_source.get_theta().detach().item(),
    }


def print_summary(input_file, result):
    print(f"\n{input_file.name} | step {STEPS}")
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


def main() -> None:
    print("Hello PINN-Starlight-core")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {INPUT_DIR}")

    input_files = sorted(path for path in INPUT_DIR.iterdir() if path.is_file())
    if not input_files:
        raise FileNotFoundError(f"No input images found in: {INPUT_DIR}")

    for input_file in input_files:
        result = single_train(input_file, device)
        print_summary(input_file, result)


if __name__ == "__main__":
    main()
