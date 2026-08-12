import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.nn import functional as F
from tqdm import tqdm

from pinn_starlight_core.image import ImageData
from pinn_starlight_core.model import SkyglowModel, physics_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def train_one(path: str, output_dir: str, device: torch.device) -> None:
    base = Path(path).stem
    print(f"Processing {Path(path).name}...")

    image = ImageData(path)
    coordinates, brightness = image.training_data()
    model = SkyglowModel(image.gray, kernel_size=21).to(device)
    optimizer = optim.Adam(
        [
            {"params": model.background.parameters(), "lr": 1e-3},
            {"params": model.icity.parameters(), "lr": 1e-3},
            {"params": [model.raw_alpha], "lr": 1e-3},
        ]
    )

    for _ in tqdm(range(60_000)):
        index = torch.randint(coordinates.shape[0], (10_240,))
        batch_xy = coordinates[index].to(device).requires_grad_(True)
        observed = brightness[index].to(device)

        predicted = model(batch_xy)
        alpha = model.alpha
        city = model.icity(batch_xy, alpha)
        loss = F.mse_loss(predicted, observed) + 0.5 * physics_loss(
            predicted, city, alpha, batch_xy
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    prediction = torch.empty(coordinates.shape[0], device=device)
    with torch.inference_mode():
        for start in range(0, coordinates.shape[0], 50_000):
            batch = coordinates[start : start + 50_000].to(device)
            prediction[start : start + len(batch)] = model(batch)

    observed = brightness.reshape(image.height, image.width).numpy()
    predicted = prediction.reshape(image.height, image.width).cpu().numpy()
    residual = (observed - predicted).clip(0, 1)

    print(f"alpha: {model.alpha.item()}")
    plt.imsave(Path(output_dir) / f"{base}_observed.png", observed, cmap="gray")
    plt.imsave(Path(output_dir) / f"{base}_predicted.png", predicted, cmap="gray")
    plt.imsave(Path(output_dir) / f"{base}_residual.png", residual, cmap="gray")


if __name__ == "__main__":
    training_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {training_device}")

    input_dir = PROJECT_ROOT / "data" / "image"
    output_dir = PROJECT_ROOT / "experiments" / "outputs" / "photo_training"
    os.makedirs(output_dir, exist_ok=True)

    for input_file in sorted(path for path in input_dir.iterdir() if path.is_file()):
        train_one(str(input_file), str(output_dir), training_device)

    print("Done.")
