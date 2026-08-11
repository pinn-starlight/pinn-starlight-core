# Previous implementation retained as comments for reference.
#
# from __future__ import annotations
#
# import os
# import sys
# from pathlib import Path
#
# import matplotlib.pyplot as plt
# import torch
# from torch import Tensor, optim
# from torch.nn import functional as F
# from tqdm import tqdm
#
# ROOT = Path(__file__).resolve().parents[2]
# if str(ROOT / "src") not in sys.path:
#     sys.path.insert(0, str(ROOT / "src"))
#
# from pinn_starlight_core.image import ImageData
# from pinn_starlight_core.model import SkyglowModel, physics_loss
#
#
# IMAGE_PATH = "data/image/f7.5.tif"
# OUTPUT_DIR = "experiments/outputs/e1_learnable_alpha"
# HIDDEN_DIMS = [128, 128]
# KERNEL_SIZE = 31
# BATCH_SIZE = 8192
# MAX_STEPS = 3000
# PHYSICS_WEIGHT = 0.4
# MODEL_LR = 1e-3
# ICITY_LR = 1e-3
# ALPHA_MODE = "learnable"
# ALPHA_VALUE = 0.5
# ALPHA_INIT = 0.55
# ALPHA_MIN = 0.4
# ALPHA_MAX = 0.6
# ALPHA_LR = 1e-4
# RENDER_CHUNK_SIZE = 50_000
# SEED = 20260728
# LOG_INTERVAL = 50
#
#
# def load_data() -> tuple[ImageData, Tensor, Tensor]:
#     image = ImageData(ROOT / IMAGE_PATH)
#     coordinates, brightness = image.training_data()
#     return image, coordinates, brightness
#
#
# def create_optimizer(model: SkyglowModel) -> optim.Optimizer:
#     parameter_groups = [
#         {"params": model.background.parameters(), "lr": MODEL_LR},
#         {"params": model.city_source.parameters(), "lr": ICITY_LR},
#     ]
#     if model.raw_alpha is not None:
#         parameter_groups.append({"params": [model.raw_alpha], "lr": ALPHA_LR})
#     return optim.Adam(parameter_groups)
#
#
# def train_one_experiment() -> tuple[
#     dict[str, list[float]], Tensor, Tensor, int, int, SkyglowModel
# ]:
#     torch.manual_seed(SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#
#     image, coordinates, brightness = load_data()
#     if ALPHA_MODE not in {"fixed", "learnable"}:
#         raise ValueError("ALPHA_MODE must be 'fixed' or 'learnable'")
#     model = SkyglowModel(
#         image.gray,
#         hidden_dims=tuple(HIDDEN_DIMS),
#         kernel_size=KERNEL_SIZE,
#         alpha_init=ALPHA_INIT if ALPHA_MODE == "learnable" else ALPHA_VALUE,
#         alpha_min=ALPHA_MIN,
#         alpha_max=ALPHA_MAX,
#         learnable_alpha=ALPHA_MODE == "learnable",
#     ).to(device)
#     optimizer = create_optimizer(model)
#
#     history = {
#         "loss": [],
#         "data_loss": [],
#         "physics_loss": [],
#         "alpha": [],
#         "sigma_x": [],
#         "sigma_y": [],
#         "theta": [],
#     }
#
#     for step in tqdm(range(MAX_STEPS), desc="Training"):
#         index = torch.randint(0, coordinates.shape[0], (BATCH_SIZE,), device="cpu")
#         batch_coordinates = coordinates[index].to(device=device).detach().requires_grad_(True)
#         batch_brightness = brightness[index].to(device=device)
#         alpha = model.alpha
#
#         predicted = model(batch_coordinates)
#         city_brightness = model.city_source(batch_coordinates, alpha)
#         data_loss = F.mse_loss(predicted, batch_brightness)
#         pde_loss = physics_loss(predicted, city_brightness, alpha, batch_coordinates)
#         loss = data_loss + PHYSICS_WEIGHT * pde_loss
#
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         optimizer.step()
#
#         if step % LOG_INTERVAL == 0 or step == MAX_STEPS - 1:
#             sigma_x, sigma_y = model.city_source.sigma
#             history["loss"].append(loss.detach().cpu().item())
#             history["data_loss"].append(data_loss.detach().cpu().item())
#             history["physics_loss"].append(pde_loss.detach().cpu().item())
#             history["alpha"].append(alpha.detach().cpu().item())
#             history["sigma_x"].append(sigma_x.detach().cpu().item())
#             history["sigma_y"].append(sigma_y.detach().cpu().item())
#             history["theta"].append(model.city_source.theta.detach().cpu().item())
#
#     return history, coordinates, brightness, image.width, image.height, model
#
#
# def render_full_prediction(model: SkyglowModel, coordinates: Tensor, device: torch.device) -> Tensor:
#     prediction = torch.empty(coordinates.shape[0], device=device)
#     with torch.inference_mode():
#         for start in range(0, coordinates.shape[0], RENDER_CHUNK_SIZE):
#             end = start + RENDER_CHUNK_SIZE
#             batch_coordinates = coordinates[start:end].to(device)
#             prediction[start:end] = model(batch_coordinates)
#     return prediction
#
#
# def save_outputs(
#     history: dict[str, list[float]],
#     coordinates: Tensor,
#     brightness: Tensor,
#     width: int,
#     height: int,
#     model: SkyglowModel,
# ) -> None:
#     output_dir = ROOT / OUTPUT_DIR
#     output_dir.mkdir(parents=True, exist_ok=True)
#     device = next(model.parameters()).device
#     prediction = render_full_prediction(model, coordinates, device)
#
#     observed = brightness.reshape(height, width).cpu().numpy()
#     predicted = prediction.reshape(height, width).cpu().numpy()
#     residual = observed - predicted
#     residual_limit = max(float(abs(residual).max()), 1e-6)
#
#     plt.imsave(output_dir / "observed.png", observed, cmap="gray", vmin=0, vmax=1)
#     plt.imsave(output_dir / "predicted.png", predicted, cmap="gray", vmin=0, vmax=1)
#     plt.imsave(output_dir / "residual.png", residual, cmap="coolwarm", vmin=-residual_limit, vmax=residual_limit)
#
#     figure, axes = plt.subplots(3, 1, figsize=(8, 12))
#     axes[0].plot(history["loss"], label="total loss")
#     axes[0].plot(history["data_loss"], label="data loss")
#     axes[0].plot(history["physics_loss"], label="physics loss")
#     axes[0].set_xlabel("log step")
#     axes[0].set_ylabel("loss")
#     axes[0].legend()
#
#     axes[1].plot(history["alpha"], color="green")
#     axes[1].set_xlabel("log step")
#     axes[1].set_ylabel("alpha")
#
#     axes[2].plot(history["sigma_x"], label="sigma_x")
#     axes[2].plot(history["sigma_y"], label="sigma_y")
#     axes[2].set_xlabel("log step")
#     axes[2].set_ylabel("sigma")
#     axes[2].legend()
#
#     figure.tight_layout()
#     figure.savefig(output_dir / "training_curves.png")
#     plt.close(figure)
#
#     sigma_x, sigma_y = model.city_source.sigma
#     summary = {
#         "image": IMAGE_PATH,
#         "steps": MAX_STEPS,
#         "batch_size": BATCH_SIZE,
#         "alpha_mode": ALPHA_MODE,
#         "alpha": model.alpha.item(),
#         "sigma_x": sigma_x.item(),
#         "sigma_y": sigma_y.item(),
#         "theta": model.city_source.theta.item(),
#         "width": width,
#         "height": height,
#     }
#     (output_dir / "summary.txt").write_text(
#         "\n".join(f"{key}: {value}" for key, value in summary.items()) + "\n",
#         encoding="utf-8",
#     )
#     torch.save(
#         {
#             "model": model.state_dict(),
#             "summary": summary,
#             "history": history,
#         },
#         output_dir / "model.pt",
#     )
#
#
# def main() -> None:
#     history, coordinates, brightness, width, height, model = train_one_experiment()
#     save_outputs(history, coordinates, brightness, width, height, model)
#     print(f"Saved outputs to: {ROOT / OUTPUT_DIR}")
#
#
# if __name__ == "__main__":
#     os.chdir(ROOT)
#     main()
