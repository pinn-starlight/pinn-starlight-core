"""E0-E4 共用的坐标 PINN 背景估计。"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import pinn_starlight_core.nn.pinn_loss as loss
from experiments.common.utils import experiment_utils as utils
from pinn_starlight_core.nn.physics_model import Icity
from pinn_starlight_core.nn.pinn_layers import SkyglowMLP

DEFAULT_CONFIG = {
    "hidden_dims": [128, 128],
    "physics_weight": 0.1,
    "kernel_size": 31,
    "steps": 3000,
    "batch_size": 8192,
    "model_lr": 1e-3,
    "icity_lr": 1e-3,
    "alpha": 0.5,
    # The bright-region estimate is physically interpretable and more stable
    # than an unconstrained center under the current weak PDE auxiliary loss.
    # Keep learning available as an explicit E3 ablation instead of making it
    # the implicit formal setting.
    "center_mode": "bright_init_fixed",
    "prediction_batch_size": 65536,
    "log_every": 100,
}


def train_background(
    observed,
    config=None,
    device=None,
    seed=20260728,
    resume_state=None,
    show_progress=True,
    return_state=False,
):
    """逐图优化 PINN，返回预测、曲线、参数与可继续训练的状态。"""
    config = normalized_config(config)
    if config["alpha"] != 0.5:
        raise ValueError("正式实验要求 alpha 固定为 0.5")

    utils.set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    utils.reset_peak_vram(device)
    observed = _load_observed(observed)
    height, width = observed.shape
    coords = _coordinate_grid(height, width, device)
    values = torch.from_numpy(observed.reshape(-1)).to(device)

    model = SkyglowMLP(config["hidden_dims"]).to(device)
    city_source = Icity(
        device=device,
        kernel_size=config["kernel_size"],
        gray_img=observed,
        center_mode=config["center_mode"],
    ).to(device)
    optimizer = _optimizer(model, city_source, config)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    start_step = 0
    history = []
    elapsed_before = 0.0
    if resume_state is not None:
        model.load_state_dict(resume_state["model_state"])
        city_source.load_state_dict(resume_state["city_state"])
        optimizer.load_state_dict(resume_state["optimizer_state"])
        start_step = int(resume_state["step"])
        history = list(resume_state.get("history", []))
        elapsed_before = float(resume_state.get("runtime_s", 0.0))
        generator.set_state(resume_state["generator_state"])

    alpha = torch.tensor(float(config["alpha"]), dtype=torch.float32, device=device)
    steps = int(config["steps"])
    batch_size = min(int(config["batch_size"]), coords.shape[0])
    log_every = max(1, int(config["log_every"]))
    progress_bar = tqdm(range(steps), desc="PINN", leave=False) if show_progress else None
    progress = progress_bar if progress_bar is not None else range(steps)

    started = time.perf_counter()
    final_total = final_data = final_physics = float("nan")
    model.train()
    city_source.train()
    for local_step in progress:
        indices = torch.randint(
            0,
            coords.shape[0],
            (batch_size,),
            device=device,
            generator=generator,
        )
        batch_xy = coords[indices].clone().requires_grad_(True)
        batch_observed = values[indices]
        background_pred = torch.sigmoid(model(batch_xy).squeeze(-1))
        data_loss = loss.mse_data(batch_observed, background_pred)

        if config["physics_weight"] > 0:
            city_pred = city_source(batch_xy, alpha)
            physics_loss = loss.mse_physics(
                background_pred,
                city_pred,
                alpha,
                batch_xy,
            )
        else:
            physics_loss = torch.zeros((), device=device)
        total_loss = data_loss + float(config["physics_weight"]) * physics_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        final_total = float(total_loss.detach().item())
        final_data = float(data_loss.detach().item())
        final_physics = float(physics_loss.detach().item())
        global_step = start_step + local_step + 1
        if global_step == 1 or global_step % log_every == 0 or local_step + 1 == steps:
            history.append(_history_row(global_step, final_total, final_data, final_physics, city_source))
        if progress_bar is not None and local_step % 20 == 0:
            progress_bar.set_postfix(loss=f"{final_total:.6f}")

    runtime_s = elapsed_before + time.perf_counter() - started
    background_pred = _predict_background(
        model,
        coords,
        height,
        width,
        int(config["prediction_batch_size"]),
    )
    residual_pred = observed - background_pred
    sigma_x, sigma_y = city_source.get_sigma()
    final_step = start_step + steps

    result = {
        "observed": observed,
        "background_pred": background_pred,
        "residual_pred": residual_pred,
        "history": history,
        "step": final_step,
        "runtime_s": float(runtime_s),
        "peak_vram_mb": utils.peak_vram_mb(device),
        "total_loss": final_total,
        "data_loss": final_data,
        "physics_loss": final_physics,
        "alpha": float(config["alpha"]),
        "center_x": float(city_source.x.detach().item()),
        "center_y": float(city_source.y.detach().item()),
        "sigma_x": float(sigma_x.detach().item()),
        "sigma_y": float(sigma_y.detach().item()),
        "theta": float(city_source.get_theta().detach().item()),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters())
        + sum(parameter.numel() for parameter in city_source.parameters()),
        "config": config,
        "seed": int(seed),
    }
    if return_state:
        result["state"] = _to_cpu(
            {
                "model_state": model.state_dict(),
                "city_state": city_source.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "step": final_step,
                "history": history,
                "runtime_s": runtime_s,
            }
        )
    return result


def single_train(input_path, device, step, batch_size):
    """兼容 E0 的简短入口。"""
    config = dict(DEFAULT_CONFIG)
    config.update({"steps": int(step), "batch_size": int(batch_size)})
    result = train_background(input_path, config=config, device=device, return_state=False)
    return result["observed"], result["background_pred"], result["residual_pred"]


def normalized_config(config=None):
    result = dict(DEFAULT_CONFIG)
    if config:
        result.update(config)
    result["hidden_dims"] = [int(value) for value in result["hidden_dims"]]
    result["steps"] = int(result["steps"])
    result["batch_size"] = int(result["batch_size"])
    result["kernel_size"] = int(result["kernel_size"])
    if result["center_mode"] not in {
        "origin_fixed",
        "bright_init_fixed",
        "bright_init_learnable",
    }:
        raise ValueError(f"未知 center_mode：{result['center_mode']}")
    if result["steps"] <= 0 or result["batch_size"] <= 0:
        raise ValueError("steps 和 batch_size 必须大于 0")
    if result["kernel_size"] <= 1 or result["kernel_size"] % 2 == 0:
        raise ValueError("kernel_size 必须是大于 1 的奇数")
    if result["physics_weight"] < 0:
        raise ValueError("physics_weight 不能小于 0")
    return result


def save_checkpoint(path, result) -> None:
    """保存可继续训练的 PINN 状态。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state": result["state"],
            "config": result["config"],
            "seed": result["seed"],
        },
        path,
    )


def _to_cpu(value):
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _optimizer(model, city_source, config):
    groups = [{"params": model.parameters(), "lr": float(config["model_lr"])}]
    city_parameters = [
        parameter for parameter in city_source.parameters() if parameter.requires_grad
    ]
    if config["physics_weight"] > 0 and city_parameters:
        groups.append({"params": city_parameters, "lr": float(config["icity_lr"])})
    return torch.optim.Adam(groups)


def _load_observed(observed):
    if isinstance(observed, (str, Path)):
        array = utils.load_gray_image(observed)
    else:
        array = np.asarray(observed, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"observed 必须是二维灰度图，实际为 {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("observed 包含 NaN 或 Inf")
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def _coordinate_grid(height, width, device):
    x = torch.linspace(-1.0, 1.0, width, device=device)
    y = torch.linspace(-1.0, 1.0, height, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)


def _predict_background(model, coords, height, width, batch_size):
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            batch = coords[start : start + batch_size]
            predictions.append(torch.sigmoid(model(batch).squeeze(-1)).cpu())
    return torch.cat(predictions).reshape(height, width).numpy().astype(np.float32)


def _history_row(step, total, data, physics, city_source):
    sigma_x, sigma_y = city_source.get_sigma()
    return {
        "step": int(step),
        "total_loss": float(total),
        "data_loss": float(data),
        "physics_loss": float(physics),
        "center_x": float(city_source.x.detach().item()),
        "center_y": float(city_source.y.detach().item()),
        "sigma_x": float(sigma_x.detach().item()),
        "sigma_y": float(sigma_y.detach().item()),
        "theta": float(city_source.get_theta().detach().item()),
    }
