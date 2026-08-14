"""E0-E4 共用的坐标 PINN 背景估计 baseline。

职责约定：
- 复用 ``src/pinn_starlight_core`` 中的 ImageLoader、网络、物理模型和损失函数，
  不在实验脚本中复制训练循环或重新实现核心模型。
- 对单张 observed 图像执行逐图优化，返回 background_pred、residual_pred 以及
  loss、alpha、I_city 参数和耗时等实验记录。
- 本模块只负责可复用的训练与预测过程；E0-E4 负责选择数据、提供配置、固定种子、
  计算指标和保存结果。
- E0 只用少量步数检查流程能否运行；正式步数和超参数在 E1 锁定，随后供 E2-E4 复用。

TODO:
1. 定义清晰的配置和返回结果结构，避免依赖 main.py 中的全局常量。
2. 将 main.py 的单图训练流程整理成可调用函数，并支持固定或可学习 alpha。
3. 补充整图推理，将坐标分块送入模型，避免一次性推理导致显存不足。
4. 保留浮点 background_pred，不要在本模块中转成展示用图片或裁剪 residual_pred。
"""
import sys

import torch
from tqdm import tqdm

from pinn_starlight_core.data.image_loader import ImageLoader
from pinn_starlight_core.nn.pinn_layers import SkyglowMLP
from pinn_starlight_core.nn.physics_model import Alpha
from pinn_starlight_core.nn.physics_model import Icity
import pinn_starlight_core.nn.pinn_loss as loss

def single_train(input_path, device, step, batch_size):
    """对单张输入逐图优化，返回 observed、background_pred 和 residual_pred。"""
    loader = ImageLoader(input_path, device)
    coords, brightness, W, H = loader.get_gray_data()

    model = SkyglowMLP().to(device)
    i_city = Icity(device, loader=loader).to(device)
    alpha_module = Alpha().to(device)
    optimizer = torch.optim.Adam([
        {"params": model.parameters()},
        {"params": i_city.parameters()},
        {"params": alpha_module.parameters()},
    ])

    for _ in tqdm(range(step), file=sys.stdout):
        index = torch.randint(0, coords.shape[0], (batch_size,), device=device)
        batch_xy = coords[index].clone().requires_grad_(True)
        batch_observed = brightness[index]

        alpha = alpha_module()
        background_pred = model(batch_xy).squeeze(-1)
        i_city_pred = i_city(batch_xy, alpha)
        data_loss = loss.mse_data(batch_observed, background_pred)
        physics_loss = loss.mse_physics(background_pred, i_city_pred, alpha, batch_xy)
        total_loss = data_loss + physics_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

    observed = brightness.reshape(H, W).detach().cpu().numpy()
    predicted = _predict_background(model, coords, batch_size, H, W)
    residual = observed - predicted

    return observed, predicted, residual

def _predict_background(model, coords, batch_size, H, W):
    predicted = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            end = start + batch_size
            batch_xy = coords[start:end]

            batch_pred = model(batch_xy).squeeze(-1)
            predicted.append(batch_pred.cpu())

    background_pred = torch.cat(predicted).reshape(H, W).detach().cpu().numpy()

    return background_pred
    
