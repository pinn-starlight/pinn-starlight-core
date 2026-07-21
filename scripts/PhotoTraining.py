from pinn_starlight_core.nn import Icity
import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import pinn_starlight_core.nn.Layers as Layers
import pinn_starlight_core.nn.Losses as Loss
import pinn_starlight_core.data.PhotoLoader as Loader
import pinn_starlight_core.nn.Alpha as Alpha


def build_model(input_device: torch.device):
    model = nn.Sequential(
        Layers.SkyglowLinear(2, 512),
        nn.Tanh(),
        Layers.SkyglowLinear(512, 64),
        nn.Tanh(),
        Layers.SkyglowLinear(64, 1),
    ).to(input_device)

    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
        model = nn.DataParallel(model)

    return model


def train_one(path: str, output_direct: str, input_device: torch.device):
    base = os.path.splitext(os.path.basename(path))[0]
    print(f'Processing {os.path.basename(path)}...')

    loader = Loader.RAWLoader(path)
    coords, values, W, H = loader.get_gray_data(input_device)

    model = build_model(input_device)
    data_loss_fn = Loss.MSEData()
    physics_loss_fn = Loss.MSEPhysics()
    alpha_module = Alpha.Alpha(0.5).to(input_device)
    kernel_size = 21
    i_city_module = Icity.Icity(path, input_device, kernel_size).to(input_device)
    phy_weight = 0.5

    optimizer = optim.Adam(
        list(model.parameters()) +
        list(i_city_module.parameters()) +
        list(alpha_module.parameters()),
        lr=0.001,
    )

    num_steps = int(coords.shape[0] * 0.00618 * 0.25)
    batch_size = int(coords.shape[0] * 0.00618 * 0.1)

    for _ in tqdm(range(num_steps)):
        idx = torch.randint(0, coords.shape[0], (batch_size,))
        batch_xy = coords[idx].to(input_device).clone().requires_grad_(True)
        batch_I = values[idx].to(input_device)
        alpha = alpha_module()

        predicted = model(batch_xy).squeeze().to(input_device)
        i_city = i_city_module(batch_xy, alpha)

        data_loss = data_loss_fn.forward(batch_I, predicted)
        physics_loss = physics_loss_fn.forward(batch_I, predicted, i_city, alpha, batch_xy)
        loss = data_loss + phy_weight * physics_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        full_pred = torch.empty(coords.shape[0], device=input_device)
        for start in range(0, coords.shape[0], 50000):
            end = min(start + 50000, coords.shape[0])
            full_pred[start:end] = model(coords[start:end]).squeeze()

    obs = values.reshape(H, W).cpu().numpy()
    pred = full_pred.reshape(H, W).cpu().numpy()
    res = (obs - pred).clip(0, 1)

    print(f'alpha: {alpha_module().item()}')
    plt.imsave(f'{output_direct}/{base}_observed.png', obs, cmap='gray')
    plt.imsave(f'{output_direct}/{base}_predicted.png', pred, cmap='gray')
    plt.imsave(f'{output_direct}/{base}_residual.png', res, cmap='gray')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

    input_dir = '/workspace/data/origin'
    output_dir = '/workspace/data/trained'
    os.makedirs(output_dir, exist_ok=True)

    for file in sorted(os.listdir(input_dir)):
        train_one(os.path.join(input_dir, file), output_dir, device)

    print('Done.')
