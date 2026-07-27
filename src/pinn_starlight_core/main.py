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
import pinn_starlight_core.nn.Alpha as A


def build_layers(device):
    models = nn.Sequential(
        Layers.SkyglowLinear(2, 128),
        nn.Tanh(),
        Layers.SkyglowLinear(128, 128),
        nn.Tanh(),
        Layers.SkyglowLinear(128, 1),
    ).to(device)

    if device.type == 'cuda' and torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
        models = nn.DataParallel(models)

    return models


def train_one(input_file: str, dir_output: str) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    loader = Loader.PhotoLoader(input_file)
    coords, values, _, _ = loader.get_gray_data(device)
    losses = Loss

    models = build_layers(device)
    alpha_module = A.Alpha(init=0.55, alpha_min=0.4, alpha_max=0.6).to(device)
    kernel_size = 31
    i_city_module = Icity.Icity(device, kernel_size, loader).to(device)
    phy_weight = 0.4

    optimizer = optim.Adam(
        [
            {'params': models.parameters(), 'lr': 1e-3},
            {'params': i_city_module.parameters(), 'lr': 1e-3},
            {'params': alpha_module.parameters(), 'lr': 1e-4}
        ]
    )

    loss_history = []
    physics_loss_history = []
    alpha_history = []
    sigma_x_history = []
    sigma_y_history = []

    for step in tqdm(range(50000)):
        index = torch.randint(0, coords.shape[0], (10240,), device=device)
        batch_xy = coords[index].clone().requires_grad_(True)
        batch_I = values[index]
        alpha = alpha_module()

        predicted = models(batch_xy).squeeze()
        i_city = i_city_module(batch_xy, alpha)

        data_loss = losses.mse_data(batch_I, predicted)
        physics_loss = losses.mse_physics(batch_I, predicted, i_city, alpha, batch_xy)
        loss = data_loss + phy_weight * physics_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sigma_x, sigma_y = i_city_module.get_sigma()

        loss_history.append(loss.item())
        physics_loss_history.append(physics_loss.item())
        alpha_history.append(alpha.item())
        sigma_x_history.append(sigma_x.item())
        sigma_y_history.append(sigma_y.item())

        if step % 1000 == 0:
            print(f"loss:{loss.item() * 1000:.6f}")
            print(f"physics_loss:{physics_loss.item() * 1000:.6f}")
            print(f"alpha:{alpha.item():.6f}")

    base = os.path.splitext(os.path.basename(input_file))[0]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    ax1.plot(loss_history, label='total loss')
    ax1.plot(physics_loss_history, label='physics loss')
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(alpha_history, color='green')
    ax2.set_xlabel('step')
    ax2.set_ylabel('alpha')

    ax3.plot(sigma_x_history, label='sigma_x')
    ax3.plot(sigma_y_history, label='sigma_y')
    ax3.set_xlabel('step')
    ax3.set_ylabel('sigma')
    ax3.legend()

    fig.tight_layout()
    fig.savefig(f'{dir_output}/{base}_{alpha_module.get_str()}.png')
    plt.close(fig)

    print('Done.')


if __name__ == '__main__':
    if torch.cuda.is_available():
        print(f'GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

    input_dir = '/workspace/data/origin'
    output_dir = '/workspace/data/trained'
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        train_one(os.path.join(input_dir, file), output_dir)
