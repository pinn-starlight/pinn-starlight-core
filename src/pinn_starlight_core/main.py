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


def build_layers(device):
    models = nn.Sequential(
        Layers.SkyglowLinear(2, 512),
        nn.Tanh(),
        Layers.SkyglowLinear(512, 64),
        nn.Tanh(),
        Layers.SkyglowLinear(64, 1),
    ).to(device)

    return models


def train_one(input_file: str, dir_output: str) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    loader = Loader.RAWLoader(input_file)
    coords, values, _, _ = loader.get_gray_data(device)

    models = build_layers(device)
    data_loss_fn = Loss.MSEData()
    physics_loss_fn = Loss.MSEPhysics()
    alpha_module = Alpha.Alpha().to(device)
    kernel_size = 31
    i_city_module = Icity.Icity(input_file, device, kernel_size).to(device)
    phy_weight = 0.4

    optimizer = optim.Adam(
        list(models.parameters()) +
        list(i_city_module.parameters()) +
        list(alpha_module.parameters()),
        lr=0.001,
    )

    loss_history = []
    physics_loss_history = []

    for step in tqdm(range(30000)):
        index = torch.randint(0, coords.shape[0], (216300,))
        batch_xy = coords[index].to(device).clone().requires_grad_(True)
        batch_I = values[index].to(device)
        alpha = alpha_module()

        predicted = models(batch_xy).squeeze().to(device)
        i_city = i_city_module(batch_xy, alpha)

        data_loss = data_loss_fn.forward(batch_I, predicted)
        physics_loss = physics_loss_fn.forward(batch_I, predicted, i_city, alpha, batch_xy)
        loss = data_loss + phy_weight * physics_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        physics_loss_history.append(physics_loss.item())

        if step % 1000 == 0:
            print(f"loss:{loss.item() * 1000:.6f}")
            print(f"physics_loss:{physics_loss.item() * 1000:.6f}")
            print(f"alpha:{alpha.item() * 1000:.6f}")

    base = os.path.splitext(os.path.basename(input_file))[0]
    fig, ax = plt.subplots()
    ax.plot(loss_history, label='total loss')
    ax.plot(physics_loss_history, label='physics loss')
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    ax.legend()
    fig.savefig(f'{dir_output}/{base}_loss.png')
    plt.close(fig)

    print('Done.')


if __name__ == '__main__':
    input_dir = '/workspace/data/origin'
    output_dir = '/workspace/data/trained'
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        train_one(os.path.join(input_dir, file), output_dir)
