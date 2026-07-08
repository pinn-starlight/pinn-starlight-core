from pinn_starlight_core.nn import Icity
import os
import torch
from torch import optim
from tqdm.notebook import tqdm

import pinn_starlight_core.nn.Layers as Layers
import pinn_starlight_core.nn.Losses as Loss
import pinn_starlight_core.data.PhotoLoader as RAWLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dir  = '/workspace/data/origin'

for file in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, file)
    base = file.rsplit('.', 1)[0]

    loader = RAWLoader.RAWLoader()
    loader.load(path)
    coords, values, W, H = loader.get_gray_data(device)

    layers = [
        Layers.SkyglowLinear(2, 512).to(device),
        Layers.SkyglowActivation(),
        Layers.SkyglowLinear(512, 64).to(device),
        Layers.SkyglowActivation(),
        Layers.SkyglowLinear(64, 1).to(device),
    ]

    params = []
    for layer in layers:
        if isinstance(layer, Layers.SkyglowLinear):
            params += list(layer.parameters())

    ld = Loss.MSEData()
    lp = Loss.MSEPhysics()

    alpha = 0.5

    I_city_module = Icity.Icity(path, device).to(device)
    phy_weight = 0.1

    optimizer = optim.Adam(
        params + list(I_city_module.parameters()),
        lr = 0.001
    )

    for step in tqdm(range(int(coords.shape[0] / 330))):
        idx = torch.randint(0, coords.shape[0], (int(coords.shape[0] / 420),))
        batch_xy = coords[idx].to(device).clone().requires_grad_(True)
        batch_I = values[idx].to(device)

        a = batch_xy
        for layer in layers:
            a = layer.forward(a)
        I_pred = a.squeeze().to(device)

        I_city_vals = I_city_module(batch_xy)

        data_loss = ld.forward(batch_I, I_pred)
        phys_loss = lp.forward(batch_I, I_pred, I_city_vals, alpha, batch_xy)
        loss = data_loss + phy_weight * phys_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss:{loss}")

print('Done.')
