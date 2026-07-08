from pinn_starlight_core.nn import Icity
import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

import pinn_starlight_core.nn.Layers as Layers
import pinn_starlight_core.nn.Losses as Loss
import pinn_starlight_core.data.PhotoLoader as RAWLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')

input_dir  = '/workspace/data/origin'
output_dir = '/workspace/data/trained'
os.makedirs(output_dir, exist_ok=True)

for file in sorted(os.listdir(input_dir)):
    path = os.path.join(input_dir, file)
    base = file.rsplit('.', 1)[0]
    print(f'Processing {file}...')

    loader = RAWLoader.RAWLoader()
    loader.load(path)
    coords, values, W, H = loader.get_gray_data(device)

    model = torch.nn.Sequential(
        Layers.SkyglowLinear(2, 512),
        Layers.SkyglowActivation(),
        Layers.SkyglowLinear(512, 64),
        Layers.SkyglowActivation(),
        Layers.SkyglowLinear(64, 1),
    ).to(device)

    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs with DataParallel')
        model = torch.nn.DataParallel(model)

    ld = Loss.MSEData()
    lp = Loss.MSEPhysics()

    alpha = 0.5

    I_city_module = Icity.Icity(path, device).to(device)
    phy_weight = 0.1

    optimizer = optim.Adam(
        list(model.parameters()) + list(I_city_module.parameters()),
        lr = 0.001
    )

    for step in tqdm(range(int(coords.shape[0] * 0.00618 * 0.25))):
        idx = torch.randint(0, coords.shape[0], (int(coords.shape[0]  * 0.00618 * 0.1),))
        batch_xy = coords[idx].to(device).clone().requires_grad_(True)
        batch_I = values[idx].to(device)

        I_pred = model(batch_xy).squeeze().to(device)

        I_city_vals = I_city_module(batch_xy)

        data_loss = ld.forward(batch_I, I_pred)
        phys_loss = lp.forward(batch_I, I_pred, I_city_vals, alpha, batch_xy)
        loss = data_loss + phy_weight * phys_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        I_pred = torch.empty(coords.shape[0], device=device)
        for start in range(0, coords.shape[0], 50000):
            end = min(start + 50000, coords.shape[0])
            I_pred[start:end] = model(coords[start:end]).squeeze()

    obs = values.reshape(H, W).cpu().numpy()
    pred = I_pred.reshape(H, W).cpu().numpy()
    res = (obs - pred).clip(0, 1)

    print(f'alpha:{alpha},')
    plt.imsave(f'{output_dir}/{base}_observed.png', obs, cmap='gray')
    plt.imsave(f'{output_dir}/{base}_predicted.png', pred, cmap='gray')
    plt.imsave(f'{output_dir}/{base}_residual.png', res, cmap='gray')

print('Done.')
