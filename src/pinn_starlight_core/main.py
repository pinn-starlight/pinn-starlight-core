import torch
from torch import optim
from tqdm.notebook import tqdm

import pinn_starlight_core.nn.Layers as Layers
import pinn_starlight_core.nn.Losses as Loss
import pinn_starlight_core.data.RAWLoader as RAWLoader
import pinn_starlight_core.data.FakeRAW as FakeRAW

raw_loader = RAWLoader.RAWLoader()
raw_loader.from_array(FakeRAW.FakeRaw().get_fake_raw())
coords, values, W, H = raw_loader.get_raw_data()

layers = [
    Layers.SkyglowLinear(2, 512),
    Layers.SkyglowActivation(),
    Layers.SkyglowLinear(512, 64),
    Layers.SkyglowActivation(),
    Layers.SkyglowLinear(64, 1),
]

params = []
for layer in layers:
    if isinstance(layer, Layers.SkyglowLinear):
        params += list(layer.parameters())

optimizer = optim.Adam(params, lr=0.001)

ld = Loss.MSEData()
lp = Loss.MSEPhysics()

alpha = 9.0
bg = 0.3 * torch.cos(3.0 * coords[:, 0]) * torch.cos(3.0 * coords[:, 1])
I_city = (alpha - 18.0) * bg
phy_weight = 0.01

for step in tqdm(range((coords.shape[0] / 240).interger())):
    idx = torch.randint(0, coords.shape[0], (coords.shape[0] / 320,))
    batch_xy = coords[idx].clone().requires_grad_(True)
    batch_I = values[idx]

    a = batch_xy
    for layer in layers:
        a = layer.forward(a)
    I_pred = a.squeeze()

    data_loss = ld.forward(batch_I, I_pred)
    phys_loss = lp.forward(batch_I, I_pred, I_city[idx], alpha, phy_weight, batch_xy)
    loss = data_loss + phys_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {(step // 500) + 1}, data={data_loss.item() * 100:.6f}%, phys={phys_loss.item() * 100:.6f}%, total={loss.item() * 100:.6f}%")