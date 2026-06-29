"""PINN 光污染分离 — 主入口 (Screened Poisson, v2.0)"""
import torch
from torch import optim
from tqdm.notebook import tqdm

import pinn_starlight_core.nn.Layers as Layers
import pinn_starlight_core.nn.Losses as Loss
import pinn_starlight_core.data.RAWLoader as RAWLoader
import pinn_starlight_core.data.FakeRAW as FakeRAW
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 数据 ---
raw_loader = RAWLoader.RAWLoader()
raw_loader.from_array(FakeRAW.FakeRaw().get_fake_raw())
coords, values, W, H = raw_loader.get_gray_data(device=device)

# --- 模型 ---
layers = [
    Layers.SkyglowLinear(2, 512).to(device), Layers.SkyglowActivation(),
    Layers.SkyglowLinear(512, 64).to(device), Layers.SkyglowActivation(),
    Layers.SkyglowLinear(64, 1).to(device),
]
params = [p for l in layers if isinstance(l, Layers.SkyglowLinear) for p in l.parameters()]
optimizer = optim.Adam(params, lr=0.001)
ld, lp = Loss.MSEData(), Loss.MSEPhysics()

# --- I_city: 指数背景 bg=A*exp(-(x+y)/D), ∇²bg=2bg/D² ---
# Screened Poisson: ∇²I - αI + I_city = 0  →  I_city = (α - 2/D²)*bg
alpha, D_bg = 4.0, 0.7
bg = 0.3 * torch.exp(-(coords[:, 0] + coords[:, 1]) / D_bg).to(device)
I_city = (alpha - 2.0 / D_bg**2) * bg

phy_weight = 0.01
batch_size = max(1024, min(8192, coords.shape[0] // 200))
steps = max(2000, coords.shape[0] // 100)

for step in tqdm(range(steps)):
    idx = torch.randint(0, coords.shape[0], (batch_size,), device=device)
    batch_xy = coords[idx].clone().requires_grad_(True)
    batch_I = values[idx]

    a = batch_xy
    for layer in layers:
        a = layer.forward(a)
    I_pred = a.squeeze()

    I_city_b = I_city[idx]

    data_loss = ld.forward(batch_I, I_pred)
    phys_loss = lp.forward(batch_I, I_pred, I_city_b, alpha, batch_xy)
    loss = data_loss + phys_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}, data={data_loss.item():.6f}, phys={phys_loss.item():.6f}")
