import torch
import torch.nn as nn
from pinn_starlight_core.data.FakeRAW import FakeRaw
from pinn_starlight_core.nn.Layers import SkyglowMLP
from pinn_starlight_core.nn.Losses import MSEData, MSEPhysics


# ---- 数据 ----
n_images = 8
all_images = []
for i in range(n_images):
    fake = FakeRaw(H=128, W=128, n_stars=5, seed=i)
    coords, values = fake.get_raw_data()
    all_images.append((coords, values))

bg = 0.3 * torch.cos(3.0 * coords[:, 0]) * torch.cos(3.0 * coords[:, 1])
I_city = (18.0 + 9.0) * bg       # 真 α=9 时的 I_city
batch_size = 1024

# ---- 可学习 α ----
# α 从 1.0 出发（远离真值 9.0），看能不能自己收敛到 9.0
# I_city 固定，PDE: ∇²I - α·I + 27·bg = 0

print("=== 可学习 alpha ===\n")
print(f"真值 alpha=9.0, I_city=27*bg\n")

for init_val, lr_alpha in [(1.0, 0.01), (5.0, 0.01)]:
    torch.manual_seed(42)

    mlp = SkyglowMLP([2, 64, 32, 1])
    alpha_param = nn.Parameter(torch.tensor([init_val]))

    all_params = list(mlp.parameters()) + [alpha_param]
    optimizer = torch.optim.Adam([
        {'params': mlp.parameters(), 'lr': 0.001},
        {'params': [alpha_param], 'lr': lr_alpha},       # α 自己的学习率可以大一点
    ])

    loss_data = MSEData()
    loss_physics = MSEPhysics()
    phy_weight = 0.01

    for epoch in range(80):
        for coords_img, I_obs in all_images:
            idx = torch.randperm(len(coords_img))[:batch_size]
            batch_xy = coords_img[idx].clone().requires_grad_(True)
            batch_I = I_obs[idx]

            I_pred = mlp.forward(batch_xy).squeeze()

            ld = loss_data.forward(batch_I, I_pred)
            lp = loss_physics.forward(batch_I, I_pred, I_city[idx],
                                      alpha_param, phy_weight, batch_xy)
            loss = ld + lp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"  起点={init_val:.1f}  Epoch {epoch+1:2d}  "
                  f"alpha={alpha_param.item():.4f}  (目标=9.0)")

    coords_test, I_test = all_images[0]
    with torch.no_grad():
        I_pred_test = mlp.forward(coords_test).squeeze()
        mse = ((I_test - I_pred_test) ** 2).mean().item()
        res = (I_test - I_pred_test).max().item()
    print(f"  最终 alpha={alpha_param.item():.4f}  MSE={mse:.6f}  残差max={res:.4f}\n")

# ---- 基线 ----
print("=== 基线（固定 alpha=9.0）===")
torch.manual_seed(42)
mlp_fixed = SkyglowMLP([2, 64, 32, 1])
opt_fixed = torch.optim.Adam(mlp_fixed.parameters(), lr=0.001)

for epoch in range(80):
    for coords_img, I_obs in all_images:
        idx = torch.randperm(len(coords_img))[:batch_size]
        batch_xy = coords_img[idx].clone().requires_grad_(True)

        I_pred = mlp_fixed.forward(batch_xy).squeeze()
        ld = loss_data.forward(I_obs[idx], I_pred)
        lp = loss_physics.forward(I_obs[idx], I_pred, I_city[idx], 9.0, phy_weight, batch_xy)
        (ld + lp).backward()
        opt_fixed.step()
        opt_fixed.zero_grad()

coords_test, I_test = all_images[0]
with torch.no_grad():
    I_pred_test = mlp_fixed.forward(coords_test).squeeze()
    mse = ((I_test - I_pred_test) ** 2).mean().item()
    res = (I_test - I_pred_test).max().item()
print(f"固定 alpha=9.0  MSE={mse:.6f}  残差max={res:.4f}")
