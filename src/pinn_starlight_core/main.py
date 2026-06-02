import torch
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
batch_size = 1024


def train_one_alpha(alpha, phy_weight, label):
    """用给定的 alpha 训练一个小 PINN，返回最终 loss 和残差"""
    I_city = (18.0 + alpha) * bg

    mlp = SkyglowMLP([2, 64, 32, 1])
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
    loss_data = MSEData()
    loss_physics = MSEPhysics()

    for epoch in range(30):
        for coords_img, I_obs in all_images:
            idx = torch.randperm(len(coords_img))[:batch_size]
            batch_xy = coords_img[idx].clone().requires_grad_(True)
            batch_I = I_obs[idx]

            I_pred = mlp.forward(batch_xy).squeeze()

            ld = loss_data.forward(batch_I, I_pred)
            lp = loss_physics.forward(batch_I, I_pred, I_city[idx], alpha, phy_weight, batch_xy)
            (ld + lp).backward()
            optimizer.step()
            optimizer.zero_grad()

    coords_test, I_test = all_images[0]
    with torch.no_grad():
        I_pred_test = mlp.forward(coords_test).squeeze()
        residual = (I_test - I_pred_test).max().item()
        final_loss = ((I_test - I_pred_test) ** 2).mean().item()

    print(f"  α={alpha:5.1f}  λ={phy_weight:.4f}  "
          f"MSE={final_loss:.6f}  残差max={residual:.4f}")


# ---- 参数扫描 ----
print("PDE: Laplacian - alpha * I + I_city = 0\n")
for alpha, weight, tag in [(9.0, 0.005, "α=9"), (7.0, 0.005, "α=7"), (0.5, 0.01, "α=0.5")]:
    torch.manual_seed(42)
    train_one_alpha(alpha, weight, tag)
