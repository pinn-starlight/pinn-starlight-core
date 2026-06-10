"""提升极限星等 — 暗星在残差中的可见性提升 (夏 & Claude 合写)"""
import torch, numpy as np
from torch import optim

from pinn_starlight_core.nn.Layers import SkyglowLinear, SkyglowActivation
from pinn_starlight_core.nn.Losses import MSEData, MSEPhysics
from pinn_starlight_core.data.RAWLoader import RAWLoader
from pinn_starlight_core.data.FakeRAW import FakeRaw

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 数据：cos 背景 + 已知亮度的星 ---
fake = FakeRaw(W=256, H=256, n_stars=15, bg_amplitude=0.30, star_brightness=0.7, seed=42)
loader = RAWLoader()
loader.from_array(fake.get_fake_raw())
coords, values, W, H = loader.get_raw_data(device=device)
N = coords.shape[0]

# 手动撒一批已知坐标的暗星 (0.03 ~ 0.10)，记录其位置用于后续验证
rng_np = np.random.default_rng(99)
n_faint = 30
faint_centers = rng_np.uniform(0.15, 0.85, (n_faint, 2))  # 避免贴边
faint_brightness = rng_np.uniform(0.02, 0.12, n_faint)      # 极限暗星
faint_map = torch.zeros(N, device=device)
for i in range(n_faint):
    cx, cy = float(faint_centers[i, 0]), float(faint_centers[i, 1])
    dist2 = (coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2
    faint_map += float(faint_brightness[i]) * torch.exp(-dist2 / 0.0003)
I_obs = values + faint_map

# --- PINN 训练 ---
layers = [
    SkyglowLinear(2, 256).to(device), SkyglowActivation(),
    SkyglowLinear(256, 64).to(device),  SkyglowActivation(),
    SkyglowLinear(64, 1).to(device),
]
params = [p for l in layers if isinstance(l, SkyglowLinear) for p in l.parameters()]
opt = optim.Adam(params, lr=0.001)

bg = 0.30 * torch.cos(3.0 * coords[:, 0]) * torch.cos(3.0 * coords[:, 1])
I_city = (9.0 - 18.0) * bg
batch_size = max(1024, min(8192, N // 200))
ld = MSEData(); lp = MSEPhysics()

for step in range(2000):
    idx = torch.randint(0, N, (batch_size,), device=device)
    xy = coords[idx].clone().requires_grad_(True)
    Io = I_obs[idx]

    a = xy
    for l in layers: a = l.forward(a)
    Ip = a.squeeze()

    loss = ld.forward(Io, Ip) + lp.forward(Io, Ip, I_city[idx], 9.0, 0.01, xy)
    opt.zero_grad(); loss.backward(); opt.step()

# --- 推理 ---
with torch.no_grad():
    I_pred = torch.empty(N, device=device)
    for s in range(0, N, 50000):
        e = min(s + 50000, N)
        a = coords[s:e]
        for l in layers: a = l.forward(a)
        I_pred[s:e] = a.squeeze()

residual = (I_obs - I_pred).clamp(0, 1)

# --- 评估：在每个暗星中心测 SNR ---
print(f"\n=== 极限星等提升评估 ===")
print(f"{'星点':>5s} {'真实亮度':>8s}  {'原图SNR':>8s}  {'残差SNR':>8s}  {'提升':>6s}")

improved = 0
detected_in_obs = 0
detected_in_res = 0
SNR_THRESHOLD = 2.0

for i in range(n_faint):
    cx, cy = faint_centers[i, 0], faint_centers[i, 1]
    # 在星点中心取 3×3 区域
    r = 0.01
    mask = ((coords[:, 0] - cx) ** 2 + (coords[:, 1] - cy) ** 2) < (r ** 2)
    if mask.sum() < 4:
        continue
    bg_obs = (I_obs[~mask].mean() + I_obs[mask].mean()) / 2  # 近似背景
    bg_res = (residual[~mask].mean() + residual[mask].mean()) / 2
    snr_obs = (I_obs[mask].max() - I_obs[~mask].mean()) / (I_obs[~mask].std() + 1e-8)
    snr_res = (residual[mask].max() - residual[~mask].mean()) / (residual[~mask].std() + 1e-8)

    if snr_obs > SNR_THRESHOLD: detected_in_obs += 1
    if snr_res > SNR_THRESHOLD: detected_in_res += 1
    if snr_res > snr_obs + 0.5: improved += 1

    if i < 10:
        print(f"  #{i+1:2d}  {faint_brightness[i]:8.4f}  {snr_obs:8.2f}  {snr_res:8.2f}  {'+' if snr_res>snr_obs else '-'}{abs(snr_res-snr_obs):5.2f}")

print(f"\n原图可检测 ({SNR_THRESHOLD}σ): {detected_in_obs}/{n_faint}")
print(f"残差可检测 ({SNR_THRESHOLD}σ): {detected_in_res}/{n_faint}")
print(f"SNR 提升:       {improved}/{n_faint} 颗星")

# --- 出图：三栏 + 暗星标记 ---
import matplotlib.pyplot as plt
obs_2d = I_obs.reshape(W, H).cpu().numpy()
pred_2d = I_pred.reshape(W, H).cpu().numpy()
res_2d = residual.reshape(W, H).cpu().numpy()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
titles = ['Observed (bg+stars)', 'Predicted (light pollution)', 'Residual (stars uncovered)']
for a, arr, t in zip(ax, [obs_2d, pred_2d, res_2d], titles):
    a.imshow(arr, cmap='gray', vmin=0, vmax=0.4)
    # 标出前 10 颗暗星位置
    for i in range(min(n_faint, 10)):
        cx, cy = faint_centers[i]
        a.plot(cx * W, cy * H, 'r+', markersize=8, markeredgewidth=1.5)
    a.set_title(t)
    a.axis('off')
fig.tight_layout()
fig.savefig('limiting_magnitude_demo.png', dpi=150)
print('Saved: limiting_magnitude_demo.png')
