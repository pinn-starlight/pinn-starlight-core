# === 待写 ===
#
# TODO[1]: 亮度自适应权重 — 暗区松绑
#   return weight * (I_obs.detach() * f**2).mean()
#   I_obs 大→物理严约束, I_obs 小→只跟数据走
#
# TODO[2]: I_city 的可学习化 (Phase 2 用)
#   当前: 固定公式 I_city = (18+α)·bg，假设单一径向光源
#   目标: 低分辨率网格 (如 32×32) + 双线性插值
#     self.grid = nn.Parameter(torch.zeros(1,1,32,32))
#     I_city = F.grid_sample(grid, coords_2d).squeeze()
#   物理含义: 放弃"单点城市"假设 → 允许多方向非均匀光源
#   PDE 形式不变 (∇²I + αI = I_city)，只改 I_city 计算方式
#   注意: 可能需要对网格加平滑正则 (TV norm / Laplacian prior)
#   避免网格自身学到高频噪声
#
# TODO[3]: 大 FOV 场景 — 小角度近似的边界
#   项目的数学推导假设 r/D ≪ 1（FOV < 10°，天顶附近）。
#   实际照片大多 20-90°，画面内天顶角从 cos 主导跨越到 ψ⁴ 主导。
#   好消息: PDE 本身 (∇²I + αI = I_city) 不依赖小角度近似——
#   ∂²/∂x²+∂²/∂y² 对任何二维函数都成立，小角度只影响 I_city 的解析形式。
#   所以修复方案就是 TODO[2] 的可学习 I_city 网格——
#   让 I_city 从数据中自适配视角，不再依赖手写 cos+ψ⁴ 公式。
#   同时 α = 1/D² 在大 FOV 下 D 的含义从"散射层高度"变成"等效衰减半径"，
#   实验中可让 α 本身也可学习（已有 nn.Parameter 实现）。
#   论文写法: "PDE 对小角度近似免疫——近似只约束 I_city 的解析形式。
#   在大视场下我们放弃解析形式，改为可学习网格表示，PDE 骨架不变。"
#
# TODO[4]: 光穹顶 (light dome) 去除
#   方向性散射碰巧满足 PDE → 物理损失不排斥
#   短期: 裁天顶区域; 长期: 多虚拟光源 / 方向性 I_city
# 路线图:
#   Phase 1 (当前): 小 FOV, 固定 I_city → 出对比图
#   Phase 2 (6/8~): 大 FOV, 可学习 I_city 网格 → 手机 45° 测试
#   Phase 3 (6/16~): 消融: 固定 vs 可学习, 单源 vs 多源,
#                          Helmholtz vs SP (大 FOV)


import torch
from torch import optim
from tqdm.notebook import tqdm

import pinn_starlight_core.nn.Layers as Layers
import pinn_starlight_core.nn.Losses as Loss
import pinn_starlight_core.data.RAWLoader as RAWLoader
import pinn_starlight_core.data.FakeRAW as FakeRAW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

raw_loader = RAWLoader.RAWLoader()
raw_loader.from_array(FakeRAW.FakeRaw().get_fake_raw())
coords, values, W, H = raw_loader.get_raw_data(device= device)

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

optimizer = optim.Adam(params, lr=0.001)

ld = Loss.MSEData()
lp = Loss.MSEPhysics()


# === I_city: 三种模式任选 ===
# A.固定: I_city = α·bg (线性背景 ∇²bg=0)
# B.可学偏移: IcityLearnable(A=0.3, D=1.0), cx,cy 自动移向光穹
# C.可学网格: IcityGrid(32), 完全不限制光源形状
from pinn_starlight_core.nn.Icity import IcityFixed, IcityLearnable, IcityGrid

# --- 当前: 模式 A (固定) ---
alpha = 2.0
bg = 0.3 * (1.0 - (coords[:, 0] + coords[:, 1]) / 2.0).to(device)
ic = IcityFixed(alpha * bg)

# --- 模式 B (可学偏移, 用单独 lr) ---
# ic = IcityLearnable(A=0.3, D=1.0)
# cx_cy_params = ic.extra_params()
# optimizer = optim.Adam(params + cx_cy_params, lr=0.001)
# # 给 cx,cy 单独设 lr:
# optimizer = optim.Adam([
#     {'params': params, 'lr': 0.001},
#     {'params': cx_cy_params, 'lr': 0.05},   # 源点需要大 lr 才能移动
# ])

# --- 模式 C (可学网格) ---
# ic = IcityGrid(32)
# grid_params = ic.extra_params()
# optimizer = optim.Adam(params + grid_params, lr=0.001)

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

    # 模式A: I_city 已预计算，直接取 idx；模式B/C: 当场算 batch_xy
    if isinstance(ic, IcityFixed):
        I_city_b = ic.I_city[idx]
    else:
        I_city_b = ic(batch_xy)

    data_loss = ld.forward(batch_I, I_pred)
    phys_loss = lp.forward(batch_I, I_pred, I_city_b, alpha, phy_weight, batch_xy)
    loss = data_loss + phys_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}, data={data_loss.item():.6f}, phys={phys_loss.item():.6f}")