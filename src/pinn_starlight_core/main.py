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
#
# === 待证明 / 待整理 (2026-06) ===
#
# TODO[5]: 手动证明 Helmholtz 解在小 FOV 的非振荡性
#   核心: 验证 Bessel J₀(√α r) 在 r/D ≪ 1 时展成 1 − z²/4 + ...
#   确认第一个零点 z≈2.4 在当前像素/角度域内不可达
#   需: 手推 Helmholtz 齐次解的级数展开, 代入实际 D 和 FOV
#   证明完才能写进论文 — 目前是实验观察, 缺数学闭环
#
# 实验观察 (待证明后提升为结论):
#   cos(r/D) → cos'' = -α·cos → ∇²cos + α·cos = 0 (Helmholtz)
#   实验中 Helmholtz 优于 SP, 猜测原因 = 上述非振荡性
#
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


# TODO alpha调小
alpha = 9.0
bg = 0.3 * torch.cos(3.0 * coords[:, 0]) * torch.cos(3.0 * coords[:, 1]).to(device)
I_city = (alpha - 18.0) * bg
phy_weight = 0.01

# TODO 参数待调
for step in tqdm(range((coords.shape[0] / 240).int())):
    # TODO 参数待调
    idx = torch.randint(0, coords.shape[0], (coords.shape[0] / 320,)).to(device)
    batch_xy = coords[idx].to(device).clone().requires_grad_(True)
    batch_I = values[idx].to(device)

    a = batch_xy
    for layer in layers:
        a = layer.forward(a)
    I_pred = a.squeeze().to(device)

    data_loss = ld.forward(batch_I, I_pred).to(device)
    phys_loss = lp.forward(batch_I, I_pred, I_city[idx], alpha, phy_weight, batch_xy).to(device)
    loss = data_loss + phys_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {(step // 500) + 1}, data={data_loss.item() * 100:.6f}%, phys={phys_loss.item() * 100:.6f}%, total={loss.item() * 100:.6f}%")