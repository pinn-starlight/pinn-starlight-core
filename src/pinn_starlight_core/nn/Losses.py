import torch
from pinn_starlight_core.utils.Laplacian import laplacian

class MSEData:
    def __init__(self):
        self.I_obs = None
        self.I_pred = None

    def forward(self, I_obs, I_pred):
        self.I_obs = I_obs
        self.I_pred = I_pred
        return ((I_pred - I_obs) ** 2).mean()

    def backward(self):
        return 2 * (self.I_pred - self.I_obs) / self.I_obs.size(0)


class MSEPhysics:
    def __init__(self) -> None:
        self.I_obs = None
        self.I_pred = None
        self.I_city = None
        self.alpha = None
        self.weight = None
        self.coords = None

    def forward(self, I_obs, I_pred, I_city, alpha, weight, coords):
        self.I_obs = I_obs
        self.I_pred = I_pred
        self.I_city = I_city
        self.alpha = alpha
        self.weight = weight
        self.coords = coords

        # Helmholtz: ∇²I + αI = I_city  (α = 1/D² > 0)
        f = laplacian(I_pred, coords) + alpha * I_pred - I_city

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
        # TODO[3]: 大 FOV 场景的坐标缩放
        #   当前: coords 归一化到 [0,1] → autograd ∇² 直接可用
        #   大 FOV (手机 1x, 70-80°): 需要把像素坐标映射到实际天球角尺度
        #   否则 ∂²/∂x² 的量纲是 1/pixel² 而非 1/rad²
        #   α 也跟着变了: α = 1/D² 中 D 需用弧度而不是像素
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
        return weight * (f ** 2).mean()

    def backward(self):
        f = laplacian(self.I_pred, self.coords) + self.alpha * self.I_pred - self.I_city

        return 2 * self.weight * (laplacian(f, self.coords) + self.alpha * f) / self.I_pred.size(0)