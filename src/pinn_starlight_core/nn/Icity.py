# TODO: 自己从头写可学习 I_city
#
# Screened Poisson: ∇²I - αI + I_city = 0
# I_city 容纳所有无法被拉普拉斯+αI吸收的贡献
#
# 三种模式:
#   A. 标量: nn.Parameter(tensor([0.0])) — 全局常数，适均匀发光
#   B. 偏移: cx,cy → r=√((x-cx)²+(y-cy)²) → Garstang I_city
#       注意: cx,cy 梯度弱，需独立 lr
#   C. 网格: nn.Parameter(1,1,32,32) → F.grid_sample
#       无光源形状假设，需平滑正则
#
# 论文要点: PDE 不依赖小角度近似，只有 I_city 解析式依赖。
#   大 FOV 下放弃解析式 → 用可学习表示。
