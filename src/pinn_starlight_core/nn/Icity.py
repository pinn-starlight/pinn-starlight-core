# TODO: 自己从头写 I_city — 粗卷积法
#
# 思路:
#   粗卷积求粗光污染梯度 → 梯度指向源点方向 → 源点自然定位
#   不需要显式参数 cx,cy, 也不需要 Garstang 解析式
#
# 参考:
#   Screened Poisson: ∇²I - αI + I_city = 0
#   I_city 容纳所有无法被拉普拉斯+αI吸收的贡献
