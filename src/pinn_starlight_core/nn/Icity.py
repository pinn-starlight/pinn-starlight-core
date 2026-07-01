# TODO: 自己从头写 I_city — 粗卷积法
#
# 思路:
#   粗卷积求粗光污染梯度 → 梯度指向源点方向 → 源点自然定位
#   不需要显式参数 cx,cy, 也不需要 Garstang 解析式
#
# 当前问题: 星云和光污染在灰度上无法区分 (都亮+大尺度)
#
# 解法 A (分通道):
#   光污染偏橙/黄 (钠灯 589nm), 星云偏红 (Hα 656nm) 或蓝绿 (OIII 500nm)
#   R/G/B 分别算 I_city → 星云信号在某个通道明显弱
#   通道间差异 = 区分星云 vs 光污染的钥匙
#   前提: RAWLoader 已支持 RGB, get_raw_data 返三通道
#
# 解法 B (物理矫正):
#   粗卷积 I_city 作为初值, LearnableIcity 在 PINN 训练中自调整
#   PDE Screened Poisson 约束会反压误判: 星云不符合源项分布
#   类似: 先用模糊图当"先验", 再让物理方程做"纠错"
#
# 参考:
#   Screened Poisson: ∇²I - αI + I_city = 0
#   I_city 容纳所有无法被拉普拉斯+αI吸收的贡献
from torch import nn


class LearnableIcity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
