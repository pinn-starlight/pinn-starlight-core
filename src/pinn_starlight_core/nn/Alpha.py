# 可学习 alpha 说明
# alpha = 1 / D**2，其中 D 表示等效散射尺度。
# TODO: 通过学习正值 D（或 log_D）来得到 alpha，而不是直接学习裸 alpha。
# TODO: 为消融实验保留 alpha 与 I_city 的解耦设置，避免两者互相补偿。
