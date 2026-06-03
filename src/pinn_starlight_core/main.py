# TODO: 自己从头写这个文件
#
# 作用: 训练 PINN 分离光污染 → 输出对比图
#
# 流程:
#   1. 数据加载
#     - 合成数据: FakeRaw(H, W, n_stars, seed) → coords, I_obs
#       或
#     - 真实数据: ImageLoader().load(path) → coords, I_obs
#
#   2. 模型
#     mlp = SkyglowMLP([2, 64, 32, 1])
#     alpha_param = nn.Parameter(tensor([5.0]))  ← α 可学习
#     optimizer = Adam(mlp.parameters() + [alpha_param], lr=0.001)
#
#   3. 训练循环
#     for step in range(N_steps):
#         idx = random N pixels
#         xy = coords[idx].requires_grad_(True)   ← 物理损失要对坐标求导
#         I_pred = mlp.forward(xy).squeeze()
#         loss = MSEData(I_obs[idx], I_pred)
#              + MSEPhysics(I_obs[idx], I_pred, I_city, alpha, weight, xy)
#         loss.backward(); optimizer.step(); optimizer.zero_grad()
#
#   4. 推理 + 输出
#     - 分块 (每 50000 像素) 跑全图 forward, 避免 OOM
#     - I_pred 和残差 (I_obs - I_pred).clamp(0,1) 分别保存
#     - 合成数据需要设 I_city = (18+α)*bg; 真实数据用 I_city=0.5 常数
#
# 注意:
#   - batch_xy 需要 .clone().requires_grad_(True), 否则 autograd 找不到坐标依赖
#   - 真实图片归一化到 [0,1] 后 I_city 也应在 [0,1] 量级
#   - 大图推理时用分块循环, 不要一次性算全图
