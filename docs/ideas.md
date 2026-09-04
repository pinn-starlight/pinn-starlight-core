# Ideas

> 本文件只记录可能的后续方向，不代表当前计划、论文贡献或已完成功能。
>
> 当前正式工作以[论文实验计划](./论文准备资料/03_实验/论文实验计划.md)为准。新想法先记在这里，不在 E0-E4 完成前临时加入主实验。

## 1. 类堆栈多张处理

### 想法

当同一场景有多张连续曝光时，不必只对每张图独立运行 PINN。可以利用多帧之间共享的星空与光污染结构，联合估计更稳定的背景。

可能的形式：

1. 先用专业软件完成星点对齐，再对多张图的 PINN 背景估计取均值或中位数；
2. 多张图共享同一个背景网络或 $I_{city}$ 参数，每张图保留独立噪声与曝光参数；
3. 使用多帧一致性损失，使对齐后的背景估计在不同帧之间保持一致；
4. 将传统堆栈结果作为低噪声参考，而不是与单图 PINN 直接比较。

### 需要注意

- 多帧方法使用了额外输入，不能与单图方法直接作公平排名；
- 相机位移、地球自转、旋转和镜头畸变需要先处理；
- 堆栈主要降低随机噪声，不会自动消除稳定存在的光污染；
- 当前论文研究的是单幅图像背景估计，本想法作为后续扩展。

## 2. RGB-aware 背景估计

### 想法

从灰度扩展到 RGB，利用光污染与星点、银河或星云之间的颜色差异，减少仅依赖空间平滑性造成的误判。

可能的实现：

- 数据损失比较三通道观测与预测；
- PDE 先只约束亮度分量，再评估是否需要逐通道约束；
- 输出 RGB 背景，并保留线性颜色空间；
- 对比灰度与 RGB 的背景误差和星点保留情况。

风险：模型与实验复杂度会明显增加，而且真实图仍缺少背景真值。

## 3. 天空区域 Mask

### 想法

对建筑、山体、树木和其他地景建立 mask，使数据损失和物理损失只在天空区域计算。

可以先从人工 mask 开始，再考虑自动天空分割。该方向适合解决真实图中的场景干扰，但不应在缺少可靠 mask 时仓促加入正式实验。

## 4. 多光源或更灵活的 $I_{city}$

当前单个椭圆源项难以表达多个城市方向、复杂天际线和不规则光晕。后续可以研究：

- 多个源项的加和；
- 自动选择源项数量；
- 更灵活但仍受约束的空间包络；
- 将参数数量与拟合收益一起报告，避免只靠增加自由度提升结果。

## 5. 边界条件

当前方法没有显式边界损失。后续可比较：

- 无边界条件；
- Neumann 型边界约束；
- 边缘平滑或低梯度正则；
- 反射 padding 或扩大坐标区域。

只有在数学定义、代码和消融结果一致后，才能写成方法的一部分。

## 6. 预训练与跨图初始化

PINN 当前针对每张图独立优化。可以研究：

- 使用上一张相似图的参数初始化；
- 先进行数据项预训练，再加入物理项；
- 在合成数据上学习初始化参数；
- 比较从零训练、预训练和邻近图初始化的收敛速度。

该方向的目标是减少逐图优化时间，不改变当前单图背景估计任务。

## 7. 更稳健的数据项

当前数据项使用 MSE，可能促使背景网络吸收高亮星点。后续可比较：

- MAE；
- Huber loss；
- 对高亮像素降权的稳健损失；
- 基于星点 mask 的加权 MSE。

这类修改会改变当前方法定义，必须通过同一验证集和消融实验决定，不能只凭单张图观感选择。

## 8. 不确定性估计

输出背景预测的同时，估计每个区域的不确定性，提示银河、薄云、暗角等不可辨识区域。可能方案包括多随机种子方差、模型集成或概率输出。

该方向更适合解释模型边界，不作为当前论文的必要功能。

## 9. 天文指标扩展

如果后续具备可靠的配准、点扩散函数和检测阈值控制，可以加入：

- Gaia 星表匹配；
- 星点测光误差；
- 检测完整度与虚警率；
- 极限星等变化。

在这些条件未满足前，不使用“恢复不可见星光”或“提高极限星等”等表述。

## 10. 工程与复现

- checkpoint 保存与恢复；
- 统一配置文件和命令行入口；
- 自动保存运行环境、Git commit、指标和浮点输出；
- 训练中断恢复；
- 更低显存的坐标采样与分块预测。

这些内容有助于复现，但只有真正实现后才能写入论文实验设置。

## 采用新想法前的判断标准

1. 它是否直接回答当前论文的研究问题？
2. 是否有时间实现对应 baseline、消融和评价？
3. 是否会改变已经锁定的数据或测试协议？
4. 它带来的收益是否值得增加的解释成本？

只要会影响 E0-E4 的按时完成，就先留在本文件，不进入当前版本。
### PINN boundary conditions (follow-up)

The current PINN has no explicit boundary loss. A future implementation can sample points on the four image edges and use automatic differentiation to penalize the normal derivative.

- Zero Neumann: `dB/dx = 0` on the left and right edges, and `dB/dy = 0` on the top and bottom edges.
- Periodic: match the background value and normal derivative across opposite edges.
- Add `boundary_weight * boundary_loss` to the total loss.

This requires a new loss function in `src/pinn_starlight_core/nn/pinn_loss.py` and boundary-point sampling plus loss integration in `experiments/common/baselines/coordinate_pinn.py`. The FFT baseline currently assumes periodic boundaries; switching it to DCT would correspond to zero-Neumann boundaries. Keep this as a future experiment and do not mix results produced under different boundary assumptions.
