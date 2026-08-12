# 论文实验脚本

每个实验都放在自己的文件夹里，入口统一叫 `run.py`：

```text
scripts/
  e0/run.py       # 少量步骤试试代码能不能跑
  e1/run.py       # 选择并锁定 PINN 配置
  e2/run.py       # FFT、U-Net-small、PINN 主对比
  e3/run.py       # PDE 与源点中心消融
  e4/run.py       # 固定真实图对比
  baselines/      # FFT 和 U-Net-small 公共实现
  common/         # 保存、随机种子、指标等公共工具
  data/           # 合成数据生成
  legacy/         # 历史原型，不作为正式实验入口
```

开工顺序建议：`data -> common -> baselines -> e0 -> e1 -> e2 -> e3 -> e4`。

E0 说白了就是每种方法少跑几步，能打印 loss、输出背景图和残差图，确认代码没断。它不比较效果，也不进入论文结果。

统一命名：

```text
clean_true       = 合成干净星图真值
background_true  = 合成背景真值
observed         = 污染观测
background_pred  = 方法估计的背景
residual_pred    = observed - background_pred
```

当前文件只是带注释的开工骨架。遇到 `NotImplementedError`，按该函数上方的 `TODO` 补实现；旧代码只去 `legacy/` 查，不直接作为正式实验运行。
