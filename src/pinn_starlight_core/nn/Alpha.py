import torch
from torch import nn


# 可学习Alpha（其实学不了多少）
# TODO:编写正式验证alpha稳定在0.5附近
class Alpha(nn.Module):
    def __init__(self, init=1, alpha_min=0.5, alpha_max=1.5):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.init = init

        ratio = (init - alpha_min) / (alpha_max - alpha_min)
        ratio = torch.tensor(ratio, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)

        self.raw_alpha = nn.Parameter(torch.logit(ratio))

    def forward(self):
        length = self.alpha_max - self.alpha_min
        ratio = torch.sigmoid(self.raw_alpha)
        return self.alpha_min + length * ratio

    def get_str(self):
        return f"[min:{self.alpha_min},max:{self.alpha_max},init:{self.init}]"