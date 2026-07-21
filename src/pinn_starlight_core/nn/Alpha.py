import torch
from torch import nn


# 可学习Alpha（其实学不了多少）
class Alpha(nn.Module):
    def __init__(self, init=0.5, min=0.3, max=2.0):
        super().__init__()
        self.alpha_min = min
        self.alpha_max = max

        ratio = (init - min) / (max - min)
        ratio = torch.tensor(ratio, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)

        self.raw_alpha = nn.Parameter(torch.logit(ratio))

    def forward(self):
        length = self.alpha_max - self.alpha_min
        ratio = torch.sigmoid(self.raw_alpha)
        return self.alpha_min + length * ratio
