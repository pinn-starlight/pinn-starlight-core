import torch
from torch import nn


class Alpha(nn.Module):
    def __init__(self, alpha_init=0.5, alpha_min=0.3, alpha_max=2.0):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        p = (alpha_init - alpha_min) / (alpha_max - alpha_min)
        p = torch.tensor(p, dtype=torch.float32).clamp(1e-6, 1 - 1e-6)

        self.raw_alpha = nn.Parameter(torch.logit(p))

    def forward(self):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_alpha)
