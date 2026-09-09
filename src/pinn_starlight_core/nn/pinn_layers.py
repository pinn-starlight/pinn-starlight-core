from itertools import pairwise

from torch import nn
from torch.nn import init


def skyglow_linear(in_dim, out_dim):
    layer = nn.Linear(in_dim, out_dim)
    init.xavier_uniform_(layer.weight)
    init.zeros_(layer.bias)
    return layer


class SkyglowMLP(nn.Sequential):
    def __init__(self, hidden_dims=(128, 128)):
        """坐标 MLP；默认结构为 2 -> 128 -> 128 -> 1。"""
        super().__init__()
        hidden_dims = tuple(int(width) for width in hidden_dims)
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("hidden_dims 必须包含正整数")

        dimensions = (2, *hidden_dims, 1)
        modules = []
        for index, (in_dim, out_dim) in enumerate(pairwise(dimensions)):
            modules.append(nn.Linear(in_dim, out_dim))
            if index < len(dimensions) - 2:
                modules.append(nn.Tanh())
        self.extend(modules)
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
