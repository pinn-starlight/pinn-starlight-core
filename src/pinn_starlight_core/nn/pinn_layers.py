import torch.nn as nn
import torch.nn.init as init


def skyglow_linear(in_dim, out_dim):
    layer = nn.Linear(in_dim, out_dim)
    init.xavier_uniform_(layer.weight)
    init.zeros_(layer.bias)
    return layer


class SkyglowMLP(nn.Sequential):
    def __init__(self):
        """默认是2 -> 128 -> 128 -> 1"""
        super().__init__()
        self.extend(
            [
                nn.Linear(2, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            ]
        )
        self.apply(self._init_linear)

    @staticmethod
    def _init_linear(module):
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
