import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import tanh


class SkyglowLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr = 0.001):
        super(SkyglowLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr

        self.a_l = None
        self.W_l = nn.Parameter(torch.empty(in_dim, out_dim))
        init.xavier_uniform_(self.W_l)
        self.b_l = nn.Parameter(torch.zeros(out_dim))

    def forward(self, a_l):
        self.a_l = a_l
        return a_l @ self.W_l + self.b_l


class SkyglowActivation:
    def __init__(self) -> None :
        self.Z_l = None

    def forward(self, Z_l):
        self.Z_l = Z_l
        return tanh(Z_l)

    def backward(self, dL_da):
        return dL_da * (1 - tanh(self.Z_l) ** 2)


class SkyglowMLP:
    def __init__(self, layer_sizes):
        self.layers = []       # Linear 和 Activation 交替存放
        for i in range(len(layer_sizes) - 1):
            self.layers.append(SkyglowLinear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:           # 输出层不用接激活
                self.layers.append(SkyglowActivation())

    def forward(self, x):
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def parameters(self):
        for layer in self.layers:
            if isinstance(layer, SkyglowLinear):
                yield from layer.parameters()