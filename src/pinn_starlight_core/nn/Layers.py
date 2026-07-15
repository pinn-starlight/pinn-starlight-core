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

class SkyglowActivation(nn.Module):
    def __init__(self) -> None :
        super().__init__()

    def forward(self, Z_l):
        return tanh(Z_l)