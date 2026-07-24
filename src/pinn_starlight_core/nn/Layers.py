import torch
import torch.nn as nn
import torch.nn.init as init


class SkyglowLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_l = nn.Parameter(torch.empty(in_dim, out_dim))
        init.xavier_uniform_(self.W_l)
        self.b_l = nn.Parameter(torch.zeros(out_dim))

    def forward(self, a_l):
        return a_l @ self.W_l + self.b_l
