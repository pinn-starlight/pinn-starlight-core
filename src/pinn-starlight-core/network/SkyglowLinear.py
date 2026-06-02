import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SkyglowLinear(nn.Module):
    def __init__(self):
        super(SkyglowLinear, self).__init__()
        self.vector = None

    def forward(self, input):
        