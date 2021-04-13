import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x -= y.clone().detach()
        x = torch.pow(x, 2)
        while x.dim() > 1:
            x = torch.sum(x, dim=-1)
        return x
