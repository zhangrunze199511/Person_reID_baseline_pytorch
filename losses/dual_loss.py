import torch
import torch.nn as nn
import numpy as np

from .hard_mine_triplet_loss import TripletLoss


class DualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xentl = nn.CrossEntropyLoss()
        self.htril = TripletLoss()

    def forward(self,input, target):
        #return self.xentl(input, target) + self.htril(input, target)
        return self.htril(input, target)
