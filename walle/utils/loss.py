import torch
import torch.nn as nn

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, z):
        squared_difference = (x - z) ** 2
        mean = torch.mean(squared_difference)
        rmse = torch.sqrt(mean)
        return rmse
