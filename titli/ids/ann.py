import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, num_features: int, **kwargs):
        super().__init__(**kwargs)
        self.num_features = num_features

    def forward(self, x):
        raise NotImplementedError

    def train(self, data):
        raise NotImplementedError

    def infer(self, data):
        raise NotImplementedError

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
