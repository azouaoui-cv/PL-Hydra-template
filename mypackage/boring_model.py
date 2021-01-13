import torch.nn as nn
import torch
from .base import LitClassifier


class BoringModel(LitClassifier):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.linear = nn.Linear(in_features=32*32*3, out_features=10, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)
