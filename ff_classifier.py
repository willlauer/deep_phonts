import torch
from torch import nn
import torch.nn.functional as F
from utils import Flatten


"""
class flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(x.shape[0],-1)
"""

class FF_Classifier(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        # nn.Conv2d (in_channels, out_channels, kerclnel_size)

        self.net = nn.Sequential(
            nn.Conv2d(1, 28, 5, stride=1),
            nn.ELU(),
            Flatten(),
            nn.Linear(28*24*24, n_classes)
        )

    def forward(self, X):

        return self.net(X)
