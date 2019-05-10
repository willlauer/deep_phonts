import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class SmallVGG(nn.Module):

    def __init__(self, num_classes, in_channel, c1, c2, c3):
        super().__init__()

        # Initialize our layers. Following the VGG architecture, we use 3x3 convolutional layers
        # with stride 1 and pad 1, as well as 2x2 pooling layers

        self.conv1 = nn.Conv2d(in_channel, c1, (3,3), stride=1, bias=True, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, (3,3), stride=1, bias=True, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, (3,3), stride=1, bias=True, padding=1)

        self.fc = nn.Linear(c1*28*28, num_classes, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.fc.weight)


    def flatten(self, x):
        return x.view(x.shape[0],-1)

    def forward(self, x):

        # {conv2d (pool - maybe?)} x L; fully_connected layer for classification
        
        r1 = F.relu(self.conv1(x))
        #r2 = F.relu(self.conv2(r1))
        #r3 = F.relu(self.conv3(r2))
        scores = self.fc(self.flatten(r1))

        return scores