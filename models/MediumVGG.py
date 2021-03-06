import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class MediumVGG(nn.Module):

    def __init__(self, num_classes, in_channel, c1, c2, c3, c4, c5):
        super().__init__()

        # Initialize our layers. Following the VGG architecture, we use 3x3 convolutional layers
        # with stride 1 and pad 1, as well as 2x2 pooling layers

        self.conv1 = nn.Conv2d(in_channel, c1, (3, 3), stride=1, bias=True, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, (3, 3), stride=1, bias=True, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, (3, 3), stride=1, bias=True, padding=1)

        self.max_pool = nn.MaxPool2d((2, 2))

        self.conv4 = nn.Conv2d(c3, c4, (3, 3), stride=1, bias=True, padding=1)
        self.conv5 = nn.Conv2d(c4, c5, (3, 3), stride=1, bias=True, padding=1)

        # Output size is reduced after maxpooling layers
        self.max_pool_layers = 0 # Modify this appropriately
        self.output_size = 28 // (2 ** self.max_pool_layers)
        self.fc = nn.Linear(c5 * self.output_size * self.output_size, num_classes, bias=True)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.conv5.weight)


        nn.init.kaiming_normal_(self.fc.weight)

        # mode should be set to either 'classify' or 'transfer'
        self.mode = 'classify'

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def gram(self, r):
        """
        Params:
            r (tensor): (in_channels x out_channels (i.e. num filters) x width x height)

        Compute the gram matrix using method from
        https://github.com/aleju/papers/blob/master/neural-nets/A_Neural_Algorithm_for_Artistic_Style.md

        1. Take the activations of a layer (passed in as r)
        2. Convert each filter's activations to a 1-d vector
        3. Pick all pairs of filters. Calculate the scalar product of both filter's vectors
        4. Add the scalar product result as an entry to a matrix of size (#filters, #filters)
        5. Repeat for every pair to get the Gram Matrix
        """

        a, b, c, d = r.shape
        r = r.view(a * b, c * d)
        return r.mm(r.t()) / (a * b * c * d)

    def contrast_loss(self, x):
        """
        :param x: input image
        :return: a metric quantifying how much contrast there is between the pixels. The goal is
        to promote well-defined edges
        """
        pass




    def forward(self, x):

        # {conv2d (pool - maybe?)} x L; fully_connected layer for classification

        r1 = F.relu(self.conv1(x))
        r2 = F.relu(self.conv2(r1))
        r3 = F.relu(self.conv3(r2))
        r4 = F.relu(self.conv4(r3))
        r5 = F.relu(self.conv5(r4))

        scores = self.fc(self.flatten(r5))

        if self.mode == 'transfer':

            # Compute the gram matrices output from each of the convolutional layers (post-activation)
            # Only maybe do this for the early layers
            style = [self.gram(r1), self.gram(r2), self.gram(r3), self.gram(r4), self.gram(r5)]
            content = r5

            #contrast = self.contrast_loss(x)

        else:

            style = None
            content = None
            #contrast = None

        return scores, content, style #, contrast