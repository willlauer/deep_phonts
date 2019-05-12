import torch
from torch import nn
import torch.nn.functional as F
from SmallVGG import SmallVGG
from utils import visualize_samples
from tqdm import tqdm

class Solver:


    def __init__(self, model, train_loader, val_loader):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader


    def check_accuracy(self):

        num_correct = 0
        num_samples = 0

        self.model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                scores = self.model.forward(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, acc))




    def transfer(self):

        """
        Assuming we have a pre-trained model from the result of train(), perform style transfer

        :return: None
        """
        pass







    def train(self, num_epochs):


        print_every = 10

        optimizer = torch.optim.Adam(self.model.parameters())


        # stop at ct == debug_stop_count if we're tryna end early
        debug_stop_count = 10


        for ep in range(num_epochs):

            ct = 0
            print('Epoch {}'.format(ep))
            for x,y in self.train_loader:

                if ct == debug_stop_count:
                    break

                optimizer.zero_grad()
                scores, _, _ = self.model.forward(x)
                loss = F.cross_entropy(scores, y)

                optimizer.zero_grad()

                loss.backward() # compute gradients
                optimizer.step()

                ct += 1

                if ct % print_every == 0:
                    print('Iteration {}, loss {}'.format(ct, loss.item()))
                    self.check_accuracy()
