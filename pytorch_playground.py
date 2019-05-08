import torch 
import torchvision
import numpy as np
from tqdm import tqdm

import torch.nn.functional as F



import matplotlib.pyplot as plt

from ff_classifier import FF_Classifier


class PytorchPlayground:

    def __init__(self):

        self.needs_download = False
        self.batch_size_train = 1000
        self.batch_size_test = 1000


    def load_mnist(self):
        
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=True, download=self.needs_download,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
        batch_size=self.batch_size_train, shuffle=True)

        self.val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data/', train=False, download=self.needs_download,
                                        transform=torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])),
        batch_size=self.batch_size_test, shuffle=True)




    def visualize_samples(self):

        examples = enumerate(self.val_loader)
        batch_idx, (example_data, example_targets) = next(examples)


        fig = plt.figure()
        for i in range(6):
            plt.subplot(2,3,i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            plt.title('Ground truth: {}'.format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
            #print(example_data[i][0].shape)
        #plt.show()


    def check_accuracy(self, model):

        num_correct = 0
        num_samples = 0

        model.eval()
        with torch.no_grad():
            for x, y in self.val_loader:
                scores = model.forward(x)
                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, acc))



    def run(self):

        print_every = 10

        self.load_mnist()
        self.visualize_samples()

        model = FF_Classifier(10)

        optimizer = torch.optim.Adam(model.parameters())

        for _ in tqdm(range(1)):

            ct = 0
            for x,y in self.train_loader:

                optimizer.zero_grad()
                scores = model.forward(x)
                loss = F.cross_entropy(scores, y)

                optimizer.zero_grad()

                loss.backward() # compute gradients
                optimizer.step()

                ct += 1
                print(ct)

                if ct % print_every == 0:
                    print('Iteration {}, loss {}'.format(ct, loss.item()))
                    self.check_accuracy(model)



def main():

    ptp = PytorchPlayground()
    ptp.run()
    

if __name__=='__main__':
    main()