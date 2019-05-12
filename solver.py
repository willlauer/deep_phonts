import torch
from torch import nn
import torch.nn.functional as F
from models.SmallVGG import SmallVGG
from utils import visualize_samples
from tqdm import tqdm
import imageio
from torch.autograd import Variable
from torch.distributions.normal import Normal

class Solver:


    def __init__(self, model, train_loader, val_loader):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.normal = Normal(0, 1)

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




    def transfer(self, num_iters):

        """
        Assuming we have a pre-trained model from the result of train(), perform style transfer
        from some image a to image b

        :return: None
        """

        content_img = imageio.imread("data/consolas/a2.png")
        style_img = imageio.imread("data/times_new_roman/a2.png")

        # Compute the content and style of our content and style images
        _, content_target, _ = self.model.forward(content_img)
        _, _, style_target = self.model.forward(style_img)


        # sample from normal distribution, wrap in a variable, and let requires_grad=True
        noise = Variable(self.normal.sample(content_target.shape), requires_grad=True)

        self.model.mode = 'transfer'    # we now want to compute content and style



        optimizer = torch.optim.SGD([noise]) # optimize the noise


        for i in range(num_iters):


            # send our noise forward through the model, computing its style and content
            optimizer.zero_grad()
            _, content, style = self.model.forward(noise)


            # compute the loss as the sum of the mean-squared-error loss for the content and style
            # TODO: unsure if we should sum across all style representations or just use one
            content_loss = F.mse_loss(content, content_target)
            style_loss = F.mse_loss(style[1], style_target[1])
            loss = content_loss + style_loss

            # compute gradient with respect to the input and take a step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




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
