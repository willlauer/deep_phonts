import torch
from torch import nn 
from SmallVGG import SmallVGG
from utils import load_emnist, visualize_samples, visualize_weights
from solver import Solver


def main():

    # Define the parameters to use for this model
    split = "bymerge"
    num_epochs = 1


    # the number of classes in each nist split
    classes = {
        "byclass":  62, 
        "bymerge":  47, 
        "balanced": 47, 
        "letters":  26, 
        "digits":   10, 
        "mnist":    10
    }

    model = SmallVGG(classes[split], 1, 3, 3, 1) 
    train_loader, val_loader = load_emnist(split, False, batch_size_train=5000)
    visualize_samples(val_loader)
    solver = Solver(model, train_loader, val_loader)
    
    solver.train(num_epochs)

    visualize_weights(model)



if __name__=='__main__':
    main()