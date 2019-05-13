import torch
from torch import nn 
from models.SmallVGG import SmallVGG
from utils import load_emnist, visualize_samples, visualize_weights
from solver import Solver
import torch
import sys
from hyper_params import params


def main():

    # Define the parameters to use for this model
    split = "balanced"

    MODEL_NAME = 'smallvgg.pt'
    SAVE_PATH = './saved_models/'


    # the number of classes in each nist split
    classes = {
        "byclass":  62, 
        "bymerge":  47, 
        "balanced": 47, 
        "letters":  26, 
        "digits":   10, 
        "mnist":    10
    }

    train_loader, val_loader = load_emnist(split, False,
            batch_size_train=params["batch_size_train"], batch_size_test=params["batch_size_val"])

    if sys.argv[1] == '-l': # load model
        model = SmallVGG(classes[split], 1, 3, 3, 1)
        model.load_state_dict(torch.load(SAVE_PATH+MODEL_NAME))
        solver = Solver(model, train_loader, val_loader)


    elif sys.argv[1] == '-t': # train model
        model = SmallVGG(classes[split], 1, 3, 3, 1)
        #visualize_samples(val_loader)
        solver = Solver(model, train_loader, val_loader)
        solver.train(params["train_num_epochs"])
        torch.save(model.state_dict(), SAVE_PATH+MODEL_NAME)


    # run the transfer on out output for 100 iterations
    solver.transfer(params["transfer_num_iters"])


if __name__=='__main__':
    main()