import torch
from torch import nn 
from models.SmallVGG import SmallVGG
from utils import load_emnist, visualize_samples, visualize_weights
from solver import Solver
import torch




def main():

    # Define the parameters to use for this model
    split = "bymerge"
    num_epochs = 1

    SAVE_MODEL = True
    MODEL_NAME = 'test_model.pt'
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

    model = SmallVGG(classes[split], 1, 3, 3, 1) 
    train_loader, val_loader = load_emnist(split, True, batch_size_train=5000)
    visualize_samples(val_loader)
    solver = Solver(model, train_loader, val_loader)
    
    solver.train(num_epochs)

    if SAVE_MODEL:
        torch.save(model.state_dict(), SAVE_PATH+MODEL_NAME)


if __name__=='__main__':
    main()