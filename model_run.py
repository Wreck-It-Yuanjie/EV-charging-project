import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models
from torchsummary import summary

import model_structure
import model_train

def model_fit(train_dl, val_dl, model, device, num_epochs, loss_function, optimizer):
    '''
    For num_epochs, fit the model parameters to the training data, evaluation on validation data
    For each epoch, calculate and return the training and validation loss
    '''
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        ## step 1: training
        train_loss = model_train.train_one_epoch(model, train_dl, loss_function, device, optimizer)
        train_losses.append(train_loss)

        ## step 2: validation
        val_loss = model_train.val_one_epoch(model, val_dl, loss_function, device, optimizer)
        val_losses.append(val_loss)

        print(f"Epoch{epoch} | train loss: {train_loss:.3f} | val loss: {val_loss:.3f}")
    return train_losses, val_losses

def model_run(train_dl, val_dl, test_dl, model, device, learning_rate = 0.001, num_epochs = 500, loss_function = None, optimizer = None):
    ## Define loss function
    if loss_function is None:
        loss_function = nn.MSELoss()
    
    ## Define optimization function
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    ## run the model fit function
    train_losses, val_losses = model_fit(train_dl, val_dl, model, device, num_epochs, loss_function, optimizer)

    ## test the model
    test_losses = test_model(model, test_dl, loss_function, optimizer)

    return train_losses, val_losses, test_losses

def model_bundle(train_dataloader, val_dataloader, test_dataloader, device, seq_len = 3, model_type = 4, num_epoch = 1000, verbose = True):
    '''
    Model bundle function that takes "Linear", "MLP", "CNN", "LSTM" as input
    "Linear": 1
    "MLP": 2
    "CNN": 3
    "LSTM": 4
    If "verbose", output the model summary
    '''
    if(model_type == 1):
        model = model_structure.linear_Net(seq_len, 1)
    elif(model_type == 2):
        model = model_structure.mlp_Net(seq_len, hidden_size = 32, output_size = 1)
    elif(model_type == 3):
        model = model_structure.cnn_Net(seq_len, kernel_size = 3, output_size = 1)
    elif(model_type == 4):
        model = model_structure.lstm_Net(seq_len, hidden_size = 32, output_size = 1)

    if(verbose and model_type < 4):
        print("+--------------------+")
        print("|Print the linear model summary|")
        print("+--------------------+")
        print(summary(model, (1, seq_len)))
    
    ## train the model
    train_losses, val_losses, test_losses = model_run(train_dataloader, val_dataloader, test_dataloader, model, device, num_epochs = num_epoch)

    return train_losses, val_losses, test_losses