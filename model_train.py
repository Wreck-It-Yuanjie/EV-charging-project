import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def model_loss(model, loss_function, x, y, optimizer = None):
    '''
    Apply loss function to a batch of inputs. If no optimizer is provided, 
    skip the back propagation step
    '''
    ## prediction
    # print("         Enter function model_loss:\n")
    output = model(x.float())
    # print("             output:", output)

    loss = loss_function(output, y.float())

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # print("         model_loss:", loss.item(), "\n")
    return loss.item(), len(x)

def train_one_epoch(model, train_dl, loss_function, device, optimizer):
    '''
    Execute 1 set of batched training within an epoch
    '''
    # print("     Enter function train_one_epoch():\n")
    ## Set the model to training mode
    model.train()
    train_losses = []
    batch_sizes = []

    ## Loop through train dataloader
    for x_train, y_train in train_dl:
        # print("train data x: ", x_train)
        # print("train data y: ", y_train)
        ## transfer the data to GPU if any
        x_train, y_train = x_train.to(device), y_train.to(device)

        ## Back propagation
        train_loss, batch_size = model_loss(model, loss_function, x_train, y_train, optimizer)

        ## Append train loss and batch size
        train_losses.append(train_loss)
        batch_sizes.append(batch_size)
    ## Calculate the average losses over all batches
    train_loss = np.sum(np.multiply(train_losses, batch_sizes)) / np.sum(batch_sizes)
    # print("batch train loss: ", train_loss)

    return train_loss

def val_one_epoch(model,val_dl,loss_function,device, optimizer):
    '''
    Excute 1 set of batched validation within an epoch
    '''
    # print("     Enter function val_one_epoch():\n")
    model.eval()
    with torch.no_grad():
        validation_losses = []
        batch_sizes = []

        ## Loop through the validation dataloader
        for x_valid, y_valid in val_dl:
            ## transfer the data to GPU if any
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)

            ## Calculate the loss WITHOUT BACK PROPOGATION
            validation_loss, batch_size = model_loss(model, loss_function, x_valid, y_valid)

            ## Append validation loss and batch size
            validation_losses.append(validation_loss)
            batch_sizes.append(batch_size)

        ## Calculate the average losses over all batches
        val_loss = np.sum(np.multiply(validation_losses, batch_sizes)) / np.sum(batch_sizes)
        # print("batch validation loss: ", val_loss)

        return val_loss
            