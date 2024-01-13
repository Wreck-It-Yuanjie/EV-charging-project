import numpy as np
import pandas as pd

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def data_split(data, split_rate = 0.7):
    train_size = int(len(data) * split_rate)
    train_data, test_data = data[:train_size], data[train_size:]
    return train_data, test_data

def create_sequences(data, seq_length):
    sequence = []
    for i in range(len(data) - seq_length):
        one_data = (torch.tensor(data[i:i+seq_length]), data[i+seq_length])
        sequence.append(one_data)
    # print(sequence)
    return sequence

def build_data_loader(data, seq_length = 3, split_rate_train_test = 0.7, split_rate_train_val = 0.7, batch_size = 1, shuffle_train = False, shuffle_val = False, shuffle_test = False):
    '''
    Input the data, return training dataloader, validation dataloader, and test dataloader
    '''
    train_data, test_data = data_split(data, split_rate_train_test)
    train_data, val_data = data_split(train_data, split_rate_train_val)

    training_data_seq = create_sequences(train_data, seq_length)
    val_data_seq = create_sequences(val_data, seq_length)
    test_data_seq = create_sequences(test_data, seq_length)

    train_dataloader = DataLoader(training_data_seq, batch_size, shuffle = shuffle_train) ## Change to true later
    val_dataloader = DataLoader(val_data_seq, batch_size, shuffle = shuffle_val)
    test_dataloader = DataLoader(test_data_seq, batch_size, shuffle = shuffle_test)

    return train_dataloader, val_dataloader, test_dataloader

def show_data_loader(train_loader):
    '''
    a function shows every entry of train_loader
    '''
    for i, data in enumerate(train_loader):
            inputs, labels = data
            print("inputs: ", inputs, "labels: ", labels)