import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

## Define a simple linear model
class linear_Net(nn.Module):
    '''
    A very simple linear model
    '''
    def __init__(self, seq_unit, output_size = 1):
        super().__init__()
        self.seq_unit = seq_unit
        self.fc1 = nn.Linear(seq_unit, output_size)
    
    def forward(self, x):
        x = torch.flatten(x, 1) 
        x = self.fc1(x)
        return x

class mlp_Net(nn.Module):
    '''
    A basic multiple linear perceptron model with a hidden layer
    '''
    def __init__(self, seq_unit, hidden_size, output_size = 1):
        super().__init__()
        self.activation = F.relu
        self.fc1 = nn.Linear(seq_unit, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.flatten(x, 1) 
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class cnn_Net(nn.Module):
    '''
    A basic convolutional neural network with a number of filters
    '''
    def __init__(self, seq_unit, kernel_size = 3, output_size = 1):
        super().__init__()
        self.seq_unit = seq_unit
        self.conv_net = nn.Sequential(
            ## vector size: 3
            nn.Conv1d(in_channels=1, out_channels= 16, kernel_size=kernel_size, stride=1, padding=1), ## 4x16
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(seq_unit*16, 1)
        )
    
    def forward(self, x):
        ## permute to put channel in correct order
        # x = x.permute(0, 2, 1)
        out = self.conv_net(x)
        return x

## Define the LSTM
class lstm_Net(nn.Module):
    '''
    A long short term memory model
    '''
    def __init__(self, seq_unit, hidden_size = 32, output_size = 1):
        super().__init__()
        self.lstm = nn.LSTM(seq_unit, hidden_size, num_layers = 1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        # x = self.fc(x[:, -1, :]) ## extract only the last time step
        return x