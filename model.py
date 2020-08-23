import torch
import torch.nn as nn
import torch.autograd as Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 sequence_length):
        super(Net, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.rnn1 = nn.RNN(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=1)

        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size)
        #        print(h0)
        out, _ = self.rnn1(x, h0)
        #        print("Output Shape in model is : %" % out.shape())
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.softmax(self.fc3(out))
        print('shape of output is {}'.format(out.size()))
        # out = out.reshape(out.shape[0], -1)
        return out
