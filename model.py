import torch
import torch.nn as nn

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
                           num_layers=num_layers,
                           batch_first=True)

        self.fc1 = nn.Linear(512, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        print('size of x in the forward pass {} '.format(x.shape))
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        #        print(h0)
        out, _ = self.rnn1(x, h0)
        #        print("Output Shape in model is : %" % out.shape())
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = F.softmax(out, dim=0)
        # out = out.reshape(out.shape[0], -1)
        return out
