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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # uncoment for for lstm
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # for gru and rnn
        #out, _ = self.rnn1(x, h0)

        # for lstm
        #out, _ = self.rnn1(x, (h0, c0))

        out = torch.reshape(out, (out.shape[0], -1))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))

        # BCE with logits combines a sigmoid layer for the output node
        out = self.fc3(out)
        return out
