import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


class Controller(nn.Module):
    def __init__(self, vector_length, hidden_size):
        super(Controller, self).__init__()
        self.layer = nn.LSTM(input_size=vector_length, hidden_size=hidden_size)
        # The hidden state is a learned parameter
        self.lstm_h_state = Parameter(torch.randn(1, 1, hidden_size) * 0.05)
        self.lstm_c_state = Parameter(torch.randn(1, 1, hidden_size) * 0.05)
        for p in self.layer.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(vector_length + hidden_size))
                nn.init.uniform_(p, -stdev, stdev)

    def forward(self, x, state):
        output, state = self.layer(x.view(1, 1, -1), state)
        return output.view(1, -1), state

    def get_initial_state(self):
        lstm_h = self.lstm_h_state.clone()
        lstm_c = self.lstm_c_state.clone()
        return lstm_h, lstm_c
