import torch
from torch import nn
from torch.nn import Parameter
import numpy as np
import torch.nn.functional as F


class Controller(nn.Module):
    def __init__(self, lstm_controller, vector_length, hidden_size):
        super(Controller, self).__init__()
        # We allow either a feed-forward network or a LSTM for the controller
        self._lstm_controller = lstm_controller
        if self._lstm_controller:
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
        else:
            self.layer_1 = nn.Linear(vector_length, hidden_size)
            self.layer_2 = nn.Linear(hidden_size, hidden_size)
            stdev = 5 / (np.sqrt(vector_length + hidden_size))
            nn.init.uniform_(self.layer_1.weight, -stdev, stdev)
            nn.init.uniform_(self.layer_2.weight, -stdev, stdev)

    def forward(self, x, state):
        if self._lstm_controller:
            output, state = self.layer(x.view(1, 1, -1), state)
            output = output.view(1, -1)
        else:
            x1 = F.relu(self.layer_1(x))
            output = F.relu(self.layer_2(x1))
        return output, state

    def get_initial_state(self):
        if self._lstm_controller:
            lstm_h = self.lstm_h_state.clone()
            lstm_c = self.lstm_c_state.clone()
            return lstm_h, lstm_c
        return 0, 0
