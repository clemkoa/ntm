import torch
from torch import nn
from torch.nn import Parameter


class Controller(nn.Module):
    def __init__(self, vector_length, hidden_size):
        super(Controller, self).__init__()
        self.layer = nn.LSTM(vector_length, hidden_size)
        # The hidden state is a learned parameter
        self.lstm_h_state = Parameter(torch.randn(1, 1, hidden_size) * 0.05)
        self.lstm_c_state = Parameter(torch.randn(1, 1, hidden_size) * 0.05)

    def forward(self, x, state):
        output, state = self.layer(x.view(1, 1, -1), state)
        return output.view(1, -1), state

    def get_initial_state(self):
        lstm_h = self.lstm_h_state.clone()
        lstm_c = self.lstm_c_state.clone()
        return lstm_h, lstm_c
