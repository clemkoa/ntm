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
            self._controller = LSTMController(vector_length, hidden_size)
        else:
            self._controller = FeedForwardController(vector_length, hidden_size)

    def forward(self, x, state):
        return self._controller(x, state)

    def get_initial_state(self, batch_size):
        return self._controller.get_initial_state(batch_size)


class LSTMController(nn.Module):
    def __init__(self, vector_length, hidden_size):
        super(LSTMController, self).__init__()
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
        # LSTM configured to accept : sequence_length * batch_size * input || 1 * 1 * input_representation_size
        output, state = self.layer(x.unsqueeze(0), state)
        # Final outputs : time_step * hidden_representation || i.e output at each unrolling, with batch size one. Thus, the squeezing
        # Assumption below : Squeeze the sequence-length dimension, if the first-dimension (sequence-length) is 1.
        return output.squeeze(0), state

    def get_initial_state(self, batch_size):
        # For multiple Batches, clone the same state
        # Currently, Batch_Size is 1. WHY ? .. Gotta Come back in a while
        # as we want minimal training .. Right ? 
        lstm_h = self.lstm_h_state.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_state.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c


class FeedForwardController(nn.Module):
    def __init__(self, vector_length, hidden_size):
        super(FeedForwardController, self).__init__()
        self.layer_1 = nn.Linear(vector_length, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        stdev = 5 / (np.sqrt(vector_length + hidden_size))
        nn.init.uniform_(self.layer_1.weight, -stdev, stdev)
        nn.init.uniform_(self.layer_2.weight, -stdev, stdev)

    def forward(self, x, state):
        x1 = F.relu(self.layer_1(x))
        output = F.relu(self.layer_2(x1))
        return output, state

    def get_initial_state(self):
        return 0, 0
