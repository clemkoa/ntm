from torch import nn


class Controller(nn.Module):
    def __init__(self, vector_length, hidden_size):
        super(Controller, self).__init__()
        self.layer = nn.LSTM(vector_length, hidden_size)

    def forward(self, x, state):
        output, state = self.layer(x.view(1, 1, -1), state)
        return output.view(1, -1), state
