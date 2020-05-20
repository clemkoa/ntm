import torch
from torch import nn
from torch.nn import Parameter
import numpy as np
import torch.nn.functional as F


class Controller(nn.Module):
    def __init__(self, vector_length, hidden_size):
        super(Controller, self).__init__()
        self.layer_1 = nn.Linear(vector_length, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        stdev = 5 / (np.sqrt(vector_length + hidden_size))
        nn.init.uniform_(self.layer_1.weight, -stdev, stdev)
        nn.init.uniform_(self.layer_2.weight, -stdev, stdev)

    def forward(self, x):
        x1 = F.relu(self.layer_1(x))
        x2 = F.relu(self.layer_2(x1))
        return x2

    def get_initial_state(self):
        return 0, 0
