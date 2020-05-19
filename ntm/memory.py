import torch
from torch import nn
import numpy as np


class Memory(nn.Module):
    def __init__(self, memory_size):
        super(Memory, self).__init__()
        self._memory_size = memory_size

        # Initialize memory bias
        stdev = 1 / (np.sqrt(memory_size[0] + memory_size[1]))
        intial_state = torch.Tensor(memory_size[0], memory_size[1]).uniform_(-stdev, stdev)
        self.register_buffer('intial_state', intial_state.data)
        self.reset()

        initial_read = torch.randn(1, self._memory_size[1]) * 0.01
        self.register_buffer("initial_read", initial_read.data)

    def get_size(self):
        return self._memory_size

    def reset(self):
        self.memory = self.intial_state.clone()

    def get_initial_state(self):
        return self.initial_read.clone()

    def read(self):
        return self.memory

    def write(self, w, e, a):
        self.memory = self.memory * (1 - torch.t(w) * e)
        self.memory = self.memory + torch.t(w) * a
        return self.memory

    def size(self):
        return self.memory.shape
