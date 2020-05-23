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

        initial_read = torch.randn(1, self._memory_size[1]) * 0.01
        self.register_buffer("initial_read", initial_read.data)

    def get_size(self):
        return self._memory_size

    def reset(self, batch_size):
        self.memory = self.intial_state.clone().repeat(batch_size, 1, 1)

    def get_initial_state(self, batch_size):
        return self.initial_read.clone().repeat(batch_size, 1)

    def read(self):
        return self.memory

    def write(self, w, e, a):
        self.memory = self.memory * (1 - torch.matmul(torch.t(w), e))
        self.memory = self.memory + torch.matmul(torch.t(w), a)
        return self.memory

    def size(self):
        return self._memory_size
