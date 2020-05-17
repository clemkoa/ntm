import torch
from torch import nn
import numpy as np


class Memory(nn.Module):
    def __init__(self, memory_size):
        super(Memory, self).__init__()
        self._memory_size = memory_size
        self.register_buffer('mem_bias', torch.Tensor(memory_size[0], memory_size[1]))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(memory_size[0] + memory_size[1]))
        nn.init.uniform_(self.mem_bias, -stdev, stdev)
        self.reset()

    def get_size(self):
        return self._memory_size

    def reset(self):
        self.memory = self.mem_bias.clone()

    def get_initial_state(self):
        return torch.zeros((1, self._memory_size[1]))

    def read(self):
        return self.memory

    def write(self, w, e, a):
        self.memory = self.memory * (1 - torch.matmul(w, e.view(self.memory.shape)))
        self.memory = self.memory + torch.t(w) * a
        return self.memory

    def size(self):
        return self.memory.shape
