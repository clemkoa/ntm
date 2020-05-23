import torch
from torch import nn
from torch.nn import Parameter


class Memory(nn.Module):
    def __init__(self, memory_size):
        super(Memory, self).__init__()
        self._memory_size = memory_size

        # Initialize memory bias
        initial_state = torch.ones(memory_size) * 1e-6
        self.register_buffer('initial_state', initial_state.data)
        self.reset()

        # Initial read vector is a learnt parameter
        self.initial_read = Parameter(torch.randn(1, self._memory_size[1]) * 0.01)

    def get_size(self):
        return self._memory_size

    def reset(self):
        self.memory = self.initial_state.clone()

    def get_initial_read(self):
        return self.initial_read.clone()

    def read(self):
        return self.memory

    def write(self, w, e, a):
        self.memory = self.memory * (1 - torch.t(w) * e)
        self.memory = self.memory + torch.t(w) * a
        return self.memory

    def size(self):
        return self.memory.shape
