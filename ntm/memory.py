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

        # Initial read vector is a learnt parameter
        self.initial_read = Parameter(torch.randn(1, self._memory_size[1]) * 0.01)

    def get_size(self):
        return self._memory_size

    def reset(self, batch_size):
        self.memory = self.initial_state.clone().repeat(batch_size, 1, 1)

    def get_initial_read(self, batch_size):
        return self.initial_read.clone().repeat(batch_size, 1)

    def read(self):
        return self.memory

    def write(self, w, e, a):
        self.memory = self.memory * (1 - torch.matmul(w.unsqueeze(-1), e.unsqueeze(1)))
        self.memory = self.memory + torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        return self.memory

    def size(self):
        return self._memory_size
