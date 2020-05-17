import torch
import numpy as np


class Memory:
    def __init__(self, memory_size):
        self._memory_size = memory_size
        self.reset()

    def get_size(self):
        return self._memory_size

    def reset(self):
        self.memory = torch.zeros(self._memory_size)

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
