import torch
from torch import nn
import torch.nn.functional as F
from ntm.utils import circular_convolution


class ReadHead(nn.Module):
    def __init__(self, memory, hidden_size):
        super(ReadHead, self).__init__()
        self.memory = memory
        memory_length, memory_vector_length = memory.size()
        # (k : vector, beta: scalar, g: scalar, s: vector, gamma: scalar)
        self.k_layer = nn.Linear(hidden_size, memory_vector_length)
        self.beta_layer = nn.Linear(hidden_size, 1)
        self.g_layer = nn.Linear(hidden_size, 1)
        self.s_layer = nn.Linear(hidden_size, memory_length)
        self.gamma_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, previous_state):
        # temporary
        k = self.k_layer(x)
        beta = F.softplus(self.beta_layer(x))
        g = F.sigmoid(self.g_layer(x))
        s = F.softmax(self.s_layer(x))
        gamma = 1 + F.softplus(self.gamma_layer(x))
        # Focusing by content
        memory = self.memory.read().detach()
        w_c = F.softmax(beta * F.cosine_similarity(memory, k, dim=-1), dim=1)
        # Focusing by location
        w_g = g * w_c + (1 - g) * previous_state
        w_t = circular_convolution(w_g, s)
        w = w_t ** gamma
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return torch.matmul(w, memory), w.detach()


class WriteHead(nn.Module):
    def __init__(self, memory, hidden_size):
        super(WriteHead, self).__init__()
        self.memory = memory
        memory_length, memory_vector_length = memory.size()
        # (k : vector, beta: scalar, g: scalar, s: vector, gamma: scalar)
        self.k_layer = nn.Linear(hidden_size, memory_vector_length)
        self.beta_layer = nn.Linear(hidden_size, 1)
        self.g_layer = nn.Linear(hidden_size, 1)
        self.s_layer = nn.Linear(hidden_size, memory_length)
        self.gamma_layer = nn.Linear(hidden_size, 1)
        self.e_layer = nn.Linear(hidden_size, memory_vector_length * memory_length)
        self.a_layer = nn.Linear(hidden_size, memory_vector_length)

    def forward(self, x, previous_state):
        # temporary
        k = self.k_layer(x)
        beta = F.softplus(self.beta_layer(x))
        g = F.sigmoid(self.g_layer(x))
        s = F.softmax(self.s_layer(x))
        gamma = 1 + F.softplus(self.gamma_layer(x))
        e = F.sigmoid(self.e_layer(x))
        a = self.a_layer(x)

        # Focusing by content
        memory = self.memory.read().detach()
        w_c = F.softmax(beta * F.cosine_similarity(memory, k, dim=-1), dim=1)
        # Focusing by location
        w_g = g * w_c + (1 - g) * previous_state
        w_t = circular_convolution(w_g, s)
        w = w_t ** gamma
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)

        # write to memory (w, memory, e , a)
        self.memory.write(w.detach(), e.detach(), a.detach())
        return w.detach()
