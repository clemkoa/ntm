import torch
from torch import nn
import torch.nn.functional as F
from ntm.utils import circular_convolution, _convolve


class Head(nn.Module):
    def __init__(self, memory, hidden_size):
        super(Head, self).__init__()
        self.memory = memory
        memory_length, memory_vector_length = memory.get_size()
        # (k : vector, beta: scalar, g: scalar, s: vector, gamma: scalar)
        self.k_layer = nn.Linear(hidden_size, memory_vector_length)
        self.beta_layer = nn.Linear(hidden_size, 1)
        self.g_layer = nn.Linear(hidden_size, 1)
        self.s_layer = nn.Linear(hidden_size, 3)
        self.gamma_layer = nn.Linear(hidden_size, 1)
        for layer in [self.k_layer, self.beta_layer, self.g_layer, self.s_layer, self.gamma_layer]:
            nn.init.xavier_uniform_(layer.weight, gain=1.4)
            nn.init.normal_(layer.bias, std=0.01)

    def get_initial_state(self, batch_size):
        return torch.zeros(batch_size, self.memory.get_size()[0])

    def get_head_weight(self, x, previous_state, memory_read):
        k = self.k_layer(x)
        beta = F.softplus(self.beta_layer(x))
        g = F.sigmoid(self.g_layer(x))
        s = F.softmax(self.s_layer(x), dim=1)
        gamma = 1 + F.softplus(self.gamma_layer(x))
        # Focusing by content
        w_c = F.softmax(beta * F.cosine_similarity(memory_read + 1e-16, k.unsqueeze(1) + 1e-16, dim=-1), dim=1)
        # Focusing by location
        w_g = g * w_c + (1 - g) * previous_state
        w_t = _convolve(w_g, s)
        w = w_t ** gamma
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w


class ReadHead(Head):
    def forward(self, x, previous_state):
        memory_read = self.memory.read()
        w = self.get_head_weight(x, previous_state, memory_read)
        return torch.matmul(w, memory_read), w


class WriteHead(Head):
    def __init__(self, memory, hidden_size):
        super(WriteHead, self).__init__(memory, hidden_size)
        memory_length, memory_vector_length = memory.get_size()
        self.e_layer = nn.Linear(hidden_size, memory_vector_length)
        self.a_layer = nn.Linear(hidden_size, memory_vector_length)
        for layer in [self.e_layer, self.a_layer]:
            nn.init.xavier_uniform_(layer.weight, gain=1.4)
            nn.init.normal_(layer.bias, std=0.01)

    def forward(self, x, previous_state):
        memory_read = self.memory.read()
        w = self.get_head_weight(x, previous_state, memory_read)
        e = F.sigmoid(self.e_layer(x))
        a = self.a_layer(x)

        # write to memory (w, memory, e , a)
        self.memory.write(w, e, a)
        return w
