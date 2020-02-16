import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class NTM(nn.Module):
    def __init__(self, hidden_size, memory_size):
        super(NTM, self).__init__()
        self.controller = Controller(hidden_size)
        self.memory = Memory(memory_size)
        self.read_head = ReadHead(self.memory, hidden_size)
        self.write_head = WriteHead(self.memory, hidden_size)
        self.fc = nn.Linear(hidden_size + memory_size[1], hidden_size)

    def forward(self, x, previous_state):
        previous_read_head_state, previous_write_head_state = previous_state
        controller_output = self.controller(x)
        # Read
        read_head_output, read_head_state = self.read_head(controller_output, previous_read_head_state)

        # Write
        write_head_output, write_head_state = self.write_head(controller_output, previous_read_head_state)
        fc_input = torch.cat((controller_output, read_head_output), 1)
        state = (read_head_state, write_head_state)
        return F.sigmoid(self.fc(fc_input)), state

class Controller(nn.Module):
    def __init__(self, hidden_size):
        super(Controller, self).__init__()
        self.layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.layer(x)

class ReadHead(nn.Module):
    def __init__(self, memory, hidden_size):
        super(ReadHead, self).__init__()
        self.memory = memory
        memory_vector_length = memory.size()[1]
        # (k : vector, beta: scalar, g: scalar, s: vector, gamma: scalar)
        self.k_layer = nn.Linear(hidden_size, memory_vector_length)
        self.beta_layer = nn.Linear(hidden_size, 1)
        self.g_layer = nn.Linear(hidden_size, 1)
        self.s_layer = nn.Linear(hidden_size, memory_vector_length)
        self.gamma_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, previous_state):
        # temporary
        k = self.k_layer(x)
        beta = F.softplus(self.beta_layer(x))
        g = F.sigmoid(self.g_layer(x))
        s = self.s_layer(x)
        gamma = self.gamma_layer(x)

        # Focusing by content
        memory = self.memory.read().detach()
        w_c = F.softmax(beta * F.cosine_similarity(memory, k, dim=-1), dim=1)
        # Focusing by location
        # TODO
        state = g * w_c + (1 - g) * previous_state
        return torch.matmul(state, memory), state.detach()

class WriteHead(nn.Module):
    def __init__(self, memory, hidden_size):
        super(WriteHead, self).__init__()
        self.memory = memory
        memory_length, memory_vector_length = memory.size()
        # (k : vector, beta: scalar, g: scalar, s: vector, gamma: scalar)
        self.k_layer = nn.Linear(hidden_size, memory_vector_length)
        self.beta_layer = nn.Linear(hidden_size, 1)
        self.g_layer = nn.Linear(hidden_size, 1)
        self.s_layer = nn.Linear(hidden_size, memory_vector_length)
        self.gamma_layer = nn.Linear(hidden_size, 1)
        self.e_layer = nn.Linear(hidden_size, memory_vector_length * memory_length)
        self.a_layer = nn.Linear(hidden_size, memory_vector_length)

    def forward(self, x, previous_state):
        # temporary
        k = self.k_layer(x)
        beta = F.softplus(self.beta_layer(x))
        g = F.sigmoid(self.g_layer(x))
        s = self.s_layer(x)
        gamma = self.gamma_layer(x)
        e = F.sigmoid(self.e_layer(x))
        a = self.a_layer(x)

        # Focusing by content
        memory = self.memory.read().detach()
        w_c = F.softmax(beta * F.cosine_similarity(memory, k, dim=-1), dim=1)
        # Focusing by location
        # TODO
        state = g * w_c + (1 - g) * previous_state
        read = torch.matmul(state, memory)

        # write to memory (state, memory, e , a)
        self.memory.write(state, e, a)
        return read, state.detach()

class Memory:
    def __init__(self, memory_size):
        self.memory = torch.ones(memory_size, dtype=torch.float)

    def read(self):
        return self.memory

    def write(self, w, e, a):
        self.memory = self.memory * (1 - torch.matmul(w, e.view(*self.memory.shape)))
        self.memory = self.memory + torch.t(w) * a
        return self.memory

    def size(self):
        return self.memory.shape

vector_length = 8
memory_size = (10, 20)
hidden_layer_size = 6

input = torch.tensor([[0.0, 1.0, 0, 1, 1, 0]])
target = torch.tensor([[0.0, 1.0, 0, 1, 1, 0]])

initial_read_head_weights = torch.ones((1, 10))
initial_write_head_weights = torch.ones((1, 10))
state = (initial_read_head_weights, initial_write_head_weights)

model = NTM(hidden_layer_size, memory_size)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for i in range(1000):
    optimizer.zero_grad()
    output, state = model(input, state)
    loss = F.mse_loss(output, target)
    print(loss)
    loss.backward()
    optimizer.step()
