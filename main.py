import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

from utils import circular_convolution


class NTM(nn.Module):
    def __init__(self, hidden_size, memory_size):
        super(NTM, self).__init__()
        self.controller = Controller(hidden_size)
        self.memory = Memory(memory_size)
        self.read_head = ReadHead(self.memory, hidden_size)
        self.write_head = WriteHead(self.memory, hidden_size)
        self.fc = nn.Linear(hidden_size + memory_size[1], hidden_size)

    def forward(self, x, previous_state):
        previous_read_head_state, previous_write_head_state, previous_controller_state = previous_state
        controller_output, controller_state = self.controller(x, previous_controller_state)
        # Read
        read_head_output, read_head_state = self.read_head(controller_output, previous_read_head_state)

        # Write
        write_head_output, write_head_state = self.write_head(controller_output, previous_read_head_state)
        fc_input = torch.cat((controller_output, read_head_output), 1)
        state = (read_head_state, write_head_state, controller_state)
        return self.fc(fc_input), state


class Controller(nn.Module):
    def __init__(self, hidden_size):
        super(Controller, self).__init__()
        self.layer = nn.LSTM(hidden_size, hidden_size)

    def forward(self, x, state):
        output, state = self.layer(x.view(1, 1, -1), state)
        return output.view(1, -1), state


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
        read = torch.matmul(w, memory)

        # write to memory (w, memory, e , a)
        self.memory.write(w.detach(), e.detach(), a.detach())
        return read, w.detach()


class Memory:
    def __init__(self, memory_size):
        self.memory = torch.ones(memory_size, dtype=torch.float)

    def read(self):
        return self.memory

    def write(self, w, e, a):
        self.memory = self.memory * (1 - torch.matmul(w, e.view(self.memory.shape)))
        self.memory = self.memory + torch.t(w) * a
        return self.memory

    def size(self):
        return self.memory.shape


def get_delimiter_vector(vector_length):
    return - torch.ones(1, 1, vector_length)


def get_training_sequence(sequence_length, vector_length):
    output = []
    for i in range(sequence_length):
        output.append(torch.bernoulli(torch.Tensor(1, vector_length).uniform_(0, 1)))
    output = torch.cat(output)
    output = torch.unsqueeze(output, 1)
    input = torch.cat([output, get_delimiter_vector(vector_length)])
    return input, output


vector_length = 6
memory_size = (10, 20)
hidden_layer_size = 6
sequence_length = 2

model = NTM(hidden_layer_size, memory_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)

feedback_frequence = 100
total_loss = []

model_path = 'models/copy.pt'


checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)

initial_read_head_weights = torch.ones((1, 10)) / 10
initial_write_head_weights = torch.ones((1, 10)) / 10
initial_controller_weights = (torch.ones((1, 1, 6)) / 6, torch.ones((1, 1, 6)) / 6)
for i in range(100000):
    input, target = get_training_sequence(sequence_length, vector_length)
    state = (initial_read_head_weights, initial_write_head_weights, initial_controller_weights)
    optimizer.zero_grad()
    for vector in input:
        output, state = model(vector, state)
    loss = 0.0
    for vector in target:
        output, state = model(get_delimiter_vector(vector_length)[0], state)
        loss += F.mse_loss(output, vector)
    loss.backward()
    total_loss.append(loss.item())
    optimizer.step()
    if i % feedback_frequence == 0:
        print(f'loss at step {i}', sum(total_loss) / len(total_loss))
        total_loss = []

torch.save(model.state_dict(), model_path)
