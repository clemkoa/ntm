import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class NTM(nn.Module):
    def __init__(self):
        super(NTM, self).__init__()
        self.controller = Controller()
        self.memory = torch.ones([10, 20], dtype=torch.float)
        self.read_head = ReadHead(self.memory)
        self.write_head = WriteHead(self.memory)
        self.fc = nn.Linear(6 + 20, 6)

    def forward(self, x, previous_state):
        previous_read_head_state, previous_write_head_state = previous_state
        controller_output = self.controller(x)
        read_output, read_state = self.read_head(controller_output, previous_read_head_state)

        fc_input = torch.cat((controller_output, read_output), 1)
        state = (read_state, previous_read_head_state)
        return F.sigmoid(self.fc(fc_input)), state

class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.layer = nn.Linear(6, 6)

    def forward(self, x):
        return self.layer(x)

class ReadHead(nn.Module):
    def __init__(self, memory):
        super(ReadHead, self).__init__()
        # (k : vector, beta: scalar, g: scalar, s: vector, gamma: scalar)
        self.k_layer = nn.Linear(6, 20)
        self.beta_layer = nn.Linear(6, 1)
        self.g_layer = nn.Linear(6, 1)
        self.s_layer = nn.Linear(6, 20)
        self.gamma_layer = nn.Linear(6, 1)
        self.memory = memory

    def forward(self, x, previous_state):
        # temporary
        k = self.k_layer(x)
        beta = F.softplus(self.beta_layer(x))
        g = F.sigmoid(self.g_layer(x))
        s = self.s_layer(x)
        gamma = self.gamma_layer(x)

        # Focusing by content
        w_c = F.softmax(beta * F.cosine_similarity(self.memory, k, dim=-1), dim=1)
        # Focusing by location
        # TODO
        state = g * w_c + (1 - g) * previous_state
        return torch.matmul(state, self.memory), state.detach()

class WriteHead(nn.Module):
    def __init__(self, memory):
        super(WriteHead, self).__init__()
        self.memory = memory

    def forward(self, x):
        return x


input = torch.tensor([[0.0, 1.0, 0, 1, 1, 0]])
target = torch.tensor([[0.0, 1.0, 0, 1, 1, 0]])

initial_read_head_weights = torch.ones((1, 10))
initial_write_head_weights = torch.ones((1, 10))
state = (initial_read_head_weights, initial_write_head_weights)

model = NTM()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for i in range(1000):
    optimizer.zero_grad()
    output, state = model(input, state)
    loss = F.mse_loss(output, target)
    print(loss)
    loss.backward()
    optimizer.step()
