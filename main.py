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
        self.weights = nn.Linear(6, 10)
        self.memory = memory

    def forward(self, x, previous_state):
        return torch.matmul(self.weights(x), self.memory), previous_state

class WriteHead(nn.Module):
    def __init__(self, memory):
        super(WriteHead, self).__init__()
        self.memory = memory

    def forward(self, x):
        return x


input = torch.tensor([[0.0, 1.0, 0, 1, 1, 0]])
target = torch.tensor([[0.0, 1.0, 0, 1, 1, 0]])

initial_read_head_weights = torch.ones((6, 1))
initial_write_head_weights = torch.ones((6, 1))
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
