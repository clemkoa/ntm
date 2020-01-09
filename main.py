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

    def forward(self, x):
        controller_output = self.controller(x)
        read = self.read_head(controller_output)
        fc_input = torch.cat((controller_output, read), 1)
        return F.sigmoid(self.fc(fc_input))

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

    def forward(self, x):
        return torch.matmul(self.weights(x), self.memory)

class WriteHead(nn.Module):
    def __init__(self, memory):
        super(WriteHead, self).__init__()
        self.weights = nn.Linear(6, 10)
        self.memory = memory

    def forward(self, x):
        return x


input = torch.tensor([[0.0, 1.0, 0, 1, 1, 0]])
target = torch.tensor([[0.0, 1.0, 0, 1, 1, 0]])

model = NTM()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

for i in range(1000):
    optimizer.zero_grad()
    output = model(input)
    loss = F.mse_loss(output, target)
    print(loss)
    loss.backward()
    optimizer.step()
