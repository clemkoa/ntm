import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

class NTM(nn.Module):
    def __init__(self):
        super(NTM, self).__init__()
        self.controller = Controller()
        self.read_head = ReadHead()
        self.write_head = WriteHead()
        self.memory = torch.zeros([10, 20], dtype=torch.int32)

    def forward(self, x):
        return self.controller(x)

class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.layer = nn.Linear(6, 6)

    def forward(self, x):
        return self.layer(x)

class ReadHead(nn.Module):
    def __init__(self):
        super(ReadHead, self).__init__()

    def forward(self, x):
        return x

class WriteHead(nn.Module):
    def __init__(self):
        super(WriteHead, self).__init__()

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
