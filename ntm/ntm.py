import torch
from torch import nn
import torch.nn.functional as F
from ntm.controller import Controller
from ntm.memory import Memory
from ntm.head import ReadHead, WriteHead


class NTM(nn.Module):
    def __init__(self, vector_length, hidden_size, memory_size):
        super(NTM, self).__init__()
        self.controller = Controller(vector_length + 1 + memory_size[1], hidden_size)
        self.memory = Memory(memory_size)
        self.read_head = ReadHead(self.memory, hidden_size)
        self.write_head = WriteHead(self.memory, hidden_size)
        self.fc = nn.Linear(hidden_size + memory_size[1], vector_length)
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def get_initial_state(self):
        self.memory.reset()
        read = self.memory.get_initial_state()
        read_head_state = self.read_head.get_initial_state()
        write_head_state = self.write_head.get_initial_state()
        return (read, read_head_state, write_head_state)

    def forward(self, x, previous_state):
        previous_read, previous_read_head_state, previous_write_head_state = previous_state
        controller_input = torch.cat([x, previous_read], dim=1)
        controller_output = self.controller(controller_input)
        # Read
        read_head_output, read_head_state = self.read_head(controller_output, previous_read_head_state)
        # Write
        write_head_state = self.write_head(controller_output, previous_read_head_state)
        fc_input = torch.cat((controller_output, read_head_output), dim=1)
        state = (read_head_output, read_head_state, write_head_state)
        return F.sigmoid(self.fc(fc_input)), state
