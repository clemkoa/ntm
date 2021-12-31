import torch
from torch import nn
import torch.nn.functional as F
from ntm.controller import Controller
from ntm.memory import Memory
from ntm.head import ReadHead, WriteHead


class NTM(nn.Module):
    def __init__(self, vector_length, hidden_size, memory_size, lstm_controller=True):
        super(NTM, self).__init__()
        self.controller = Controller(lstm_controller, vector_length + 1 + memory_size[1], hidden_size)
        self.memory = Memory(memory_size)
        self.read_head = ReadHead(self.memory, hidden_size)
        self.write_head = WriteHead(self.memory, hidden_size)
        self.fc = nn.Linear(hidden_size + memory_size[1], vector_length)
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def get_initial_state(self, batch_size=1):
        self.memory.reset(batch_size)
        controller_state = self.controller.get_initial_state(batch_size)
        read = self.memory.get_initial_read(batch_size)
        read_head_state = self.read_head.get_initial_state(batch_size)
        write_head_state = self.write_head.get_initial_state(batch_size)
        return (read, read_head_state, write_head_state, controller_state)

    def forward(self, x, previous_state):

        ## Whether is LSTM or Feedforward or any network of choice, we take in inputs at only one time step
        ## This input is processed by network to create feature
        ## This feature is then, used to update (read and write ) the Memory-Matrix
        ## So at each time step, we keep on retrieving contents and updating the memory matrix
        ## The retrieved content + Controller input -> A fully Connected Layer -> Output at each time-step

        previous_read, previous_read_head_state, previous_write_head_state, previous_controller_state = previous_state
        controller_input = torch.cat([x, previous_read], dim=1)
        ## LSTM will take in : 1 (seq_length) * 1 (batch_size) * 29 (input_repr_space)
        ## is time-step 1 a Design Choice. NEED to look at other NTM implementations
        ## and output : seq_len * batch_size * 100 (hidden repr) => with first dimension squeezed, if 1 
        controller_output, controller_state = self.controller(controller_input, previous_controller_state)
        # Read
        read_head_output, read_head_state = self.read_head(controller_output, previous_read_head_state)
        # Write
        write_head_state = self.write_head(controller_output, previous_write_head_state)
        # 
        fc_input = torch.cat((controller_output, read_head_output), dim=1)
        state = (read_head_output, read_head_state, write_head_state, controller_state)
        return F.sigmoid(self.fc(fc_input)), state
