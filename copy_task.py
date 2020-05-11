import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from ntm.ntm import NTM


def get_training_sequence(sequence_length, vector_length):
    output = []
    for i in range(sequence_length):
        output.append(torch.bernoulli(torch.Tensor(1, vector_length).uniform_(0, 1)))
    output = torch.cat(output)
    output = torch.unsqueeze(output, 1)
    input = torch.zeros(sequence_length + 1, 1, vector_length + 1)
    input[:sequence_length, :, :vector_length] = output
    input[sequence_length, :, vector_length] = 1.0
    return input, output


vector_length = 8
memory_size = (128, 20)
hidden_layer_size = 100
epochs = 50_000

model = NTM(vector_length, hidden_layer_size, memory_size)
optimizer = optim.Adam(model.parameters(), lr=0.005)

feedback_frequence = 100
total_loss = []

model_path = 'models/copy.pt'


# checkpoint = torch.load(model_path)
# model.load_state_dict(checkpoint)

initial_read_head_weights = torch.zeros((1, memory_size[0])).uniform_(-0.1, 0.1)
initial_write_head_weights = torch.zeros((1, memory_size[0])).uniform_(-0.1, 0.1)
initial_read = torch.zeros((1, memory_size[1]))
for i in range(epochs):
    optimizer.zero_grad()
    sequence_length = random.randint(1, 10)
    input, target = get_training_sequence(sequence_length, vector_length)
    model.memory.initialise()
    initial_controller_weights = (torch.ones((1, 1, hidden_layer_size)).uniform_(-0.1, 0.1), torch.ones((1, 1, hidden_layer_size)).uniform_(-0.1, 0.1))
    state = (initial_read, initial_read_head_weights, initial_write_head_weights, initial_controller_weights)
    for vector in input:
        _, state = model(vector, state)
    y_out = torch.zeros(target.size())
    for j in range(len(target)):
        y_out[j], state = model(torch.zeros(1, vector_length + 1), state)
    loss = F.binary_cross_entropy(y_out, target)
    loss.backward()
    optimizer.step()

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
    cost = torch.sum(torch.abs(y_out_binarized - target))
    total_loss.append(cost.item() / sequence_length)
    if i % feedback_frequence == 0:
        print(f'cost at step {i}', sum(total_loss) / len(total_loss))
        total_loss = []

# torch.save(model.state_dict(), model_path)
