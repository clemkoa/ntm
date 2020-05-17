import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from ntm.ntm import NTM
import numpy as np
import matplotlib.pyplot as plt
random.seed(2)
np.random.seed(2)
torch.manual_seed(2)


def get_training_sequence(sequence_min_length, sequence_max_length, vector_length):
    sequence_length = random.randint(sequence_min_length, sequence_max_length)
    output = []
    for i in range(sequence_length):
        output.append(torch.bernoulli(torch.Tensor(1, vector_length).uniform_(0, 1)))
    output = torch.cat(output)
    output = torch.unsqueeze(output, 1)
    input = torch.zeros(sequence_length + 1, 1, vector_length + 1)
    input[:sequence_length, :, :vector_length] = output
    input[sequence_length, :, vector_length] = 1.0
    return input, output


def train():
    sequence_min_length = 1
    sequence_max_length = 20
    vector_length = 8
    memory_size = (128, 20)
    hidden_layer_size = 100
    epochs = 50_000

    model = NTM(vector_length, hidden_layer_size, memory_size)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    feedback_frequence = 100
    total_loss = []

    # model_path = 'models/copy.pt'

    # checkpoint = torch.load(model_path)
    # model.load_state_dict(checkpoint)

    for i in range(epochs):
        optimizer.zero_grad()
        input, target = get_training_sequence(sequence_min_length, sequence_max_length, vector_length)
        state = model.get_initial_state()
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
        total_loss.append(loss.item())
        if i % feedback_frequence == 0:
            print(f'cost at step {i}', sum(total_loss) / len(total_loss))
            total_loss = []

    # torch.save(model.state_dict(), model_path)


def eval():
    vector_length = 8
    memory_size = (128, 20)
    hidden_layer_size = 100

    model = NTM(vector_length, hidden_layer_size, memory_size)

    model_path = 'models/copy-100-20.pt'
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    eval_epochs = 4
    for i in range(eval_epochs):
        sequence_min_length = (i + 1) * 4
        sequence_max_length = (i + 1) * 4
        input, target = get_training_sequence(sequence_min_length, sequence_max_length, vector_length)
        state = model.get_initial_state()
        for vector in input:
            _, state = model(vector, state)
        y_out = torch.zeros(target.size())
        for j in range(len(target)):
            y_out[j], state = model(torch.zeros(1, vector_length + 1), state)
        y_out_binarized = y_out.clone().data
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
        plt.subplot(211)
        plt.imshow(target.view(sequence_min_length, vector_length))
        plt.axis('off')
        plt.subplot(212)
        plt.imshow(y_out_binarized.view(sequence_min_length, vector_length))
        plt.axis('off')
        # plt.savefig(f"output_{i + 1}.png")
        plt.show()

if __name__ == "__main__":
    train()
    # eval()
