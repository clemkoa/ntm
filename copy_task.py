import random
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from ntm.ntm import NTM
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--train", help="Trains the model", action="store_true")
parser.add_argument("--eval", help="Evaluates the model. Default path is models/copy.pt", action="store_true")
parser.add_argument("--modelpath", help="Specify the model path to load, for training or evaluation", type=str)
parser.add_argument("--epochs", help="Specify the number of epochs for training", type=int, default=50_000)
args = parser.parse_args()

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def plot_copy_results(target, bin_y, y, sequence_min_length, vector_length):
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax1.set_ylabel("target", rotation=0, labelpad=20)
    ax1.imshow(torch.t(target.view(sequence_min_length, vector_length)))
    ax1.tick_params(axis="both", which="both", length=0)
    ax2 = fig.add_subplot(312)
    ax2.set_ylabel("binarized output", rotation=0, labelpad=50)
    ax2.imshow(torch.t(bin_y.view(sequence_min_length, vector_length)))
    ax2.tick_params(axis="both", which="both", length=0)
    ax3 = fig.add_subplot(313)
    ax3.set_ylabel("output", rotation=0, labelpad=20)
    ax3.imshow(torch.t(y.clone().data.view(sequence_min_length, vector_length)))
    ax3.tick_params(axis="both", which="both", length=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.show()


def get_training_sequence(sequence_min_length, sequence_max_length, vector_length):
    sequence_length = random.randint(sequence_min_length, sequence_max_length)
    output = torch.bernoulli(torch.Tensor(sequence_length, vector_length).uniform_(0, 1))
    output = torch.unsqueeze(output, 1)
    input = torch.zeros(sequence_length + 1, 1, vector_length + 1)
    input[:sequence_length, :, :vector_length] = output
    input[sequence_length, :, vector_length] = 1.0
    return input, output


def train(epochs=50_000):
    print(f"Training for {epochs} epochs")
    sequence_min_length = 1
    sequence_max_length = 20
    vector_length = 8
    memory_size = (128, 20)
    hidden_layer_size = 100

    model = NTM(vector_length, hidden_layer_size, memory_size)

    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.RMSprop(model.parameters(), momentum=0.9, alpha=0.95, lr=1e-4)
    feedback_frequence = 100
    total_loss = []

    os.makedirs("models", exist_ok=True)
    if os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)

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
        total_loss.append(loss.item())
        if i % feedback_frequence == 0:
            print(f"Loss at step {i}", sum(total_loss) / len(total_loss))
            total_loss = []

    torch.save(model.state_dict(), model_path)


def eval(model_path):
    vector_length = 8
    memory_size = (128, 20)
    hidden_layer_size = 100

    model = NTM(vector_length, hidden_layer_size, memory_size)

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    lengths = [20, 100]
    for l in lengths:
        sequence_min_length = l
        sequence_max_length = l
        input, target = get_training_sequence(sequence_min_length, sequence_max_length, vector_length)
        state = model.get_initial_state()
        for vector in input:
            _, state = model(vector, state)
        y_out = torch.zeros(target.size())
        for j in range(len(target)):
            y_out[j], state = model(torch.zeros(1, vector_length + 1), state)
        y_out_binarized = y_out.clone().data
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

        plot_copy_results(target, y_out_binarized, y_out, sequence_min_length, vector_length)


if __name__ == "__main__":
    model_path = "models/copy.pt"
    if args.modelpath:
        model_path = args.modelpath
    if args.train:
        train(args.epochs)
    if args.eval:
        eval(model_path)
