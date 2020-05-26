import random
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from ntm.ntm import NTM
from ntm.utils import plot_copy_results
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("--train", help="Trains the model", action="store_true")
parser.add_argument("--ff", help="Feed forward controller", action="store_true")
parser.add_argument("--eval", help="Evaluates the model. Default path is models/copy.pt", action="store_true")
parser.add_argument("--modelpath", help="Specify the model path to load, for training or evaluation", type=str)
parser.add_argument("--epochs", help="Specify the number of epochs for training", type=int, default=50_000)
args = parser.parse_args()

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def get_training_sequence(sequence_min_length, sequence_max_length, vector_length, batch_size=1):
    sequence_length = random.randint(sequence_min_length, sequence_max_length)
    output = torch.bernoulli(torch.Tensor(sequence_length, batch_size, vector_length).uniform_(0, 1))
    input = torch.zeros(sequence_length + 1, batch_size, vector_length + 1)
    input[:sequence_length, :, :vector_length] = output
    input[sequence_length, :, vector_length] = 1.0
    return input, output


def train(epochs=50_000):
    tensorboard_log_folder = f"runs/copy-task-{datetime.now().strftime('%Y-%m-%dT%H%M%S')}"
    writer = SummaryWriter(tensorboard_log_folder)
    print(f"Training for {epochs} epochs, logging in {tensorboard_log_folder}")
    sequence_min_length = 1
    sequence_max_length = 20
    vector_length = 8
    memory_size = (128, 20)
    hidden_layer_size = 100
    batch_size = 4
    lstm_controller = not args.ff

    writer.add_scalar("sequence_min_length", sequence_min_length)
    writer.add_scalar("sequence_max_length", sequence_max_length)
    writer.add_scalar("vector_length", vector_length)
    writer.add_scalar("memory_size0", memory_size[0])
    writer.add_scalar("memory_size1", memory_size[1])
    writer.add_scalar("hidden_layer_size", hidden_layer_size)
    writer.add_scalar("lstm_controller", lstm_controller)
    writer.add_scalar("seed", seed)
    writer.add_scalar("batch_size", batch_size)

    model = NTM(vector_length, hidden_layer_size, memory_size, lstm_controller)

    optimizer = optim.RMSprop(model.parameters(), momentum=0.9, alpha=0.95, lr=1e-4)
    feedback_frequency = 100
    total_loss = []
    total_cost = []

    os.makedirs("models", exist_ok=True)
    if os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)

    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        input, target = get_training_sequence(sequence_min_length, sequence_max_length, vector_length, batch_size)
        state = model.get_initial_state(batch_size)
        for vector in input:
            _, state = model(vector, state)
        y_out = torch.zeros(target.size())
        for j in range(len(target)):
            y_out[j], state = model(torch.zeros(batch_size, vector_length + 1), state)
        loss = F.binary_cross_entropy(y_out, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
        y_out_binarized = y_out.clone().data
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)
        cost = torch.sum(torch.abs(y_out_binarized - target)) / len(target)
        total_cost.append(cost.item())
        if epoch % feedback_frequency == 0:
            running_loss = sum(total_loss) / len(total_loss)
            running_cost = sum(total_cost) / len(total_cost)
            print(f"Loss at step {epoch}: {running_loss}")
            writer.add_scalar('training loss', running_loss, epoch)
            writer.add_scalar('training cost', running_cost, epoch)
            total_loss = []
            total_cost = []

    torch.save(model.state_dict(), model_path)


def eval(model_path):
    vector_length = 8
    memory_size = (128, 20)
    hidden_layer_size = 100
    lstm_controller = not args.ff

    model = NTM(vector_length, hidden_layer_size, memory_size, lstm_controller)

    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)

    lengths = [20, 100]
    for l in lengths:
        sequence_length = l
        input, target = get_training_sequence(sequence_length, sequence_length, vector_length)
        state = model.get_initial_state()
        for vector in input:
            _, state = model(vector, state)
        y_out = torch.zeros(target.size())
        for j in range(len(target)):
            y_out[j], state = model(torch.zeros(1, vector_length + 1), state)
        y_out_binarized = y_out.clone().data
        y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

        plot_copy_results(target, y_out, vector_length)


if __name__ == "__main__":
    model_path = "models/copy.pt"
    if args.modelpath:
        model_path = args.modelpath
    if args.train:
        train(args.epochs)
    if args.eval:
        eval(model_path)
