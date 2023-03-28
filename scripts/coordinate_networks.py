from enum import unique
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from nerf2D import Positional_Encoding
import random
import torchmetrics
import seaborn as sns
import scipy
from torch import linalg as LA
import itertools
import argparse
from matplotlib import cm

sns.set_style('darkgrid')
colors = sns.color_palette()

class Net(nn.Module):
    def __init__(self, num_layers, input_dim, hidden):
        super(Net, self).__init__()
        self.hidden = hidden
        self.input_layer = nn.Linear(input_dim, hidden)
        self.linear = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(num_layers-1)]
        )
        self.output_layer = nn.Linear(hidden, 3)
        self.relu = nn.ReLU()

    def forward(self, x, act=False):

        activations = []
        out = self.relu(self.input_layer(x))

        if act:
            activations.append(out)

        for layer in range(len(self.linear)):
            out = self.relu(self.linear[layer](out))
            if act:
                activations.append(out)

        if act:
            pattern = torch.cat(activations).squeeze()
            return pattern

        out = self.output_layer(out)

        return out

def set_positive_values_to_one(tensor):
    # Create a mask that is 1 for positive values and 0 for non-positive values
    mask = tensor.gt(0.0)
    # Set all positive values to 1 using the mask
    tensor[mask] = 1
    return tensor

def compute_hamming(model, x_0, x_1):

    # hamming distance
    with torch.no_grad():
        pattern_1 = model(x_0, act=True)
        pattern_1 = set_positive_values_to_one(pattern_1)
        pattern_2 = model(x_1, act=True)
        pattern_2 = set_positive_values_to_one(pattern_2)

    hamming = torch.sum(torch.abs(pattern_1-pattern_2))

    return hamming.item()

def compute_confusion(model, optim, x_0, x_1, y_0, y_1):
    # Get confusion between the gradients for both inputs
    confusion = 0
    criterion = nn.MSELoss()

    # get gradient of first input
    optim.zero_grad()
    output = model(x_0)
    loss = criterion(output, y_0)
    loss.backward()

    # get the gradient
    grad_1 = []
    grad_1.append(torch.flatten(model.input_layer.weight.grad))
    grad_1.append(torch.flatten(model.input_layer.bias.grad))
    for layer in model.linear:
        grad_1.append(torch.flatten(layer.weight.grad))
        grad_1.append(torch.flatten(layer.bias.grad))
    grad_1.append(torch.flatten(model.output_layer.weight.grad))
    grad_1.append(torch.flatten(model.output_layer.bias.grad))
    grad_1 = torch.cat(grad_1)

    # get the gradient of second input
    optim.zero_grad()
    output = model(x_1)
    loss = criterion(output, y_1)
    loss.backward()

    # get gradient
    grad_2 = []
    grad_2.append(torch.flatten(model.input_layer.weight.grad))
    grad_2.append(torch.flatten(model.input_layer.bias.grad))
    for layer in model.linear:
        grad_2.append(torch.flatten(layer.weight.grad))
        grad_2.append(torch.flatten(layer.bias.grad))
    grad_2.append(torch.flatten(model.output_layer.weight.grad))
    grad_2.append(torch.flatten(model.output_layer.bias.grad))
    grad_2 = torch.cat(grad_2)

    # get inner products of gradients
    confusion = torchmetrics.functional.pairwise_cosine_similarity(grad_1.unsqueeze(dim=0), 
                                                                grad_2.unsqueeze(dim=0)).cpu()

    return confusion.item()

def hamming_within_regions(model, optim, inp_batch, inp_target, iterations, comp_confusion=True, comp_hamming=True):

    print('Starting hamming within local regions...')
    hamming_local = []
    confusion_local = []
    # reshape the inputs to be in image format again
    shape = inp_batch.shape
    inp_batch = torch.reshape(inp_batch, [64, 64, shape[-1]])
    inp_target = torch.reshape(inp_target, [64, 64, 3])

    for i in range(iterations):
        # get a random 3x3 patch
        rand_x = np.random.randint(4, 60)
        rand_y = np.random.randint(4, 60)

        patch = inp_batch[rand_x-1:rand_x+2, rand_y-1:rand_y+2, :]
        patch_target = inp_target[rand_x-1:rand_x+2, rand_y-1:rand_y+2, :]

        patch = torch.flatten(patch, start_dim=0, end_dim=1)
        patch_target = torch.flatten(patch_target, start_dim=0, end_dim=1)

        for j in range(9):
            for k in range(j+1, 9):
                if comp_hamming:
                    hamming = compute_hamming(model, patch[j], patch[k])
                    hamming_local.append(hamming)
                if comp_confusion:
                    confusion = compute_confusion(model, optim, patch[j], patch[k], patch_target[j], patch_target[k])
                    confusion_local.append(confusion)

    return hamming_local, confusion_local

def hamming_between_regions(model, optim, inp_batch, inp_target, iterations, comp_confusion=True, comp_hamming=True):

    print('Starting hamming across input space...')
    hamming_between = []
    confusion_between = []

    # reshape the inputs to be in image format again
    shape = inp_batch.shape
    inp_batch = torch.reshape(inp_batch, [64, 64, shape[-1]])
    inp_target = torch.reshape(inp_target, [64, 64, 3])

    for i in range(iterations):

        num = np.random.randint(0, 2)
        # upper left, lower right
        if num == 1:
            rand_x_1 = np.random.randint(0, 30)
            rand_x_2 = np.random.randint(35, 64)

            rand_y_1 = np.random.randint(35, 64)
            rand_y_2 = np.random.randint(0, 30)

        # lower left, upper right
        else:
            rand_x_1 = np.random.randint(0, 30)
            rand_x_2 = np.random.randint(35, 64)

            rand_y_1 = np.random.randint(0, 30)
            rand_y_2 = np.random.randint(35, 64)

        point_1 = inp_batch[rand_x_1, rand_y_1, :]
        point_1_target = inp_target[rand_x_1, rand_y_1, :]

        point_1 = point_1.squeeze()
        point_1_target = point_1_target.squeeze()

        point_2 = inp_batch[rand_x_2, rand_y_2, :]
        point_2_target = inp_target[rand_x_2, rand_y_2, :]

        point_2 = point_2.squeeze()
        point_2_target = point_2_target.squeeze()

        if comp_hamming:
            hamming = compute_hamming(model, point_1, point_2)
            hamming_between.append(hamming)
        if comp_confusion:
            confusion = compute_confusion(model, optim, point_1, point_2, point_1_target, point_2_target)
            confusion_between.append(confusion)

    return hamming_between, confusion_between

def get_data(image, encoding, L=10, batch_size=2048, negative=False, shuffle=True):

    # Get the training image
    trainimg= image
    trainimg = trainimg / 255.0  
    H, W, C = trainimg.shape

    # Get the encoding
    PE = Positional_Encoding(trainimg, encoding, training=True)
    inp_batch, inp_target, ind_vals = PE.get_dataset(L, negative=negative)

    inp_batch, inp_target = torch.Tensor(inp_batch), torch.Tensor(inp_target)
    inp_batch, inp_target = inp_batch.to('cuda:0'), inp_target.to('cuda:0')

    full_batches = []
    batches = []
    batch_targets = []

    random_batches = []
    random_targets = []

    for i in range(H*W):
        full_batches.append((inp_batch[i], inp_target[i]))

    if shuffle:
        random.shuffle(full_batches)
    for i in range(H*W):
        batches.append(full_batches[i][0])
        batch_targets.append(full_batches[i][1])

    batches = torch.stack(batches, dim=0)
    batch_targets = torch.stack(batch_targets, dim=0)
    for i in range(0, H*W, batch_size):
        random_batches.append(batches[i:i+batch_size])
        random_targets.append(batch_targets[i:i+batch_size])

    random_batches = torch.stack(random_batches, dim=0)
    random_targets = torch.stack(random_targets, dim=0)

    return random_batches, random_targets

def train(model, optim, criterion, im, encoding, L, args):

    # Get the training data
    train_inp_batch, train_inp_target = get_data(image=im, encoding=encoding, L=L, batch_size=args.batch_size, negative=args.negative)
    inp_batch, inp_target = get_data(image=im, encoding=encoding, L=L, batch_size=1, shuffle=False, negative=args.negative)

    mean_hamming_within = []
    std_hamming_within = []
    mean_hamming_between = []
    std_hamming_between = []

    losses = []

    # Start the training loop
    for epoch in range(args.epochs):

        running_loss = 0

        # Train the model for one epoch
        for i, pixel in enumerate(train_inp_batch):
            optim.zero_grad()
            output = model(pixel)
            loss = criterion(output, train_inp_target[i])
            running_loss += loss.item()
            loss.backward()
            optim.step()

        # Get and print loss
        epoch_loss = running_loss / (np.shape(im)[0]*np.shape(im)[0] / args.batch_size)

        if args.print_loss:
            print('Loss at epoch {}: {}'.format(epoch, epoch_loss))

        losses.append(epoch_loss)

        # hamming distance and min confusion during training
        # change comp_confusion to true if you want to get the minimum confusion bound
        if args.mean_hamming_in_region:
            if epoch % 100 == 0:
                hamming_in_region, _ = hamming_within_regions(model, optim, inp_batch, inp_target, 100, comp_confusion=False)
                mean_hamming_within.append(np.mean(np.array(hamming_in_region)))
                std_hamming_within.append(np.std(np.array(hamming_in_region)))

        if args.mean_hamming_between_region:
            if epoch % 100 == 0:
                hamming_between_region, _ = hamming_between_regions(model, optim, inp_batch, inp_target, 10000, comp_confusion=False)
                mean_hamming_between.append(np.mean(np.array(hamming_between_region)))
                std_hamming_between.append(np.std(np.array(hamming_between_region)))

        # This now just gets the confusion at the end of training
        if epoch > args.epochs-2:
            _, confusion_in_region = hamming_within_regions(model, optim, inp_batch, inp_target, 100, comp_hamming=False)

        if epoch > args.epochs-2:
            _, confusion_between_region = hamming_between_regions(model, optim, inp_batch, inp_target, 10000, comp_hamming=False)

    return losses, mean_hamming_within, std_hamming_within, mean_hamming_between, std_hamming_between, confusion_in_region, confusion_between_region

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=128, help='Number of neurons per layer')
    parser.add_argument('--layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='make a training and testing set')
    parser.add_argument('--print_loss', type=bool, default=True, help='print training loss')
    parser.add_argument('--mean_hamming_in_region', type=bool, default=True, help='doing line count')
    parser.add_argument('--mean_hamming_between_region', type=bool, default=True, help='doing line count')
    parser.add_argument('--negative', type=bool, default=False, help='-1 to 1')
    parser.add_argument('--encoding', type=str, default='raw_xy', help='raw_xy or sin_cos')
    parser.add_argument('--L', type=int, default=0, help='L value of encoding')

    args = parser.parse_args()

    # compute gradients individually for each, not sure best way to do this yet
    im2arr = np.random.randint(0, 255, (64, 64, 3))

    criterion = nn.MSELoss()

    # Set up raw_xy network
    model_raw = Net(args.layers, args.L*4, args.neurons).to('cuda:0')
    optim_raw = torch.optim.Adam(model_raw.parameters(), lr=.001)

    losses, mean_hamming_within, std_hamming_within, mean_hamming_between, std_hamming_between, confusion_in_region, confusion_between_region = train(model_raw, optim_raw, criterion, im2arr, args.encoding, args.L, args)

if __name__ == '__main__':
    main()

