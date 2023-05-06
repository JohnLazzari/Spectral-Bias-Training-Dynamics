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
    def __init__(self, input_dim, hidden):
        super(Net, self).__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.l4 = nn.Linear(hidden, hidden)
        self.l5 = nn.Linear(hidden, 3)
        self.relu = nn.ReLU()

    def forward(self, x, act=False, first_layer=True):

        out_1 = self.relu(self.l1(x))

        if act:
            act_1 = out_1
            if first_layer:
                return act_1.squeeze()

        out_2 = self.relu(self.l2(out_1))

        if act:
            act_2 = out_2

        out_3 = self.relu(self.l3(out_2))

        if act:
            act_3 = out_3

        out_4 = self.relu(self.l4(out_3))

        if act:
            act_4 = out_4

        out_5 = self.l5(out_4)

        if act:
            pattern = torch.cat((act_1, act_2, act_3, act_4), dim=1).squeeze()
            return pattern

        return out_5

def set_positive_values_to_one(tensor):
    tensor = torch.where(tensor > 0.0, torch.ones_like(tensor), tensor)
    return tensor

def get_activation_regions(model, input, L, slice, first_layer=False):

    cross_section = torch.zeros(512*512, L*4-2).to('cuda:0')
    input = input.squeeze()
    if slice == 'high':
        input = torch.cat((cross_section, input), dim=-1)
    elif slice == 'low':
        input = torch.cat((input, cross_section), dim=-1)

    with torch.no_grad():
        # Pass all pixels to the model at once
        batch_size = input.shape[0]
        activations = model(input.view(batch_size, -1), act=True, first_layer=first_layer)
        
        # Set positive values to one
        activations = set_positive_values_to_one(activations)
        
        # Count the number of unique patterns
        unique_patterns = torch.unique(activations, dim=0)
        num_unique_patterns = unique_patterns.shape[0]
        
        return num_unique_patterns, unique_patterns.cpu().numpy().tolist(), activations.cpu().numpy().tolist()

def plot_patterns(patterns, all_patterns):

    dict_patterns = {}
    colors = np.zeros(512*512)
    random.shuffle(patterns)
    for i, pattern in enumerate(patterns):
        # assign each position a color
        dict_patterns[tuple(pattern)] = i
    for i, pattern in enumerate(all_patterns):
        colors[i] = dict_patterns[tuple(pattern)]
    
    colors = np.reshape(colors, [512, 512])
    plt.pcolormesh(colors, cmap='Spectral')
    plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
    plt.show()

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

    # create batches to track batch loss and show it is more stable due to gabor encoding
    full_batches = []
    batches = []
    batch_targets = []

    random_batches = []
    random_targets = []
    # make sure to choose one which can be evenly divided by 65536
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

def train(model, optim, criterion, im, encoding, args):

    # Get the training data
    train_inp_batch, train_inp_target = get_data(image=im, encoding=encoding, L=args.L, batch_size=args.batch_size)
    # data for visualization, should be raw_xy since it is 2D slice
    inp_batch, inp_target = get_data(image=im, encoding='raw_xy', L=0, batch_size=1, shuffle=False, negative=True)

    losses = []
    num_patterns = []

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
        epoch_loss = running_loss / (64*64 / args.batch_size)

        if args.print_loss:
            print('Loss at epoch {}: {}'.format(epoch, epoch_loss))

        losses.append(epoch_loss)

    print('Counting activation patterns...')
    raw_regions, unique_patterns, all_patterns = get_activation_regions(model, inp_batch, args.L, args.frequency_slice, args.first_layer)
    print('number of unique activation regions raw_xy: {}'.format(raw_regions))
    num_patterns.append(raw_regions)
    plot_patterns(unique_patterns, all_patterns)

    return num_patterns

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=512, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=2500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8192, help='make a training and testing set')
    parser.add_argument('--print_loss', type=bool, default=True, help='print training loss')
    parser.add_argument('--L', type=int, default=4, help='Encoding frequency')
    parser.add_argument('--frequency_slice', type=str, default='low', help='2D slice based on where high and low frequency components are')
    parser.add_argument('--first_layer', type=bool, default=False, help='activation patterns in first layer only')
    args = parser.parse_args()

    # change image to any in image_demonstration
    test_data = np.load('test_data_div2k.npy')
    test_data = test_data[0]

    criterion = nn.MSELoss()

    # Set up pe network
    model_pe = Net(4*args.L, args.neurons).to('cuda:0')
    optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)
    criterion = nn.MSELoss()

    #################################### Sin Cos #############################################
    print("\nBeginning Positional Encoding Training...")
    num_patterns = train(model_pe, optim_pe, criterion, test_data, 'sin_cos', args)

if __name__ == '__main__':
    main()