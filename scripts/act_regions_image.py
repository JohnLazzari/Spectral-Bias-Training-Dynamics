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

    def forward(self, x, act=False):

        out_1 = self.relu(self.l1(x))

        if act:
            act_1 = out_1

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

def get_activation_regions(model, input):
    with torch.no_grad():
        # Pass all pixels to the model at once
        batch_size = input.shape[0]
        activations = model(input.view(batch_size, -1), act=True)
        
        # Set positive values to one
        activations = set_positive_values_to_one(activations)
        
        # Count the number of unique patterns
        unique_patterns = torch.unique(activations, dim=0)
        num_unique_patterns = unique_patterns.shape[0]
        
        return num_unique_patterns, unique_patterns.cpu().numpy().tolist(), activations.cpu().numpy().tolist()
    
def plot_patterns(patterns, all_patterns):

    dict_patterns = {}
    # only plotting for inputs, not for counting regions
    colors = np.zeros(512*512)
    random.shuffle(patterns)
    for i, pattern in enumerate(patterns):
        # assign each position a color
        dict_patterns[tuple(pattern)] = i
    for i, pattern in enumerate(all_patterns):
        colors[i] = dict_patterns[tuple(pattern)]
    
    colors = np.reshape(colors, [512, 512])
    plt.pcolormesh(colors, cmap='Spectral')
    plt.tick_params(left = False, right = False , top = False, labelleft = False ,
                labelbottom = False, labeltop=False, bottom = False)
    plt.colorbar()
    plt.show()

def plot_first_3_neurons(all_patterns):

    all_patterns = np.array(all_patterns)
    all_patterns = np.reshape(all_patterns, [512, 512, -1])

    # get neurons that arent all dead or activated to show hyperplane
    neuron_list = []

    # get random neurons to demonstrate hyperplane arrangement
    first_neuron = all_patterns[:, :, 0]
    second_neuron = all_patterns[:, :, 9]
    third_neuron = all_patterns[:, :, 100]

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    pcm1 = ax1.pcolormesh(first_neuron, cmap='Spectral')
    pcm2 = ax2.pcolormesh(second_neuron, cmap='Spectral')
    pcm3 = ax3.pcolormesh(third_neuron, cmap='Spectral')

    ax1.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
    ax2.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
    ax3.tick_params(left = False, right = False , labelleft = False ,
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

def train(model, optim, criterion, im, encoding, L, args, negative=False):

    # Get the training data
    train_inp_batch, train_inp_target = get_data(image=im, encoding=encoding, L=L, batch_size=args.batch_size, negative=negative)
    inp_batch, inp_target = get_data(image=im, encoding=encoding, L=L, batch_size=1, shuffle=False, negative=negative)

    # lists containing values for various experiments
    num_patterns = []

    losses = []

    # Start the training loop
    for epoch in range(args.epochs):

        running_loss = 0

        # Train the model for one epoch
        for i, pixel in enumerate(train_inp_batch):
            for param in model.parameters():
                param.grad = None
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

        # Get activation region count
        if epoch % 50 == 0:
            print('Counting activation patterns...')
            raw_regions, unique_patterns, all_patterns = get_activation_regions(model, inp_batch)
            print('number of unique activation regions: {}'.format(raw_regions))
            num_patterns.append(raw_regions)

    return num_patterns, losses

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=512, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8192, help='make a training and testing set')
    parser.add_argument('--print_loss', type=bool, default=True, help='print training loss')
    parser.add_argument('--train_encoding', action='store_false', default=True, help='train positional encoding')
    parser.add_argument('--train_coordinates', action='store_false', default=True, help='train coordinates')

    args = parser.parse_args()

    # change image to any in image_demonstration
    test_data = np.load('test_data_div2k.npy')
    test_data = test_data[:3]

    criterion = nn.MSELoss()

    L_vals = [1, 2, 8]

    # lists for all confusion for each image
    averaged_patterns_xy = []
    averaged_patterns_neg_xy = []
    averaged_patterns_pe = {}
    averaged_patterns_pe_std = {}

    for l in L_vals:
        averaged_patterns_pe[f'{l}_val'] = []
        averaged_patterns_pe_std[f'{l}_val'] = []

    # Go through each image individually
    for im in test_data:

        #################################### Raw XY ##############################################

        # Set up raw_xy network
        model_raw = Net(2, args.neurons).to('cuda:0')
        optim_raw = torch.optim.Adam(model_raw.parameters(), lr=.001)

        if args.train_coordinates:
            print("\nBeginning Raw XY Training...")
            xy_num_patterns, xy_losses = train(model_raw, optim_raw, criterion, im, 'raw_xy', 0, args)

        averaged_patterns_xy.append(xy_num_patterns)

        ################################## larger normalizing interval ###########################

        model_neg = Net(2, args.neurons).to('cuda:0')
        optim_neg = torch.optim.Adam(model_neg.parameters(), lr=.001)

        print("\nBeginning Neg Raw XY Training...")
        neg_num_patterns, neg_losses = train(model_neg, optim_neg, criterion, im, 'raw_xy', 0, args, negative=True)

        averaged_patterns_neg_xy.append(neg_num_patterns)
        
        #################################### Sin Cos #############################################

        if args.train_encoding:

            for l in L_vals:
                print("\nBeginning Positional Encoding Training...")
                # Set up pe network
                model_pe = Net(l*4, args.neurons).to('cuda:0')
                optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)
                criterion = nn.MSELoss()

                pe_num_patterns, pe_losses = train(model_pe, optim_pe, criterion, im, 'sin_cos', l, args)

                averaged_patterns_pe[f'{l}_val'].append(pe_num_patterns)

    x = np.linspace(0, 1000, 20)

    for l in L_vals:

        averaged_patterns_pe[f'{l}_val'] = np.array(averaged_patterns_pe[f'{l}_val'])
        averaged_patterns_pe_std[f'{l}_val'] = np.std(averaged_patterns_pe[f'{l}_val'], axis=0)
        averaged_patterns_pe[f'{l}_val'] = np.mean(averaged_patterns_pe[f'{l}_val'], axis=0)

    averaged_patterns_xy = np.array(averaged_patterns_xy)
    averaged_patterns_xy_std = np.std(averaged_patterns_xy, axis=0)
    averaged_patterns_xy = np.mean(averaged_patterns_xy, axis=0)

    averaged_patterns_neg_xy = np.array(averaged_patterns_neg_xy)
    averaged_patterns_neg_xy_std = np.std(averaged_patterns_neg_xy, axis=0)
    averaged_patterns_neg_xy = np.mean(averaged_patterns_neg_xy, axis=0)

    fig1, ax1 = plt.subplots()
    ax1.plot(x, averaged_patterns_pe[f'1_val'], label='Encoding L=1', linewidth=2)
    ax1.fill_between(x, np.array(averaged_patterns_pe[f'1_val'])+np.array(averaged_patterns_pe_std[f'1_val']), np.array(averaged_patterns_pe[f'1_val'])-np.array(averaged_patterns_pe_std[f'1_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_patterns_pe[f'2_val'], label='Encoding L=2', linewidth=2)
    ax1.fill_between(x, np.array(averaged_patterns_pe[f'2_val'])+np.array(averaged_patterns_pe_std[f'2_val']), np.array(averaged_patterns_pe[f'2_val'])-np.array(averaged_patterns_pe_std[f'2_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_patterns_pe[f'8_val'], label='Encoding L=8', linewidth=2)
    ax1.fill_between(x, np.array(averaged_patterns_pe[f'8_val'])+np.array(averaged_patterns_pe_std[f'8_val']), np.array(averaged_patterns_pe[f'8_val'])-np.array(averaged_patterns_pe_std[f'8_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_patterns_xy, label='Coordinates [0,1]', linewidth=2)
    ax1.fill_between(x, np.array(averaged_patterns_xy)+np.array(averaged_patterns_xy_std), np.array(averaged_patterns_xy)-np.array(averaged_patterns_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_patterns_neg_xy, label='Coordinates [-1,1]', linewidth=2)
    ax1.fill_between(x, np.array(averaged_patterns_neg_xy)+np.array(averaged_patterns_neg_xy_std), np.array(averaged_patterns_neg_xy)-np.array(averaged_patterns_neg_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Num Activation Regions")
    fig1.savefig('act_region_growth/act_pattern_growth')

if __name__ == '__main__':
    main()

