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
            pattern = torch.cat((act_1, act_2, act_3, act_4), dim=0).squeeze()
            return pattern

        return out_5

def set_positive_values_to_one(tensor):
    # Create a mask that is 1 for positive values and 0 for non-positive values
    mask = tensor.gt(0.0)
    # Set all positive values to 1 using the mask
    tensor[mask] = 1
    return tensor

def get_zero_neurons(model, input):
    # get patterns
    # Change this value to the total number of neurons in the network
    zero_neurons = np.zeros([2048])
    for i, pixel in enumerate(input):
        pixel = pixel.squeeze()
        act_pattern = model(pixel, act=True)
        act_pattern = set_positive_values_to_one(act_pattern)
        act_pattern = act_pattern.detach().cpu().numpy()
        for j in range(act_pattern.shape[0]):
            if act_pattern[j] == 0.:
                zero_neurons[j] += 1
    dead_count = 0
    for item in zero_neurons:
        if item == 512*512:
            dead_count += 1
    return dead_count

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

def train(model, optim, criterion, im, encoding, L, args):

    # Get the training data
    train_inp_batch, train_inp_target = get_data(image=im, encoding=encoding, L=L, batch_size=args.batch_size, negative=args.negative)
    inp_batch, inp_target = get_data(image=im, encoding=encoding, L=L, batch_size=1, shuffle=False, negative=args.negative)

    # lists containing values for various experiments
    param_norms = []
    num_patterns = []

    mean_hamming_within = []
    mean_hamming_between = []

    layer_1_norms = []
    layer_3_norms = []
    layer_2_norms = []

    zero_neuron_plot = []

    losses = []

    # Start the training loop
    for epoch in range(args.epochs):

        dead_neuron_count = 0
        running_loss = 0

        # get the number of dead neurons, start at initialization
        if epoch % 50 == 0:
            with torch.no_grad():
                print('Counting dead ReLU neurons...')
                dead = get_zero_neurons(model, inp_batch)
                print('Number dead neurons at epoch {}: {}'.format(epoch, dead))
                zero_neuron_plot.append(dead)

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

    return zero_neuron_plot

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=512, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8192, help='make a training and testing set')
    parser.add_argument('--print_loss', type=bool, default=True, help='print training loss')
    parser.add_argument('--negative', type=bool, default=False, help='-1 to 1')
    parser.add_argument('--visualize_regions', type=bool, default=False, help='plot the losses')
    parser.add_argument('--train_encoding', action='store_false', default=True, help='train positional encoding')
    parser.add_argument('--train_coordinates', action='store_false', default=True, help='train coordinates')

    args = parser.parse_args()

    # change image to any in image_demonstration
    test_data = np.load('test_data_div2k.npy')
    test_data = test_data[:5]

    L_vals = [4, 8, 16]
    # lists for all confusion for each image
    averaged_local_hamming_xy = []
    averaged_global_hamming_xy = []

    averaged_local_hamming_pe = {}
    averaged_global_hamming_pe = {}

    averaged_local_hamming_pe_std = {}
    averaged_global_hamming_pe_std = {}

    for l in L_vals:
        averaged_local_hamming_pe[f'{l}_val'] = []
        averaged_global_hamming_pe[f'{l}_val'] = []

    # Go through each image individually
    for im in test_data:

        #################################### Raw XY ##############################################

        # Set up raw_xy network
        model_raw = Net(2, args.neurons).to('cuda:0')
        optim_raw = torch.optim.Adam(model_raw.parameters(), lr=.001)
        criterion = nn.MSELoss()

        if args.train_coordinates:
            print("\nBeginning Raw XY Training...")
            xy_zero_neuron_plot = train(model_raw, optim_raw, criterion, im, 'raw_xy', 0, args)
        
        averaged_global_hamming_xy.append(mean_hamming_between_xy)
        averaged_local_hamming_xy.append(mean_hamming_within_xy)

        #################################### Sin Cos #############################################

        if args.train_encoding:
            print("\nBeginning Positional Encoding Training...")
            for l in L_vals:

                # Set up pe network
                model_pe = Net(l*4, args.neurons).to('cuda:0')
                optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)
                criterion = nn.MSELoss()

                pe_zero_neuron_plot = train(model_pe, optim_pe, criterion, im, 'sin_cos', l, args)

                averaged_global_hamming_pe[f'{l}_val'].append(mean_hamming_between_pe)
                averaged_local_hamming_pe[f'{l}_val'].append(mean_hamming_within_pe)
    
    x = np.linspace(0, 1000, 40)

    # Get mean and std across images
    for l in L_vals:

        averaged_global_hamming_pe[f'{l}_val'] = np.array(averaged_global_hamming_pe[f'{l}_val'])
        averaged_global_hamming_pe_std[f'{l}_val'] = np.std(averaged_global_hamming_pe[f'{l}_val'], axis=0)
        averaged_global_hamming_pe[f'{l}_val'] = np.mean(averaged_global_hamming_pe[f'{l}_val'], axis=0)

        averaged_local_hamming_pe[f'{l}_val'] = np.array(averaged_local_hamming_pe[f'{l}_val'])
        averaged_local_hamming_pe_std[f'{l}_val'] = np.std(averaged_local_hamming_pe[f'{l}_val'], axis=0)
        averaged_local_hamming_pe[f'{l}_val'] = np.mean(averaged_local_hamming_pe[f'{l}_val'], axis=0)

    averaged_global_hamming_xy = np.array(averaged_global_hamming_xy)
    averaged_global_hamming_xy_std = np.std(averaged_global_hamming_xy, axis=0)
    averaged_global_hamming_xy = np.mean(averaged_global_hamming_xy, axis=0)

    averaged_local_hamming_xy = np.array(averaged_local_hamming_xy)
    averaged_local_hamming_xy_std = np.std(averaged_local_hamming_xy, axis=0)
    averaged_local_hamming_xy = np.mean(averaged_local_hamming_xy, axis=0)

    # Global Hamming Distances plot

    fig1, ax1 = plt.subplots()
    ax1.plot(x, averaged_global_hamming_pe[f'4_val'], label='L=4', linewidth=2)
    ax1.fill_between(x, np.array(averaged_global_hamming_pe[f'4_val'])+np.array(averaged_global_hamming_pe_std[f'4_val']), np.array(averaged_global_hamming_pe[f'4_val'])-np.array(averaged_global_hamming_pe_std[f'4_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_global_hamming_pe[f'8_val'], label='L=8', linewidth=2)
    ax1.fill_between(x, np.array(averaged_global_hamming_pe[f'8_val'])+np.array(averaged_global_hamming_pe_std[f'8_val']), np.array(averaged_global_hamming_pe[f'8_val'])-np.array(averaged_global_hamming_pe_std[f'8_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_global_hamming_pe[f'16_val'], label='L=16', linewidth=2)
    ax1.fill_between(x, np.array(averaged_global_hamming_pe[f'16_val'])+np.array(averaged_global_hamming_pe_std[f'16_val']), np.array(averaged_global_hamming_pe[f'16_val'])-np.array(averaged_global_hamming_pe_std[f'16_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_global_hamming_xy, label='coordinates', linewidth=2)
    ax1.fill_between(x, np.array(averaged_global_hamming_xy)+np.array(averaged_global_hamming_xy_std), np.array(averaged_global_hamming_xy)-np.array(averaged_global_hamming_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.legend()
    fig1.savefig('hamming_images/hamming_global')

    # Local hamming distances plot

    fig2, ax2 = plt.subplots()
    ax2.plot(x, averaged_local_hamming_pe[f'4_val'], label='L=4', linewidth=2)
    ax2.fill_between(x, np.array(averaged_local_hamming_pe[f'4_val'])+np.array(averaged_local_hamming_pe_std[f'4_val']), np.array(averaged_local_hamming_pe[f'4_val'])-np.array(averaged_local_hamming_pe_std[f'4_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax2.plot(x, averaged_local_hamming_pe[f'8_val'], label='L=8', linewidth=2)
    ax2.fill_between(x, np.array(averaged_local_hamming_pe[f'8_val'])+np.array(averaged_local_hamming_pe_std[f'8_val']), np.array(averaged_local_hamming_pe[f'8_val'])-np.array(averaged_local_hamming_pe_std[f'8_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax2.plot(x, averaged_local_hamming_pe[f'16_val'], label='L=16', linewidth=2)
    ax2.fill_between(x, np.array(averaged_local_hamming_pe[f'16_val'])+np.array(averaged_local_hamming_pe_std[f'16_val']), np.array(averaged_local_hamming_pe[f'16_val'])-np.array(averaged_local_hamming_pe_std[f'16_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax2.plot(x, averaged_local_hamming_xy, label='coordinates', linewidth=2)
    ax2.fill_between(x, np.array(averaged_local_hamming_xy)+np.array(averaged_local_hamming_xy_std), np.array(averaged_local_hamming_xy)-np.array(averaged_local_hamming_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax2.legend()
    fig2.savefig('hamming_images/hamming_local')

if __name__ == '__main__':
    main()

