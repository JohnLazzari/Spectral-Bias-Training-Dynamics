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

def get_data(image, encoding, L=10, batch_size=2048, negative=False, shuffle=True):

    # Get the training image
    trainimg= image
    trainimg = trainimg / 255.0  
    H, W, C = trainimg.shape

    # Get the encoding
    PE = Positional_Encoding(trainimg, encoding, training=True)
    inp_batch, inp_target, ind_vals = PE.get_dataset(L, negative=negative)

    inp_batch, inp_target = torch.Tensor(inp_batch), torch.Tensor(inp_target)
    inp_batch, inp_target = inp_batch.to('cuda:1'), inp_target.to('cuda:1')

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

    # lists containing values for various experiments
    param_norms = []

    layer_1_norms = []
    layer_2_norms = []
    layer_3_norms = []
    layer_4_norms = []
    layer_5_norms = []

    losses = []

    # Start the training loop
    for epoch in range(args.epochs):

        dead_neuron_count = 0
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

        # Calculate the spectral norm of the weight matrices for param norms
        with torch.no_grad():
            # Go through each layer and multiply the norms
            U, S, V = torch.linalg.svd(model.l1.weight)
            norm_1 = max(S)
            layer_1_norms.append(norm_1.item())

            U, S, V = torch.linalg.svd(model.l2.weight)
            norm_2 = max(S)
            layer_2_norms.append(norm_2.item())

            U, S, V = torch.linalg.svd(model.l3.weight)
            norm_3 = max(S)
            layer_3_norms.append(norm_3.item())

            U, S, V = torch.linalg.svd(model.l4.weight)
            norm_4 = max(S)
            layer_4_norms.append(norm_4.item())

            U, S, V = torch.linalg.svd(model.l5.weight)
            norm_5 = max(S)
            layer_5_norms.append(norm_5.item())

            total_norm = norm_1 * norm_2 * norm_3 * norm_4 * norm_5
            param_norms.append(total_norm.item())

    return param_norms, layer_1_norms, layer_2_norms, layer_3_norms, layer_4_norms, layer_5_norms

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

    L_vals = [4, 8]
    # lists for all confusion for each image
    averaged_param_norms_xy = []
    averaged_param_norms_layer1_xy = []
    averaged_param_norms_layer2_xy = []
    averaged_param_norms_layer3_xy = []
    averaged_param_norms_layer4_xy = []
    averaged_param_norms_layer5_xy = []

    averaged_param_norms_neg_xy = []
    averaged_param_norms_layer1_neg_xy = []
    averaged_param_norms_layer2_neg_xy = []
    averaged_param_norms_layer3_neg_xy = []
    averaged_param_norms_layer4_neg_xy = []
    averaged_param_norms_layer5_neg_xy = []

    averaged_param_norms_pe = {}
    averaged_param_norms_layer1_pe = {}
    averaged_param_norms_layer2_pe = {}
    averaged_param_norms_layer3_pe = {}
    averaged_param_norms_layer4_pe = {}
    averaged_param_norms_layer5_pe = {}

    averaged_param_norms_pe_std = {}
    averaged_param_norms_layer1_pe_std = {}
    averaged_param_norms_layer2_pe_std = {}
    averaged_param_norms_layer3_pe_std = {}
    averaged_param_norms_layer4_pe_std = {}
    averaged_param_norms_layer5_pe_std = {}

    for l in L_vals:

        averaged_param_norms_pe[f'{l}_val'] = []
        averaged_param_norms_layer1_pe[f'{l}_val'] = []
        averaged_param_norms_layer2_pe[f'{l}_val'] = []
        averaged_param_norms_layer3_pe[f'{l}_val'] = []
        averaged_param_norms_layer4_pe[f'{l}_val'] = []
        averaged_param_norms_layer5_pe[f'{l}_val'] = []

        averaged_param_norms_pe_std[f'{l}_val'] = []
        averaged_param_norms_layer1_pe_std[f'{l}_val'] = []
        averaged_param_norms_layer2_pe_std[f'{l}_val'] = []
        averaged_param_norms_layer3_pe_std[f'{l}_val'] = []
        averaged_param_norms_layer4_pe_std[f'{l}_val'] = []
        averaged_param_norms_layer5_pe_std[f'{l}_val'] = []

    # Go through each image individually
    for im in test_data:

        #################################### Raw XY ##############################################

        # Set up raw_xy network
        model_raw = Net(2, args.neurons).to('cuda:1')
        optim_raw = torch.optim.Adam(model_raw.parameters(), lr=.001)
        criterion = nn.MSELoss()

        if args.train_coordinates:
            print("\nBeginning Raw XY Training...")
            xy_param_norms, xy_layer_1_norms, xy_layer_2_norms, xy_layer_3_norms, xy_layer_4_norms, xy_layer_5_norms = train(model_raw, optim_raw, criterion, im, 'raw_xy', 0, args)
        
        averaged_param_norms_xy.append(xy_param_norms)
        averaged_param_norms_layer1_xy.append(xy_layer_1_norms)
        averaged_param_norms_layer2_xy.append(xy_layer_2_norms)
        averaged_param_norms_layer3_xy.append(xy_layer_3_norms)
        averaged_param_norms_layer4_xy.append(xy_layer_4_norms)
        averaged_param_norms_layer5_xy.append(xy_layer_5_norms)

        ################################## larger normalizing interval ###########################

        model_neg = Net(2, args.neurons).to('cuda:1')
        optim_neg = torch.optim.Adam(model_neg.parameters(), lr=.001)

        print("\nBeginning Neg Raw XY Training...")
        neg_param_norms, neg_layer_1_norms, neg_layer_2_norms, neg_layer_3_norms, neg_layer_4_norms, neg_layer_5_norms = train(model_neg, optim_neg, criterion, im, 'raw_xy', 0, args, negative=True)
        
        averaged_param_norms_neg_xy.append(neg_param_norms)
        averaged_param_norms_layer1_neg_xy.append(neg_layer_1_norms)
        averaged_param_norms_layer2_neg_xy.append(neg_layer_2_norms)
        averaged_param_norms_layer3_neg_xy.append(neg_layer_3_norms)
        averaged_param_norms_layer4_neg_xy.append(neg_layer_4_norms)
        averaged_param_norms_layer5_neg_xy.append(neg_layer_5_norms)

        #################################### Sin Cos #############################################

        if args.train_encoding:
            print("\nBeginning Positional Encoding Training...")
            for l in L_vals:

                # Set up pe network
                model_pe = Net(l*4, args.neurons).to('cuda:1')
                optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)
                criterion = nn.MSELoss()

                pe_param_norms, pe_layer_1_norms, pe_layer_2_norms, pe_layer_3_norms, pe_layer_4_norms, pe_layer_5_norms = train(model_pe, optim_pe, criterion, im, 'sin_cos', l, args)

                averaged_param_norms_pe[f'{l}_val'].append(pe_param_norms)
                averaged_param_norms_layer1_pe[f'{l}_val'].append(pe_layer_1_norms)
                averaged_param_norms_layer2_pe[f'{l}_val'].append(pe_layer_2_norms)
                averaged_param_norms_layer3_pe[f'{l}_val'].append(pe_layer_3_norms)
                averaged_param_norms_layer4_pe[f'{l}_val'].append(pe_layer_4_norms)
                averaged_param_norms_layer5_pe[f'{l}_val'].append(pe_layer_5_norms)
    
    x = np.linspace(0, 1000, 1000)

    # Get mean and std across images
    for l in L_vals:

        averaged_param_norms_pe[f'{l}_val'] = np.array(averaged_param_norms_pe[f'{l}_val'])
        averaged_param_norms_pe_std[f'{l}_val'] = np.std(averaged_param_norms_pe[f'{l}_val'], axis=0)
        averaged_param_norms_pe[f'{l}_val'] = np.mean(averaged_param_norms_pe[f'{l}_val'], axis=0)

        averaged_param_norms_layer1_pe[f'{l}_val'] = np.array(averaged_param_norms_layer1_pe[f'{l}_val'])
        averaged_param_norms_layer1_pe_std[f'{l}_val'] = np.std(averaged_param_norms_layer1_pe[f'{l}_val'], axis=0)
        averaged_param_norms_layer1_pe[f'{l}_val'] = np.mean(averaged_param_norms_layer1_pe[f'{l}_val'], axis=0)

        averaged_param_norms_layer2_pe[f'{l}_val'] = np.array(averaged_param_norms_layer2_pe[f'{l}_val'])
        averaged_param_norms_layer2_pe_std[f'{l}_val'] = np.std(averaged_param_norms_layer2_pe[f'{l}_val'], axis=0)
        averaged_param_norms_layer2_pe[f'{l}_val'] = np.mean(averaged_param_norms_layer2_pe[f'{l}_val'], axis=0)

        averaged_param_norms_layer3_pe[f'{l}_val'] = np.array(averaged_param_norms_layer3_pe[f'{l}_val'])
        averaged_param_norms_layer3_pe_std[f'{l}_val'] = np.std(averaged_param_norms_layer3_pe[f'{l}_val'], axis=0)
        averaged_param_norms_layer3_pe[f'{l}_val'] = np.mean(averaged_param_norms_layer3_pe[f'{l}_val'], axis=0)

        averaged_param_norms_layer4_pe[f'{l}_val'] = np.array(averaged_param_norms_layer4_pe[f'{l}_val'])
        averaged_param_norms_layer4_pe_std[f'{l}_val'] = np.std(averaged_param_norms_layer4_pe[f'{l}_val'], axis=0)
        averaged_param_norms_layer4_pe[f'{l}_val'] = np.mean(averaged_param_norms_layer4_pe[f'{l}_val'], axis=0)

        averaged_param_norms_layer5_pe[f'{l}_val'] = np.array(averaged_param_norms_layer5_pe[f'{l}_val'])
        averaged_param_norms_layer5_pe_std[f'{l}_val'] = np.std(averaged_param_norms_layer5_pe[f'{l}_val'], axis=0)
        averaged_param_norms_layer5_pe[f'{l}_val'] = np.mean(averaged_param_norms_layer5_pe[f'{l}_val'], axis=0)

    # Coordinates
    averaged_param_norms_xy = np.array(averaged_param_norms_xy)
    averaged_param_norms_xy_std = np.std(averaged_param_norms_xy, axis=0)
    averaged_param_norms_xy = np.mean(averaged_param_norms_xy, axis=0)

    averaged_param_norms_layer1_xy = np.array(averaged_param_norms_layer1_xy)
    averaged_param_norms_layer1_xy_std = np.std(averaged_param_norms_layer1_xy, axis=0)
    averaged_param_norms_layer1_xy = np.mean(averaged_param_norms_layer1_xy, axis=0)

    averaged_param_norms_layer2_xy = np.array(averaged_param_norms_layer2_xy)
    averaged_param_norms_layer2_xy_std = np.std(averaged_param_norms_layer2_xy, axis=0)
    averaged_param_norms_layer2_xy = np.mean(averaged_param_norms_layer2_xy, axis=0)

    averaged_param_norms_layer3_xy = np.array(averaged_param_norms_layer3_xy)
    averaged_param_norms_layer3_xy_std = np.std(averaged_param_norms_layer3_xy, axis=0)
    averaged_param_norms_layer3_xy = np.mean(averaged_param_norms_layer3_xy, axis=0)

    averaged_param_norms_layer4_xy = np.array(averaged_param_norms_layer4_xy)
    averaged_param_norms_layer4_xy_std = np.std(averaged_param_norms_layer4_xy, axis=0)
    averaged_param_norms_layer4_xy = np.mean(averaged_param_norms_layer4_xy, axis=0)

    averaged_param_norms_layer5_xy = np.array(averaged_param_norms_layer5_xy)
    averaged_param_norms_layer5_xy_std = np.std(averaged_param_norms_layer5_xy, axis=0)
    averaged_param_norms_layer5_xy = np.mean(averaged_param_norms_layer5_xy, axis=0)

    # Neg Coordinates
    averaged_param_norms_neg_xy = np.array(averaged_param_norms_neg_xy)
    averaged_param_norms_neg_xy_std = np.std(averaged_param_norms_neg_xy, axis=0)
    averaged_param_norms_neg_xy = np.mean(averaged_param_norms_neg_xy, axis=0)

    averaged_param_norms_layer1_neg_xy = np.array(averaged_param_norms_layer1_neg_xy)
    averaged_param_norms_layer1_neg_xy_std = np.std(averaged_param_norms_layer1_neg_xy, axis=0)
    averaged_param_norms_layer1_neg_xy = np.mean(averaged_param_norms_layer1_neg_xy, axis=0)

    averaged_param_norms_layer2_neg_xy = np.array(averaged_param_norms_layer2_neg_xy)
    averaged_param_norms_layer2_neg_xy_std = np.std(averaged_param_norms_layer2_neg_xy, axis=0)
    averaged_param_norms_layer2_neg_xy = np.mean(averaged_param_norms_layer2_neg_xy, axis=0)

    averaged_param_norms_layer3_neg_xy = np.array(averaged_param_norms_layer3_neg_xy)
    averaged_param_norms_layer3_neg_xy_std = np.std(averaged_param_norms_layer3_neg_xy, axis=0)
    averaged_param_norms_layer3_neg_xy = np.mean(averaged_param_norms_layer3_neg_xy, axis=0)

    averaged_param_norms_layer4_neg_xy = np.array(averaged_param_norms_layer4_neg_xy)
    averaged_param_norms_layer4_neg_xy_std = np.std(averaged_param_norms_layer4_neg_xy, axis=0)
    averaged_param_norms_layer4_neg_xy = np.mean(averaged_param_norms_layer4_neg_xy, axis=0)

    averaged_param_norms_layer5_neg_xy = np.array(averaged_param_norms_layer5_neg_xy)
    averaged_param_norms_layer5_neg_xy_std = np.std(averaged_param_norms_layer5_neg_xy, axis=0)
    averaged_param_norms_layer5_neg_xy = np.mean(averaged_param_norms_layer5_neg_xy, axis=0)

    # Encoding L=8 Layers

    fig1, ax1 = plt.subplots()

    ax1.plot(x, averaged_param_norms_layer1_pe[f'8_val'], label='Layer 1', linewidth=2)
    ax1.fill_between(x, np.array(averaged_param_norms_layer1_pe[f'8_val'])+np.array(averaged_param_norms_layer1_pe_std[f'8_val']), np.array(averaged_param_norms_layer1_pe[f'8_val'])-np.array(averaged_param_norms_layer1_pe_std[f'8_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_param_norms_layer2_pe[f'8_val'], label='Layer 2', linewidth=2)
    ax1.fill_between(x, np.array(averaged_param_norms_layer2_pe[f'8_val'])+np.array(averaged_param_norms_layer2_pe_std[f'8_val']), np.array(averaged_param_norms_layer2_pe[f'8_val'])-np.array(averaged_param_norms_layer2_pe_std[f'8_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_param_norms_layer3_pe[f'8_val'], label='Layer 3', linewidth=2)
    ax1.fill_between(x, np.array(averaged_param_norms_layer3_pe[f'8_val'])+np.array(averaged_param_norms_layer3_pe_std[f'8_val']), np.array(averaged_param_norms_layer3_pe[f'8_val'])-np.array(averaged_param_norms_layer3_pe_std[f'8_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_param_norms_layer4_pe[f'8_val'], label='Layer 4', linewidth=2)
    ax1.fill_between(x, np.array(averaged_param_norms_layer4_pe[f'8_val'])+np.array(averaged_param_norms_layer4_pe_std[f'8_val']), np.array(averaged_param_norms_layer4_pe[f'8_val'])-np.array(averaged_param_norms_layer4_pe_std[f'8_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_param_norms_layer5_pe[f'8_val'], label='Layer 5', linewidth=2)
    ax1.fill_between(x, np.array(averaged_param_norms_layer5_pe[f'8_val'])+np.array(averaged_param_norms_layer5_pe_std[f'8_val']), np.array(averaged_param_norms_layer5_pe[f'8_val'])-np.array(averaged_param_norms_layer5_pe_std[f'8_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.legend()
    fig1.savefig('param_norms_image/encoding_layer_norms')

    # Coordinates [0,1] Layers

    fig2, ax2 = plt.subplots()

    ax2.plot(x, averaged_param_norms_layer1_xy, label='Layer 1', linewidth=2)
    ax2.fill_between(x, np.array(averaged_param_norms_layer1_xy)+np.array(averaged_param_norms_layer1_xy_std), np.array(averaged_param_norms_layer1_xy)-np.array(averaged_param_norms_layer1_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax2.plot(x, averaged_param_norms_layer2_xy, label='Layer 2', linewidth=2)
    ax2.fill_between(x, np.array(averaged_param_norms_layer2_xy)+np.array(averaged_param_norms_layer2_xy_std), np.array(averaged_param_norms_layer2_xy)-np.array(averaged_param_norms_layer2_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax2.plot(x, averaged_param_norms_layer3_xy, label='Layer 3', linewidth=2)
    ax2.fill_between(x, np.array(averaged_param_norms_layer3_xy)+np.array(averaged_param_norms_layer3_xy_std), np.array(averaged_param_norms_layer3_xy)-np.array(averaged_param_norms_layer3_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax2.plot(x, averaged_param_norms_layer4_xy, label='Layer 4', linewidth=2)
    ax2.fill_between(x, np.array(averaged_param_norms_layer4_xy)+np.array(averaged_param_norms_layer4_xy_std), np.array(averaged_param_norms_layer4_xy)-np.array(averaged_param_norms_layer4_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax2.plot(x, averaged_param_norms_layer5_xy, label='Layer 5', linewidth=2)
    ax2.fill_between(x, np.array(averaged_param_norms_layer5_xy)+np.array(averaged_param_norms_layer5_xy_std), np.array(averaged_param_norms_layer5_xy)-np.array(averaged_param_norms_layer5_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax2.legend()
    fig2.savefig('param_norms_image/coordinate_layer_norms')

    # Total Param Norms

    fig3, ax3 = plt.subplots()

    ax3.plot(x, averaged_param_norms_pe[f'4_val'], label='Encoding L=4', linewidth=2)
    ax3.fill_between(x, np.array(averaged_param_norms_pe[f'4_val'])+np.array(averaged_param_norms_pe_std[f'4_val']), np.array(averaged_param_norms_pe[f'4_val'])-np.array(averaged_param_norms_pe_std[f'4_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax3.plot(x, averaged_param_norms_pe[f'8_val'], label='Encoding L=8', linewidth=2)
    ax3.fill_between(x, np.array(averaged_param_norms_pe[f'8_val'])+np.array(averaged_param_norms_pe_std[f'8_val']), np.array(averaged_param_norms_pe[f'8_val'])-np.array(averaged_param_norms_pe_std[f'8_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax3.plot(x, averaged_param_norms_xy, label='Coordinates [0,1]', linewidth=2)
    ax3.fill_between(x, np.array(averaged_param_norms_xy)+np.array(averaged_param_norms_xy_std), np.array(averaged_param_norms_xy)-np.array(averaged_param_norms_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax3.plot(x, averaged_param_norms_neg_xy, label='Coordinates [-1,1]', linewidth=2)
    ax3.fill_between(x, np.array(averaged_param_norms_neg_xy)+np.array(averaged_param_norms_neg_xy_std), np.array(averaged_param_norms_neg_xy)-np.array(averaged_param_norms_neg_xy_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax3.legend()
    fig3.savefig('param_norms_image/overall_param_norms')

if __name__ == '__main__':
    main()

