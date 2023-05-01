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
        out_2 = self.relu(self.l2(out_1))
        out_3 = self.relu(self.l3(out_2))
        out_4 = self.relu(self.l4(out_3))
        out_5 = self.l5(out_4)

        if act:
            return out_1.squeeze(), out_2.squeeze(), out_3.squeeze(), out_4.squeeze()

        return out_5

def binarize(tensor):
    # Create a mask that is 1 for positive values and 0 for non-positive values
    mask = tensor.gt(0.0)
    # Set all positive values to 1 using the mask
    tensor[mask] = 1
    return tensor

def compute_hamming(model, x_0, x_1):

    # hamming distance
    with torch.no_grad():
        l1, l2, l3, l4 = model(x_0, act=True)
        l1, l2, l3, l4 = binarize(l1), binarize(l2), binarize(l3), binarize(l4)
        l1_p2, l2_p2, l3_p2, l4_p2 = model(x_1, act=True)
        l1_p2, l2_p2, l3_p2, l4_p2 = binarize(l1_p2), binarize(l2_p2), binarize(l3_p2), binarize(l4_p2)

    hamming_1 = torch.sum(torch.abs(l1-l1_p2))
    hamming_2 = torch.sum(torch.abs(l2-l2_p2))
    hamming_3 = torch.sum(torch.abs(l3-l3_p2))
    hamming_4 = torch.sum(torch.abs(l4-l4_p2))

    return hamming_1.item(), hamming_2.item(), hamming_3.item(), hamming_4.item()


def sample(model, optim, inp_batch, inp_target, iterations):

    print('Starting hamming across input space...')
    mean_hamming1 = []
    mean_hamming2 = []
    mean_hamming3 = []
    mean_hamming4 = []
    inp_batch = inp_batch.squeeze()

    for i in range(iterations):

        rand_1 = np.random.randint(0, 512*512-1)
        rand_2 = np.random.randint(0, 512*512-1)

        point_1, point_1_target = inp_batch[rand_1], inp_target[rand_1]
        point_1, point_1_target = point_1.squeeze(), point_1_target.squeeze()

        point_2, point_2_target = inp_batch[rand_2], inp_target[rand_2]
        point_2, point_2_target = point_2.squeeze(), point_2_target.squeeze()

        hamming1, hamming2, hamming3, hamming4 = compute_hamming(model, point_1, point_2)
        mean_hamming1.append(hamming1)
        mean_hamming2.append(hamming2)
        mean_hamming3.append(hamming3)
        mean_hamming4.append(hamming4)

    return mean_hamming1, mean_hamming2, mean_hamming3, mean_hamming4

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

    mean_hamming1 = []
    mean_hamming2 = []
    mean_hamming3 = []
    mean_hamming4 = []

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

        # mean hamming distance during training
        if epoch % 20 == 0:
            hamming1, hamming2, hamming3, hamming4 = sample(model, optim, inp_batch, inp_target, 50000)
            mean_hamming1.append(np.mean(np.array(hamming1)))
            mean_hamming2.append(np.mean(np.array(hamming2)))
            mean_hamming3.append(np.mean(np.array(hamming3)))
            mean_hamming4.append(np.mean(np.array(hamming4)))

    return mean_hamming1, mean_hamming2, mean_hamming3, mean_hamming4 

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
    test_data = test_data[:5]

    L_vals = [8]
    # lists for all confusion for each image
    averaged_hamming_l1 = []
    averaged_hamming_l2 = []
    averaged_hamming_l3 = []
    averaged_hamming_l4 = []

    averaged_hamming_pe = {}
    averaged_hamming_pe_std = {}

    for l in L_vals:
        for j in range(4):
            averaged_hamming_pe[f'{l}_layer{j}'] = []

    # Go through each image individually
    for im in test_data:

        #################################### Raw XY ##############################################

        # Set up raw_xy network
        model_raw = Net(2, args.neurons).to('cuda:0')
        optim_raw = torch.optim.Adam(model_raw.parameters(), lr=.001)
        criterion = nn.MSELoss()

        print("\nBeginning Raw XY Training...")
        mean_hamming_l1, mean_hamming_l2, mean_hamming_l3, mean_hamming_l4 = train(model_raw, optim_raw, criterion, im, 'raw_xy', 0, args)
        
        averaged_hamming_l1.append(mean_hamming_l1)
        averaged_hamming_l2.append(mean_hamming_l2)
        averaged_hamming_l3.append(mean_hamming_l3)
        averaged_hamming_l4.append(mean_hamming_l4)

        #################################### Sin Cos #############################################

        print("\nBeginning Positional Encoding Training...")
        for l in L_vals:

            # Set up pe network
            model_pe = Net(l*4, args.neurons).to('cuda:0')
            optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)
            criterion = nn.MSELoss()

            mean_hamming_l1, mean_hamming_l2, mean_hamming_l3, mean_hamming_l4 = train(model_pe, optim_pe, criterion, im, 'sin_cos', l, args)

            averaged_hamming_pe[f'{l}_layer0'].append(mean_hamming_l1)
            averaged_hamming_pe[f'{l}_layer1'].append(mean_hamming_l2)
            averaged_hamming_pe[f'{l}_layer2'].append(mean_hamming_l3)
            averaged_hamming_pe[f'{l}_layer3'].append(mean_hamming_l4)
    
    x = np.linspace(0, 1000, 50)

    # Get mean and std across images
    for l in L_vals:
        for j in range(4):

            averaged_hamming_pe[f'{l}_layer{j}'] = np.array(averaged_hamming_pe[f'{l}_layer{j}'])
            averaged_hamming_pe_std[f'{l}_layer{j}'] = np.std(averaged_hamming_pe[f'{l}_layer{j}'], axis=0)
            averaged_hamming_pe[f'{l}_layer{j}'] = np.mean(averaged_hamming_pe[f'{l}_layer{j}'], axis=0)

    averaged_hamming_l1 = np.array(averaged_hamming_l1)
    averaged_hamming_l1_std = np.std(averaged_hamming_l1, axis=0)
    averaged_hamming_l1 = np.mean(averaged_hamming_l1, axis=0)

    averaged_hamming_l2 = np.array(averaged_hamming_l2)
    averaged_hamming_l2_std = np.std(averaged_hamming_l2, axis=0)
    averaged_hamming_l2 = np.mean(averaged_hamming_l2, axis=0)

    averaged_hamming_l3 = np.array(averaged_hamming_l3)
    averaged_hamming_l3_std = np.std(averaged_hamming_l3, axis=0)
    averaged_hamming_l3 = np.mean(averaged_hamming_l3, axis=0)

    averaged_hamming_l4 = np.array(averaged_hamming_l4)
    averaged_hamming_l4_std = np.std(averaged_hamming_l4, axis=0)
    averaged_hamming_l4 = np.mean(averaged_hamming_l4, axis=0)

    # Plots 
    fig1, ax1 = plt.subplots()
    ax1.errorbar(x, averaged_hamming_pe[f'8_layer0'], yerr=averaged_hamming_pe_std[f'8_layer0'], label='Encoding Layer 1', linewidth=2, fmt='o')
    ax1.errorbar(x, averaged_hamming_pe[f'8_layer1'], yerr=averaged_hamming_pe_std[f'8_layer1'], label='Encoding Layer 2', linewidth=2, fmt='o')
    ax1.errorbar(x, averaged_hamming_pe[f'8_layer2'], yerr=averaged_hamming_pe_std[f'8_layer2'], label='Encoding Layer 3', linewidth=2, fmt='o')
    ax1.errorbar(x, averaged_hamming_pe[f'8_layer3'], yerr=averaged_hamming_pe_std[f'8_layer3'], label='Encoding Layer 4', linewidth=2, fmt='o')

    ax1.errorbar(x, averaged_hamming_l1, yerr=averaged_hamming_l1_std, label='Coordinates Layer 1', linewidth=2, fmt='+')
    ax1.errorbar(x, averaged_hamming_l2, yerr=averaged_hamming_l2_std,label='Coordinates Layer 2', linewidth=2, fmt='+')
    ax1.errorbar(x, averaged_hamming_l3, yerr=averaged_hamming_l3_std,label='Coordinates Layer 3', linewidth=2, fmt='+')
    ax1.errorbar(x, averaged_hamming_l4, yerr=averaged_hamming_l4_std,label='Coordinates Layer 4', linewidth=2, fmt='+')

    ax1.legend()
    fig1.savefig('hamming_images/hamming_layers')

if __name__ == '__main__':
    main()