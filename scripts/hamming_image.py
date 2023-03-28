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

    for param in model.parameters():
        param.grad = None

    output = model(x_0)
    loss_1 = criterion(output, y_0)
    loss_1.backward()

    grad_1 = torch.cat([torch.flatten(model.l1.weight.grad), torch.flatten(model.l2.weight.grad), torch.flatten(model.l3.weight.grad), torch.flatten(model.l4.weight.grad), torch.flatten(model.l5.weight.grad),
                        torch.flatten(model.l1.bias.grad), torch.flatten(model.l2.bias.grad), torch.flatten(model.l3.bias.grad), torch.flatten(model.l4.bias.grad), torch.flatten(model.l5.bias.grad),
                        ])

    for param in model.parameters():
        param.grad = None

    output = model(x_1)
    loss_2 = criterion(output, y_1)
    loss_2.backward()
    grad_2 = torch.cat([torch.flatten(model.l1.weight.grad), torch.flatten(model.l2.weight.grad), torch.flatten(model.l3.weight.grad), torch.flatten(model.l4.weight.grad), torch.flatten(model.l5.weight.grad),
                        torch.flatten(model.l1.bias.grad), torch.flatten(model.l2.bias.grad), torch.flatten(model.l3.bias.grad), torch.flatten(model.l4.bias.grad), torch.flatten(model.l5.bias.grad),
                        ])

    # get inner products of gradients
    confusion = torchmetrics.functional.pairwise_cosine_similarity(grad_1.unsqueeze(dim=0), 
                                                                   grad_2.unsqueeze(dim=0)).cpu()

    for param in model.parameters():
        param.grad = None

    return confusion.item()

def hamming_within_regions(model, optim, inp_batch, inp_target, iterations, comp_hamming=True, comp_confusion=True):

    print('Starting hamming within local regions...')
    hamming_local = []
    confusion_local = []
    # reshape the inputs to be in image format again
    shape = inp_batch.shape
    inp_batch = torch.reshape(inp_batch, [512, 512, shape[-1]])
    inp_target = torch.reshape(inp_target, [512, 512, 3])

    # do line count between all inputs in region
    for i in range(iterations):
        # get a random 20x20 patch
        rand_x = np.random.randint(30, 480)
        rand_y = np.random.randint(30, 480)

        patch = torch.flatten(inp_batch[rand_x-25:rand_x+25, rand_y-25:rand_y+25, :], start_dim=0, end_dim=1)
        patch_target = torch.flatten(inp_target[rand_x-25:rand_x+25, rand_y-25:rand_y+25, :], start_dim=0, end_dim=1)

        for j in range(250):

            ind1 = np.random.randint(0, 2499)
            ind2 = np.random.randint(0, 2499)

            if comp_hamming:
                hamming = compute_hamming(model, patch[ind1], patch[ind2])
                hamming_local.append(hamming)
            if comp_confusion:
                confusion = compute_confusion(model, optim, patch[ind1], patch[ind2], patch_target[ind1], patch_target[ind2])
                confusion_local.append(confusion)

    return hamming_local, confusion_local

def hamming_between_regions(model, optim, inp_batch, inp_target, iterations, comp_hamming=True, comp_confusion=True):

    print('Starting hamming across input space...')
    hamming_between = []
    confusion_between = []
    # reshape the inputs to be in image format again
    shape = inp_batch.shape
    loss = []
    inp_batch = torch.reshape(inp_batch, [512, 512, shape[-1]])
    inp_target = torch.reshape(inp_target, [512, 512, 3])

    for i in range(iterations):

        num = np.random.randint(0, 2)
        if num == 1:
            rand_x_1 = np.random.randint(0, 225)
            rand_y_1 = np.random.randint(275, 511)

            rand_x_2 = np.random.randint(275, 511)
            rand_y_2 = np.random.randint(0, 225)
        else:
            rand_x_1 = np.random.randint(0, 225)
            rand_y_1 = np.random.randint(0, 225)

            rand_x_2 = np.random.randint(275, 511)
            rand_y_2 = np.random.randint(275, 511)

        point_1, point_1_target = inp_batch[rand_x_1, rand_y_1, :], inp_target[rand_x_1, rand_y_1, :]
        point_1, point_1_target = point_1.squeeze(), point_1_target.squeeze()

        point_2, point_2_target = inp_batch[rand_x_2, rand_y_2, :], inp_target[rand_x_2, rand_y_2, :]
        point_2, point_2_target = point_2.squeeze(), point_2_target.squeeze()

        if comp_hamming:
            hamming = compute_hamming(model, point_1, point_2)
            hamming_between.append(hamming)
        if comp_confusion:    
            confusion = compute_confusion(model, optim, point_1, point_2, point_1_target, point_2_target)
            confusion_between.append(confusion)

    return hamming_between, confusion_between

def get_activation_regions(model, input):
    with torch.no_grad():
        patterns = []
        # get patterns
        for i, pixel in enumerate(input):
            pixel = pixel.squeeze()
            act_pattern = model(pixel, act=True)
            act_pattern = set_positive_values_to_one(act_pattern)
            patterns.append(list(act_pattern.detach().cpu().numpy()))
        # get the amount of unique patterns
        unique_patterns = []
        for pattern in patterns:
            if pattern not in unique_patterns:
                unique_patterns.append(pattern)
    return len(unique_patterns), unique_patterns, patterns

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

def get_zero_neurons(model, input):
    # get patterns
    # Change this value to the total number of neurons in the network
    zero_neurons = np.zeros([256])
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

    hamming_in_region = []
    confusion_in_region = []
    hamming_between_region = []
    confusion_between_region = []

    min_confusion_in_region = []
    min_confusion_between_region = []

    mean_hamming_within = []
    std_hamming_within = []
    mean_hamming_between = []
    std_hamming_between = []

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
        if args.dead_neurons:
            if epoch % 100 == 0:
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

        # This gets confusion at the end of training, specify the number of epochs
        if args.confusion_in_region:
            if epoch > args.epochs-2:
                # 100
                _, confusion_in_region = hamming_within_regions(model, optim, inp_batch, inp_target, 100, comp_hamming=False)

        if args.confusion_between_region:
            if epoch > args.epochs-2:
                # 25000
                _, confusion_between_region = hamming_between_regions(model, optim, inp_batch, inp_target, 25000, comp_hamming=False)

        # mean hamming distance during training
        if args.mean_hamming_in_region:
            if epoch % 250 == 0:
                hamming_in_region, confusion_in = hamming_within_regions(model, optim, inp_batch, inp_target, 100, comp_confusion=False)
                mean_hamming_within.append(np.mean(np.array(hamming_in_region)))
                std_hamming_within.append(np.std(np.array(hamming_in_region)))

        if args.mean_hamming_between_region:
            if epoch % 250 == 0:
                hamming_between_region, confusion_between = hamming_between_regions(model, optim, inp_batch, inp_target, 50000, comp_confusion=False)
                mean_hamming_between.append(np.mean(np.array(hamming_between_region)))
                std_hamming_between.append(np.std(np.array(hamming_between_region)))

        # Calculate the spectral norm of the weight matrices for param norms
        if args.param_norms:
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
                total_norm = norm_1 * norm_2 * norm_3
                param_norms.append(total_norm.item())

        # get activation regions during training for plot
        if args.act_patterns:
            # Get num regions for raw_xy
            if epoch % 100 == 0:
                print('Counting activation patterns...')
                raw_regions, unique_patterns, all_patterns = get_activation_regions(model, inp_batch)
                print('number of unique activation regions raw_xy: {}'.format(raw_regions))
                num_patterns.append(raw_regions)

    return param_norms, num_patterns, layer_1_norms, layer_2_norms, layer_3_norms, zero_neuron_plot, losses, inp_batch, inp_target, confusion_in_region, confusion_between_region, mean_hamming_within, mean_hamming_between

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=256, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=5000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='make a training and testing set')
    parser.add_argument('--print_loss', type=bool, default=True, help='print training loss')
    parser.add_argument('--param_norms', type=bool, default=False, help='plot param norms')
    parser.add_argument('--dead_neurons', type=bool, default=False, help='track dead neurons')
    parser.add_argument('--mean_hamming_in_region', type=bool, default=False, help='doing line count')
    parser.add_argument('--mean_hamming_between_region', type=bool, default=False, help='doing line count')
    parser.add_argument('--confusion_in_region', type=bool, default=False, help='doing line count')
    parser.add_argument('--confusion_between_region', type=bool, default=False, help='doing line count')
    parser.add_argument('--act_patterns', type=bool, default=False, help='check number of unique activation patterns')
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
    averaged_local_confusion_xy = []
    averaged_global_confusion_xy = []

    averaged_local_confusion_pe = {}
    averaged_global_confusion_pe = {}

    for l in L_vals:
        averaged_local_confusion_pe[f'{l}_val'] = []
        averaged_global_confusion_pe[f'{l}_val'] = []

    # Go through each image individually
    for im in test_data:

        #################################### Raw XY ##############################################

        # Set up raw_xy network
        model_raw = Net(2, args.neurons).to('cuda:0')
        optim_raw = torch.optim.Adam(model_raw.parameters(), lr=.001)
        criterion = nn.MSELoss()

        if args.train_coordinates:
            print("Beginning Raw XY Training...")
            xy_param_norms, xy_num_patterns, xy_layer_1_norms, xy_layer_2_norms, xy_layer_3_norms, xy_zero_neuron_plot, xy_losses, inp_batch, inp_target, confusion_within_xy, confusion_between_xy, mean_hamming_within_xy, mean_hamming_between_xy = train(model_raw, optim_raw, criterion, im, 'raw_xy', 0, args)
        
        averaged_global_confusion_xy.append(confusion_between_xy)
        averaged_local_confusion_xy.append(confusion_within_xy)

        #################################### Sin Cos #############################################

        if args.train_encoding:
            print("\nBeginning Positional Encoding Training...")
            for l in L_vals:

                # Set up pe network
                model_pe = Net(l*4, args.neurons).to('cuda:0')
                optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)
                criterion = nn.MSELoss()

                pe_param_norms, pe_num_patterns, pe_layer_1_norms, pe_layer_2_norms, pe_layer_3_norms, pe_zero_neuron_plot, pe_losses, inp_batch, inp_target, confusion_within_pe, confusion_between_pe, mean_hamming_within_pe, mean_hamming_between_pe = train(model_pe, optim_pe, criterion, im, 'sin_cos', l, args)

                averaged_global_confusion_pe[f'{l}_val'].append(confusion_between_pe)
                averaged_local_confusion_pe[f'{l}_val'].append(confusion_within_pe)

    for l in L_vals:

        averaged_global_confusion_pe[f'{l}_val'] = np.array(averaged_global_confusion_pe[f'{l}_val'])
        averaged_global_confusion_pe[f'{l}_val'] = averaged_global_confusion_pe[f'{l}_val'].flatten()

        averaged_local_confusion_pe[f'{l}_val'] = np.array(averaged_local_confusion_pe[f'{l}_val'])
        averaged_local_confusion_pe[f'{l}_val'] = averaged_local_confusion_pe[f'{l}_val'].flatten()

    averaged_global_confusion_xy = np.array(averaged_global_confusion_xy)
    averaged_global_confusion_xy = averaged_global_confusion_xy.flatten()

    averaged_local_confusion_xy = np.array(averaged_local_confusion_xy)
    averaged_local_confusion_xy = averaged_local_confusion_xy.flatten()

    sns.kdeplot(data=averaged_global_confusion_pe[f'4_val'], fill=True, label='L=4')
    sns.kdeplot(data=averaged_global_confusion_pe[f'8_val'], fill=True, label='L=8')
    sns.kdeplot(data=averaged_global_confusion_pe[f'16_val'], fill=True, label='L=16')
    sns.kdeplot(data=averaged_global_confusion_xy, fill=True, label='coordinates')
    plt.legend(frameon=False)
    plt.show()

    sns.kdeplot(data=averaged_local_confusion_pe[f'4_val'], fill=True, label='L=4')
    sns.kdeplot(data=averaged_local_confusion_pe[f'8_val'], fill=True, label='L=8')
    sns.kdeplot(data=averaged_local_confusion_pe[f'16_val'], fill=True, label='L=16')
    sns.kdeplot(data=averaged_local_confusion_xy, fill=True, label='coordinates')
    plt.legend(frameon=False)
    plt.show()

if __name__ == '__main__':
    main()

