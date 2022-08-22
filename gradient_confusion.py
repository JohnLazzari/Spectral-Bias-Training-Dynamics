import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from nerf2D import Positional_Encoding
from torch_network import Net, SIN, SIREN
import random
import torchmetrics
import seaborn as sns
from mfn import FourierNet, GaborNet
import scipy
from torch import linalg as LA
import itertools
import argparse
from matplotlib import cm

def get_activation_regions(model, input):
    with torch.no_grad():
        patterns = []
        # get patterns
        for i, pixel in enumerate(input):
            act_pattern = model(pixel, act=True)
            patterns.append(list(act_pattern))
        # get the amount of unique patterns
        unique_patterns = []
        for pattern in patterns:
            if pattern not in unique_patterns:
                unique_patterns.append(pattern)
    return len(unique_patterns), unique_patterns, patterns

def confusion_within_region(model_raw, inp_target, inp_batch, optim_raw, criterion, patterns, all_patterns):

    within_region = []
    raw_gradients = torch.empty([4096, 17283]).to('cuda:0')
    for i, pixel in enumerate(inp_batch):
        optim_raw.zero_grad()
        output = model_raw(pixel)
        loss = criterion(output, inp_target[i])
        loss.backward()
        grads = torch.cat([torch.flatten(model_raw.l1.weight.grad), torch.flatten(model_raw.l2.weight.grad), torch.flatten(model_raw.l3.weight.grad), torch.flatten(model_raw.l1.bias.grad), torch.flatten(model_raw.l2.bias.grad), torch.flatten(model_raw.l3.bias.grad)])
        raw_gradients[i] = grads

    dict_patterns = {}
    # create a list for all pattern indices
    for i, pattern in enumerate(patterns):
        dict_patterns[tuple(pattern)] = []
    for i, pattern in enumerate(all_patterns):
        # assign each position a color
        dict_patterns[tuple(pattern)].append(i)
    print(len(dict_patterns.keys()))
    all_patterns = torch.Tensor(all_patterns).to('cuda:0')
    # get confusion within regions
    for pattern in dict_patterns:
        if len(dict_patterns[pattern]) > 1:
            # go through the region and get all inner products
            for i, index in enumerate(dict_patterns[pattern][:-1]):
                for j, next_index in enumerate(dict_patterns[pattern][i+1:]):
                    within_region.append(torch.flatten(torchmetrics.functional.pairwise_cosine_similarity(raw_gradients[index].unsqueeze(dim=0), raw_gradients[next_index].unsqueeze(dim=0))).cpu())

    sns.kdeplot(data=torch.cat(within_region).cpu(), fill=True, label='raw_xy')
    plt.legend()
    plt.show()

    # get confusion between regions
    return 0

def plot_patterns(patterns, all_patterns):

    dict_patterns = {}
    colors = np.zeros(4096)
    # regions dont look different if they all have the same color
    random.shuffle(patterns)
    for i, pattern in enumerate(patterns):
        # assign each position a color
        dict_patterns[tuple(pattern)] = i
    for i, pattern in enumerate(all_patterns):
        colors[i] = dict_patterns[tuple(pattern)]
    
    colors = np.reshape(colors, [64, 64])
    plt.pcolormesh(colors, cmap='Spectral')
    plt.colorbar()
    plt.show()

def plot_first_3_neurons(all_patterns):

    all_patterns = np.array(all_patterns)
    all_patterns = np.reshape(all_patterns, [64, 64, -1])

    # get neurons that arent all dead or activated to show hyperplane
    neuron_list = []

    # get random neurons to demonstrate hyperplane arrangement
    first_neuron = all_patterns[:, :, 0]
    second_neuron = all_patterns[:, :, 9]
    third_neuron = all_patterns[:, :, 100]

    fig = plt.figure(figsize=(21, 6))

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    pcm1 = ax1.pcolormesh(first_neuron, cmap='Spectral')
    pcm2 = ax2.pcolormesh(second_neuron, cmap='Spectral')
    pcm3 = ax3.pcolormesh(third_neuron, cmap='Spectral')

    fig.colorbar(pcm1, ax=ax1)
    fig.colorbar(pcm2, ax=ax2)
    fig.colorbar(pcm3, ax=ax3)

    plt.show()

def get_zero_neurons(model, input):
    # get patterns
    zero_neurons = np.zeros([256])
    for i, pixel in enumerate(input):
        act_pattern = model(pixel, act=True)
        for j in range(act_pattern.shape[0]):
            if act_pattern[j] == 0.:
                zero_neurons[j] += 1
    dead_count = 0
    for item in zero_neurons:
        if item == 64*64:
            dead_count += 1
    return dead_count

def get_data(image, encoding, L=10, batch_size=2048, RFF=False, shuffle=True):

    # Get the training image
    trainimg= image
    trainimg = trainimg / 255.0  
    H, W, C = trainimg.shape

    # Get the encoding
    PE = Positional_Encoding(trainimg, encoding, training=True)
    inp_batch, inp_target, ind_vals = PE.get_dataset(L, RFF=False)

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

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=128, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='make a training and testing set')
    parser.add_argument('--print_loss', type=bool, default=True, help='print training loss')
    parser.add_argument('--zero_grads', type=bool, default=False, help='check zeroed gradients')
    parser.add_argument('--param_norms', type=bool, default=False, help='plot param norms')
    parser.add_argument('--confusion', type=bool, default=False, help='track confusion')
    parser.add_argument('--dead_neurons', type=bool, default=False, help='track dead neurons')
    parser.add_argument('--act_patterns', type=bool, default=False, help='check number of unique activation patterns')
    parser.add_argument('--grad_norms', type=bool, default=False, help='check the gradient norms')
    parser.add_argument('--random', type=bool, default=False, help='use random image')

    args = parser.parse_args()

    # Set up raw_xy network
    model_raw = Net(2, args.neurons).to('cuda:0')
    optim_raw = torch.optim.Adam(model_raw.parameters(), lr=.001)

    # Set up sin_cos network
    model_pe = Net(32, args.neurons).to('cuda:0')
    optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)

    # loss and epochs
    criterion = nn.MSELoss()
    epochs = args.epochs

    # compute gradients individually for each, not sure best way to do this yet
    if args.random:
        im2arr = np.random.randint(0, 255, (64, 64, 3))
    else:
        im = Image.open(f'fractal_small.jpg')
        im2arr = np.array(im) 

    # during training
    zero_grads = args.zero_grads
    param_norms = args.param_norms
    confusion = args.confusion
    dead_neurons = args.dead_neurons
    grad_norms = args.grad_norms
    # after training
    act_patterns = args.act_patterns

    #################################### Raw xy ###############################################

    # Get the encoding raw_xy
    train_inp_batch, train_inp_target = get_data(image=im2arr, encoding='raw_xy', L=0, batch_size=args.batch_size)
    inp_batch, inp_target = get_data(image=im2arr, encoding='raw_xy', L=0, batch_size=1, shuffle=False)
    full_batch, full_target = get_data(image=im2arr, encoding='raw_xy', L=0, batch_size=64*64, shuffle=False)

    raw_grad_similarities = []
    raw_param_norms = []
    raw_num_patterns = []

    layer_1_norms = []
    layer_3_norms = []
    layer_2_norms = []

    total_grad_norm_raw = []
    zero_neuron_plot = []

    raw_losses = []

    for epoch in range(epochs):

        dead_neuron_count = 0
        running_loss = 0

        if dead_neurons:
            if epoch % 100 == 0:
                with torch.no_grad():
                    dead = get_zero_neurons(model_raw, inp_batch)
                    print('Number dead neurons at epoch {}: {}'.format(epoch, dead))
                    zero_neuron_plot.append(dead)

        for i, pixel in enumerate(train_inp_batch):
            optim_raw.zero_grad()
            output = model_raw(pixel)
            loss = criterion(output, train_inp_target[i])
            running_loss += loss.item()
            loss.backward()
            optim_raw.step()
        epoch_loss = running_loss / 16
        if args.print_loss:
            print('Loss at epoch {}: {}'.format(epoch, epoch_loss))
        raw_losses.append(epoch_loss)

        if grad_norms:
            optim_raw.zero_grad()
            output = model_raw(full_batch)
            loss = criterion(output, full_target)
            loss.backward()
            U, S, V = torch.linalg.svd(model_raw.l1.weight.grad)
            norm_1 = max(S)
            U, S, V = torch.linalg.svd(model_raw.l2.weight.grad)
            norm_2 = max(S)
            U, S, V = torch.linalg.svd(model_raw.l3.weight.grad)
            norm_3 = max(S)
            grad_norm = norm_1 + norm_2 + norm_3
            total_grad_norm_raw.append(grad_norm.item())
        '''
        with torch.no_grad():
            pixel_1 = model_raw(inp_batch[100], act=True)
            pixel_2 = model_raw(inp_batch[101], act=True)
            difference = torch.sum(torch.abs(pixel_1 - pixel_2))
            print(difference)
        '''

        # This is to find parameter norms for lipschitz constant
        if param_norms:
            with torch.no_grad():
                # Go through each layer and add up the norms
                U, S, V = torch.linalg.svd(model_raw.l1.weight)
                norm_1 = max(S)
                layer_1_norms.append(norm_1.item())
                U, S, V = torch.linalg.svd(model_raw.l2.weight)
                norm_2 = max(S)
                layer_2_norms.append(norm_2.item())
                U, S, V = torch.linalg.svd(model_raw.l3.weight)
                norm_3 = max(S)
                layer_3_norms.append(norm_3.item())
                total_norm = norm_1 * norm_2 * norm_3
                raw_param_norms.append(total_norm.item())

        if zero_grads:
            # find number of weight gradients that are zero
            optim_raw.zero_grad()
            output = model_raw(full_batch)
            loss = criterion(output, full_target)
            loss.backward()
            # check the amount of dead neurons
            all_grads = torch.cat([torch.flatten(model_raw.l1.weight.grad), torch.flatten(model_raw.l2.weight.grad), torch.flatten(model_raw.l3.weight.grad)])
            # Check amount of weight gradients that are zero
            for grad in all_grads:
                if grad == 0.0:
                    dead_neuron_count += 1
            print('num dead weights: {}'.format(dead_neuron_count))

        if confusion:
            if epoch > epochs-2:
                raw_gradients = torch.empty([4096, 17283]).to('cuda:0')
                for i, pixel in enumerate(inp_batch):
                    optim_raw.zero_grad()
                    output = model_raw(pixel)
                    loss = criterion(output, inp_target[i])
                    loss.backward()
                    grads = torch.cat([torch.flatten(model_raw.l1.weight.grad), torch.flatten(model_raw.l2.weight.grad), torch.flatten(model_raw.l3.weight.grad), torch.flatten(model_raw.l1.bias.grad), torch.flatten(model_raw.l2.bias.grad), torch.flatten(model_raw.l3.bias.grad)])
                    raw_gradients[i] = grads
                # get inner products of gradients for raw xy
                for i in range(raw_gradients.shape[0]):
                    #print(torchmetrics.functional.pairwise_cosine_similarity(raw_gradients[i].unsqueeze(dim=0), raw_gradients[j].unsqueeze(dim=0)).cpu().item())
                    raw_grad_similarities.append(torch.flatten(torchmetrics.functional.pairwise_cosine_similarity(raw_gradients[i].unsqueeze(dim=0), raw_gradients[i+1:])).cpu())

        # get activation regions during training for plot
        if act_patterns:
            # Get num regions for raw_xy
            if epoch % 100 == 0:
                raw_regions, unique_patterns, all_patterns = get_activation_regions(model_raw, inp_batch)
                print('number of unique activation regions raw_xy: {}'.format(raw_regions))
                confusion_within_region(model_raw, inp_target, inp_batch, optim_raw, criterion, unique_patterns, all_patterns)
                raw_num_patterns.append(raw_regions)

    if dead_neurons:
        plt.plot(zero_neuron_plot)
        plt.show()

    if param_norms:
        plt.plot(layer_1_norms, label='layer1')
        plt.plot(layer_2_norms, label='layer2')
        plt.plot(layer_3_norms, label='layer3')
        plt.legend()
        plt.show()

    '''
    if act_patterns:
        # Get num regions for raw_xy
        with torch.no_grad():
            raw_regions, unique_patterns, all_patterns = get_activation_regions(model_raw, inp_batch)
            plot_first_3_neurons(all_patterns)
            print('number of unique activation regions raw_xy: {}'.format(raw_regions))
            plot_patterns(unique_patterns, all_patterns)
            raw_num_patterns.append(raw_regions)
    '''

    ######################## Positional Encoding ######################################

    # Get the encoding sin cos
    train_inp_batch, train_inp_target = get_data(image=im2arr, encoding='sin_cos', L=8, batch_size=args.batch_size)
    inp_batch, inp_target = get_data(image=im2arr, encoding='sin_cos', L=8, batch_size=1, shuffle=False)
    full_batch, full_target = get_data(image=im2arr, encoding='sin_cos', L=8, batch_size=64*64)

    sin_cos_grad_similarity = []
    sin_cos_norms = []

    layer_1_norms_pe = []
    layer_3_norms_pe = []
    layer_2_norms_pe = []

    total_grad_norm_pe = []

    pe_num_patterns = []
    pe_losses = []

    zero_neuron_plot_pe = []

    for epoch in range(epochs):

        dead_neuron_count = 0
        running_loss = 0

        if dead_neurons:
            if epoch % 1000 == 0:
                with torch.no_grad():
                    dead = get_zero_neurons(model_pe, inp_batch)
                    print('Number dead neurons at epoch {}: {}'.format(epoch, dead))
                    zero_neuron_plot_pe.append(dead)

        for i, pixel in enumerate(train_inp_batch):
            optim_pe.zero_grad()
            output = model_pe(pixel)
            loss = criterion(output, train_inp_target[i])
            running_loss += loss.item()
            loss.backward()
            optim_pe.step()
        epoch_loss = running_loss / 16
        pe_losses.append(epoch_loss)
        if args.print_loss:
            print('Loss at epoch {}: {}'.format(epoch, epoch_loss))

        if grad_norms:
            optim_pe.zero_grad()
            output = model_pe(full_batch)
            loss = criterion(output, full_target)
            loss.backward()
            U, S, V = torch.linalg.svd(model_pe.l1.weight.grad)
            norm_1 = max(S)
            U, S, V = torch.linalg.svd(model_pe.l2.weight.grad)
            norm_2 = max(S)
            U, S, V = torch.linalg.svd(model_pe.l3.weight.grad)
            norm_3 = max(S)
            total_grad_norm = norm_1 + norm_2 + norm_3
            total_grad_norm_pe.append(total_grad_norm.item())
        '''
        with torch.no_grad():
            pixel_1 = model_pe(inp_batch[100], act=True)
            pixel_2 = model_pe(inp_batch[101], act=True)
            difference = torch.sum(torch.abs(pixel_1 - pixel_2))
            print(difference)
        '''

        # Get param norms for lipschitz sin_cos
        if param_norms:
            with torch.no_grad():
                U, S, V = torch.linalg.svd(model_pe.l1.weight)
                norm_1 = max(S)
                layer_1_norms_pe.append(norm_1.item())
                U, S, V = torch.linalg.svd(model_pe.l2.weight)
                norm_2 = max(S)
                layer_2_norms_pe.append(norm_2.item())
                U, S, V = torch.linalg.svd(model_pe.l3.weight)
                norm_3 = max(S)
                layer_3_norms_pe.append(norm_3.item())
                total_norm = norm_1 * norm_2 * norm_3
                sin_cos_norms.append(total_norm.item())
        
        if zero_grads:
            # Get gradient norms for sin cos
            optim_pe.zero_grad()
            output = model_pe(full_batch)
            loss = criterion(output, full_target)
            loss.backward()
            all_grads = torch.cat([torch.flatten(model_pe.l1.weight.grad), torch.flatten(model_pe.l2.weight.grad), torch.flatten(model_pe.l3.weight.grad)])
            # Check amount of weight gradients which are zero
            for grad in all_grads:
                if grad == 0:
                    dead_neuron_count += 1
            print('num dead weights: {}'.format(dead_neuron_count))

        # get confusion sin_cos
        if confusion:
            if epoch > epochs-2:
                pe_gradients = torch.empty([4096, 21123]).to('cuda:0')
                for i, pixel in enumerate(inp_batch):
                    optim_pe.zero_grad()
                    output = model_pe(pixel)
                    loss = criterion(output, inp_target[i])
                    loss.backward()
                    grads = torch.cat([torch.flatten(model_pe.l1.weight.grad), torch.flatten(model_pe.l2.weight.grad), torch.flatten(model_pe.l3.weight.grad), torch.flatten(model_pe.l1.bias.grad), torch.flatten(model_pe.l2.bias.grad), torch.flatten(model_pe.l3.bias.grad)])
                    pe_gradients[i] = grads
                # get inner products of gradients for sin_cos
                for i in range(pe_gradients.shape[0]):
                    sin_cos_grad_similarity.append(torch.flatten(torchmetrics.functional.pairwise_cosine_similarity(pe_gradients[i].unsqueeze(dim=0), pe_gradients[i+1:]).cpu()))

        if act_patterns:
            # Get the number of activation regions for sin_cos
            if epoch % 50 == 0:
                with torch.no_grad():
                    pe_regions, unique_patterns, all_patterns = get_activation_regions(model_pe, inp_batch)
                    print('Num unique activation patterns sin_cos: {}'.format(pe_regions))
                    pe_num_patterns.append(pe_regions)

    if act_patterns:
        # get the regions over loss
        plt.plot(pe_num_patterns, label='positional encoding')
        plt.plot(raw_num_patterns, label='no positional encoding')
        plt.legend()
        plt.show()

    '''
    plt.plot(pe_losses, label='positional encoding')
    plt.plot(raw_losses, label='no positional encoding')
    plt.legend()
    plt.show()
    '''

    if grad_norms:
        plt.plot(total_grad_norm_raw, label='raw_grad_norms')
        plt.plot(total_grad_norm_pe, label='pe_grad_norms')
        plt.legend()
        plt.show()

    if param_norms:
        # plotting the param norms between layers for sin cos
        plt.plot(layer_1_norms_pe, label='layer1')
        plt.plot(layer_2_norms_pe, label='layer2')
        plt.plot(layer_3_norms_pe, label='layer3')
        plt.legend()
        plt.show()

    if act_patterns:
        # Get the number of activation regions for sin_cos
        with torch.no_grad():
            pe_regions, unique_patterns, all_patterns = get_activation_regions(model_pe, inp_batch)
            plot_first_3_neurons(all_patterns)
            plot_patterns(unique_patterns, all_patterns)
            print('Num unique activation patterns sin_cos: {}'.format(pe_regions))
            pe_num_patterns.append(pe_regions)

    if param_norms:
        # plot the total param norms for raw xy and sin cos
        plt.plot(sin_cos_norms, label='Fourier Features')
        plt.plot(raw_param_norms, label='raw_xy')
        plt.legend()
        plt.title('Spectral Norm Parameters')
        plt.xlabel('Epochs')
        plt.ylabel('Norm Across Layers')
        plt.show()

    if zero_grads:
        dead_neuron_count = 0
        # Get gradient norms for sin cos
        optim_pe.zero_grad()
        output = model_pe(full_batch)
        loss = criterion(output, full_target)
        loss.backward()
        all_grads = torch.cat([torch.flatten(model_pe.l1.weight.grad), torch.flatten(model_pe.l2.weight.grad), torch.flatten(model_pe.l3.weight.grad)])
        # Check amount of weight gradients which are zero
        for grad in all_grads:
            if grad == 0:
                dead_neuron_count += 1
        print('num dead weights: {}'.format(dead_neuron_count))

    if confusion:
        sns.kdeplot(data=torch.cat(sin_cos_grad_similarity).cpu(), fill=True, label='Fourier Features')
        sns.kdeplot(data=torch.cat(raw_grad_similarities).cpu(), fill=True, label='raw_xy')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()