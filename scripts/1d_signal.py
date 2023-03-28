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
from functools import reduce
from argparse import Namespace
from scipy import signal
from scipy.fft import fftshift

sns.set_style('darkgrid')
colors = sns.color_palette()

# Data Generation
opt = Namespace()
opt.N = 2048
opt.K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
opt.A = [1 for _ in opt.K]
opt.PHI = [np.random.rand() for _ in opt.K]
device = 'cuda:0'

class Net(nn.Module):
    def __init__(self, input_dim, hidden):
        super(Net, self).__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x, act=False):
        out = self.relu(self.l1(x))

        if act:
            act_1 = out

        out = self.relu(self.l2(out))

        if act:
            act_2 = out

        out = self.l3(out)

        if act:
            # using batches, cat along 1st dimension
            pattern = torch.cat((act_1, act_2), dim=0).squeeze()
            return pattern

        return out

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
    criterion = nn.MSELoss()

    optim.zero_grad()
    output = model(x_0)
    loss = criterion(output, y_0)
    loss.backward()

    grad_1 = torch.cat([torch.flatten(model.l1.weight.grad), torch.flatten(model.l2.weight.grad), 
                        torch.flatten(model.l3.weight.grad), torch.flatten(model.l1.bias.grad), 
                        torch.flatten(model.l2.bias.grad), torch.flatten(model.l3.bias.grad),
                        ])

    optim.zero_grad()
    output = model(x_1)
    loss = criterion(output, y_1)
    loss.backward()
    grad_2 = torch.cat([torch.flatten(model.l1.weight.grad), torch.flatten(model.l2.weight.grad), 
                        torch.flatten(model.l3.weight.grad), torch.flatten(model.l1.bias.grad), 
                        torch.flatten(model.l2.bias.grad), torch.flatten(model.l3.bias.grad),
                        ])

    # get inner products of gradients
    confusion = torchmetrics.functional.pairwise_cosine_similarity(grad_1.unsqueeze(dim=0), 
                                                                grad_2.unsqueeze(dim=0)).cpu()

    return confusion.item()

def hamming_within_regions(model, optim, inp_batch, inp_target, iterations, comp_hamming=True, comp_confusion=True):

    print('Starting hamming within local regions...')
    hamming_local = []
    confusion_local = []

    # do line count between all inputs in region
    for i in range(iterations):
        # get a random 3x3 patch
        rand_x = np.random.randint(26, 2020)

        patch = inp_batch[rand_x-25:rand_x+25]
        patch_target = inp_target[rand_x-25:rand_x+25]

        # get confusion and hamming for every coordinate in the 3x3 region
        for j in range(500):
            rand_1 = np.random.randint(0, 49)
            rand_2 = np.random.randint(0, 49)

            if comp_hamming:
                hamming = compute_hamming(model, patch[rand_1], patch[rand_2])
                hamming_local.append(hamming)
            if comp_confusion:
                confusion = compute_confusion(model, optim, patch[rand_1], patch[rand_2], patch_target[rand_1], patch_target[rand_2])
                confusion_local.append(confusion)

    return hamming_local, confusion_local

def hamming_between_regions(model, optim, inp_batch, inp_target, iterations, comp_hamming=True, comp_confusion=True):

    print('Starting hamming across input space...')
    hamming_between = []
    confusion_between = []

    for i in range(iterations):

        rand_x_1 = np.random.randint(0, 800)
        rand_x_2 = np.random.randint(1200, 2047)

        # first point
        point_1 = inp_batch[rand_x_1]
        point_1_target = inp_target[rand_x_1]

        # second point
        point_2 = inp_batch[rand_x_2]
        point_2_target = inp_target[rand_x_2]

        if comp_hamming:
            hamming = compute_hamming(model, point_1, point_2)
            hamming_between.append(hamming)
        if comp_confusion:    
            confusion = compute_confusion(model, optim, point_1, point_2, point_1_target, point_2_target)
            confusion_between.append(confusion)

    return hamming_between, confusion_between

def set_positive_values_to_one(tensor):
    # Create a mask that is 1 for positive values and 0 for non-positive values
    mask = tensor.gt(0.0)
    # Set all positive values to 1 using the mask
    tensor[mask] = 1
    return tensor

def get_activation_regions(model, input):
    with torch.no_grad():
        patterns = []
        # get patterns
        act_pattern = model(input, act=True)
        for i, pixel in enumerate(act_pattern):
            pre_act = set_positive_values_to_one(pixel)
            patterns.append(list(pre_act.detach().cpu().numpy()))
        # get the amount of unique patterns
        unique_patterns = []
        for pattern in patterns:
            if pattern not in unique_patterns:
                unique_patterns.append(pattern)
    return len(unique_patterns), unique_patterns, patterns

def make_phased_waves(opt):
    t = np.arange(0, 1, 1./opt.N)
    if opt.A is None:
        yt = reduce(lambda a, b: a + b, 
                    [np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, phi in zip(opt.K, opt.PHI)])
    else:
        yt = reduce(lambda a, b: a + b, 
                    [Ai * np.sin(2 * np.pi * ki * t + 2 * np.pi * phi) for ki, Ai, phi in zip(opt.K, opt.A, opt.PHI)])

    return yt, t

def plot_wave_and_spectrum(opt, x, yox):
    # Btw, "yox" --> "y of x"
    plt.figure(figsize=(9,4)) 
    plt.plot(x, yox)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()
    
def train(model, optim, criterion, x, target, args):

    mean_hamming_within = []
    std_hamming_within = []
    mean_hamming_between = []
    std_hamming_between = []

    confusion_in_region = []
    confusion_between_region = []

    for epoch in range(args.epochs):

        optim.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        print('Epoch {} Loss: {}'.format(epoch, loss.item()))
        loss.backward()
        optim.step()

        if args.confusion_in_region:
            if epoch > args.epochs-2:
                _, confusion_in_region = hamming_within_regions(model, optim, x, target, 100, comp_hamming=False)

        if args.confusion_between_region:
            if epoch > args.epochs-2:
                _, confusion_between_region = hamming_between_regions(model, optim, x, target, 10000, comp_hamming=False)

        # hamming distance and min confusion during training
        # change comp_confusion to true if you want to get the minimum confusion bound
        if args.mean_hamming_in_region:
            if epoch % 100 == 0:
                hamming_in_region, _ = hamming_within_regions(model, optim, x, target, 100, comp_confusion=False)
                mean_hamming_within.append(np.mean(np.array(hamming_in_region)))
                std_hamming_within.append(np.std(np.array(hamming_in_region)))

        if args.mean_hamming_between_region:
            if epoch % 100 == 0:
                hamming_between_region, _ = hamming_between_regions(model, optim, x, target, 10000, comp_confusion=False)
                mean_hamming_between.append(np.mean(np.array(hamming_between_region)))
                std_hamming_between.append(np.std(np.array(hamming_between_region)))

    
    return confusion_in_region, confusion_between_region, mean_hamming_between, mean_hamming_within, std_hamming_between, std_hamming_within

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--mean_hamming_in_region', type=bool, default=False, help='doing line count')
    parser.add_argument('--mean_hamming_between_region', type=bool, default=False, help='doing line count')
    parser.add_argument('--confusion_in_region', type=bool, default=False, help='doing line count')
    parser.add_argument('--confusion_between_region', type=bool, default=False, help='doing line count')

    args = parser.parse_args()

    # Define model and other training parameters
    model_raw = Net(1, 256).to(device)
    optim_raw = optim.Adam(model_raw.parameters(), lr=.001)

    model_pe = Net(12, 256).to(device)
    optim_pe = optim.Adam(model_pe.parameters(), lr=.001)

    criterion = nn.MSELoss()

    epochs = args.epochs
    yt, x = make_phased_waves(opt)

    # make data into tensors, and also add encoding for data, L=5 

    encoding = np.stack((np.sin(np.pi*x), np.cos(np.pi*x), 
                            np.sin(2*np.pi * x), np.cos(2*np.pi * x),
                            np.sin(2**2 * np.pi * x), np.cos(2**2 * np.pi * x),
                            np.sin(2**3 * np.pi * x), np.cos(2**3 * np.pi * x),
                            np.sin(2**4 * np.pi * x), np.cos(2**4 * np.pi * x),
                            np.sin(2**5 * np.pi * x), np.cos(2**5 * np.pi * x)), axis=1)

    #np.savetxt('spectrogram/orig_function.txt', target)

    # plot preliminary data and spectrogram
    #plot_wave_and_spectrum(opt, x, yt)

    x = torch.Tensor(x).to(device)
    x = x.unsqueeze(dim=1)

    encoding = torch.Tensor(encoding).to(device)

    target = torch.Tensor(yt).to(device)
    target = target.unsqueeze(dim=1)

    # Start training loop, will need to evaluate the network at every other iteration to determine the spectrogram and
    # evaluate through training
    confusion_within_xy, confusion_between_xy, hamming_between_xy, hamming_within_xy, std_hamming_between_xy, std_hamming_within_xy = train(model_raw, optim_raw, criterion, x, target, args)
    confusion_within_pe, confusion_between_pe, hamming_between_pe, hamming_within_pe, std_hamming_between_pe, std_hamming_within_pe = train(model_pe, optim_pe, criterion, encoding, target, args)

    x = np.linspace(0, 1000, 10)

    plt.plot(x, np.array(hamming_between_xy), label='Coordinates')
    plt.fill_between(x, np.array(hamming_between_xy)+np.array(std_hamming_between_xy), np.array(hamming_between_xy)-np.array(std_hamming_between_xy), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True )

    plt.plot(x, np.array(hamming_between_pe), label='Positional Encoding')
    plt.fill_between(x, np.array(hamming_between_pe)+np.array(std_hamming_between_pe), np.array(hamming_between_pe)-np.array(std_hamming_between_pe), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True )
    plt.legend()
    plt.show()

    plt.plot(x, np.array(hamming_within_xy), label='Coordinates')
    plt.fill_between(x, np.array(hamming_within_xy)+np.array(std_hamming_within_xy), np.array(hamming_within_xy)-np.array(std_hamming_within_xy), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True )

    plt.plot(x, np.array(hamming_within_pe), label='Positional Encoding')
    plt.fill_between(x, np.array(hamming_within_pe)+np.array(std_hamming_within_pe), np.array(hamming_within_pe)-np.array(std_hamming_within_pe), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True )
    plt.legend()
    plt.show()

    sns.kdeplot(confusion_within_xy, fill=True, label='Coordinates Local')
    sns.kdeplot(confusion_within_pe, fill=True, label='Positional Encoding Local')
    sns.kdeplot(confusion_between_xy, fill=True, label='Coordinates Global')
    sns.kdeplot(confusion_between_pe, fill=True, label='Positional Encoding Global')
    plt.legend()
    plt.show()

    plt.figure(figsize=(9,4)) 
    with torch.no_grad():
        out_xy = model_raw(x)
        out_xy = np.array(out_xy.cpu())
        out_pe = model_pe(encoding)
        out_pe = np.array(out_pe.cpu())
    x = np.array(x.cpu())
    plt.plot(x, yt, label='Ground Truth')
    plt.plot(x, out_xy, label='Coordinates', linestyle='dashed')
    plt.plot(x, out_pe, label='Positional Encoding', linestyle='dashed')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.show()


if __name__ == '__main__':
    main()
