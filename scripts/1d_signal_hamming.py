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
opt.K = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
opt.A = [1 for _ in opt.K]
opt.PHI = [np.random.rand() for _ in opt.K]
device = 'cuda:0'

class Net(nn.Module):
    def __init__(self, input_dim, hidden):
        super(Net, self).__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.l4 = nn.Linear(hidden, hidden)
        self.l5 = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

    def forward(self, x, act=False):
        out_1 = self.relu(self.l1(x))
        out_2 = self.relu(self.l2(out_1))
        out_3 = self.relu(self.l3(out_2))
        out_4 = self.relu(self.l4(out_3))
        out_5 = self.l5(out_4)

        if act:
            # using batches, cat along 1st dimension
            pattern = torch.cat((out_1, out_2, out_3, out_4), dim=0).squeeze()
            return pattern

        return out_5

def compute_hamming(model, x_0, x_1):

    # hamming distance
    with torch.no_grad():
        pattern_1 = model(x_0, act=True)
        pattern_1 = set_positive_values_to_one(pattern_1)
        pattern_2 = model(x_1, act=True)
        pattern_2 = set_positive_values_to_one(pattern_2)

    hamming = torch.sum(torch.abs(pattern_1-pattern_2))

    return hamming.item()

def hamming_within_regions(model, optim, inp_batch, inp_target, iterations):

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
        for j in range(250):
            rand_1 = np.random.randint(0, 49)
            rand_2 = np.random.randint(0, 49)

            hamming = compute_hamming(model, patch[rand_1], patch[rand_2])
            hamming_local.append(hamming)

    return hamming_local

def hamming_between_regions(model, optim, inp_batch, inp_target, iterations):

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

        hamming = compute_hamming(model, point_1, point_2)
        hamming_between.append(hamming)

    return hamming_between

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
    mean_hamming_between = []

    for epoch in range(args.epochs):

        optim.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        print('Epoch {} Loss: {}'.format(epoch, loss.item()))
        loss.backward()
        optim.step()

        # hamming distance and min confusion during training
        # change comp_confusion to true if you want to get the minimum confusion bound
        if epoch % 50 == 0:
            hamming_in_region = hamming_within_regions(model, optim, x, target, 25)
            mean_hamming_within.append(np.mean(np.array(hamming_in_region)))

            hamming_between_region = hamming_between_regions(model, optim, x, target, 5000)
            mean_hamming_between.append(np.mean(np.array(hamming_between_region)))

    return mean_hamming_between, mean_hamming_within

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')

    args = parser.parse_args()
    runs = [1, 2, 3, 4, 5]

    total_hamming_xy_local = []
    total_hamming_xy_global = []

    total_hamming_pe3_local = []
    total_hamming_pe3_global = []

    total_hamming_pe6_local = []
    total_hamming_pe6_global = []

    for run in runs:

        # Define model and other training parameters
        model_raw = Net(1, 128).to(device)
        optim_raw = optim.Adam(model_raw.parameters(), lr=.001)

        model_pe3 = Net(8, 128).to(device)
        optim_pe3 = optim.Adam(model_pe3.parameters(), lr=.001)

        model_pe6 = Net(14, 128).to(device)
        optim_pe6 = optim.Adam(model_pe6.parameters(), lr=.001)

        criterion = nn.MSELoss()

        yt, x = make_phased_waves(opt)

        # make data into tensors, and also add encoding for data, L=5 
        encoding_3 = np.stack((np.sin(np.pi*x), np.cos(np.pi*x), 
                                np.sin(2*np.pi * x), np.cos(2*np.pi * x),
                                np.sin(2**2 * np.pi * x), np.cos(2**2 * np.pi * x),
                                np.sin(2**3 * np.pi * x), np.cos(2**3 * np.pi * x)), axis=1)

        encoding_6 = np.stack((np.sin(np.pi*x), np.cos(np.pi*x), 
                                np.sin(2*np.pi * x), np.cos(2*np.pi * x),
                                np.sin(2**2 * np.pi * x), np.cos(2**2 * np.pi * x),
                                np.sin(2**3 * np.pi * x), np.cos(2**3 * np.pi * x),
                                np.sin(2**4 * np.pi * x), np.cos(2**4 * np.pi * x),
                                np.sin(2**5 * np.pi * x), np.cos(2**5 * np.pi * x),
                                np.sin(2**6 * np.pi * x), np.cos(2**6 * np.pi * x)), axis=1)

        x = torch.Tensor(x).to(device)
        x = x.unsqueeze(dim=1)

        encoding_3 = torch.Tensor(encoding_3).to(device)
        encoding_6 = torch.Tensor(encoding_6).to(device)

        target = torch.Tensor(yt).to(device)
        target = target.unsqueeze(dim=1)

        # Start training loop, will need to evaluate the network at every other iteration to determine the spectrogram and
        # evaluate through training
        hamming_between_xy, hamming_within_xy = train(model_raw, optim_raw, criterion, x, target, args)
        hamming_between_pe3, hamming_within_pe3 = train(model_pe3, optim_pe3, criterion, encoding_3, target, args)
        hamming_between_pe6, hamming_within_pe6 = train(model_pe6, optim_pe6, criterion, encoding_6, target, args)

        total_hamming_xy_local.append(hamming_within_xy)
        total_hamming_xy_global.append(hamming_between_xy)

        total_hamming_pe3_local.append(hamming_within_pe3)
        total_hamming_pe3_global.append(hamming_between_pe3)

        total_hamming_pe6_local.append(hamming_within_pe6)
        total_hamming_pe6_global.append(hamming_between_pe6)

    total_hamming_xy_local = np.array(total_hamming_xy_local)
    total_hamming_xy_local_std = np.std(total_hamming_xy_local, axis=0)
    total_hamming_xy_local = np.mean(total_hamming_xy_local, axis=0)
    
    total_hamming_xy_global = np.array(total_hamming_xy_global)
    total_hamming_xy_global_std = np.std(total_hamming_xy_global, axis=0)
    total_hamming_xy_global = np.mean(total_hamming_xy_global, axis=0)

    total_hamming_pe3_local = np.array(total_hamming_pe3_local)
    total_hamming_pe3_local_std = np.std(total_hamming_pe3_local, axis=0)
    total_hamming_pe3_local = np.mean(total_hamming_pe3_local, axis=0)

    total_hamming_pe3_global = np.array(total_hamming_pe3_global)
    total_hamming_pe3_global_std = np.std(total_hamming_pe3_global, axis=0)
    total_hamming_pe3_global = np.mean(total_hamming_pe3_global, axis=0)

    total_hamming_pe6_local = np.array(total_hamming_pe6_local)
    total_hamming_pe6_local_std = np.std(total_hamming_pe6_local, axis=0)
    total_hamming_pe6_local = np.mean(total_hamming_pe6_local, axis=0)

    total_hamming_pe6_global = np.array(total_hamming_pe6_global)
    total_hamming_pe6_global_std = np.std(total_hamming_pe6_global, axis=0)
    total_hamming_pe6_global = np.mean(total_hamming_pe6_global, axis=0)

    x = np.linspace(0, 1000, 20)

    # Local Regions

    plt.plot(x, np.array(total_hamming_xy_local), label='Coordinates', color=colors[3])
    plt.fill_between(x, np.array(total_hamming_xy_local)+np.array(total_hamming_xy_local_std), np.array(total_hamming_xy_local)-np.array(total_hamming_xy_local_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[3])

    plt.plot(x, np.array(total_hamming_pe3_local), label='L=3', color=colors[0])
    plt.fill_between(x, np.array(total_hamming_pe3_local)+np.array(total_hamming_pe3_local_std), np.array(total_hamming_pe3_local)-np.array(total_hamming_pe3_local_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True,color=colors[0] )

    plt.plot(x, np.array(total_hamming_pe6_local), label='L=6', color=colors[1])
    plt.fill_between(x, np.array(total_hamming_pe6_local)+np.array(total_hamming_pe6_local_std), np.array(total_hamming_pe6_local)-np.array(total_hamming_pe6_local_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True,color=colors[1] )

    plt.legend()
    plt.show()

    # Global hamming distance

    plt.plot(x, np.array(total_hamming_xy_global), label='Coordinates', color=colors[3])
    plt.fill_between(x, np.array(total_hamming_xy_global)+np.array(total_hamming_xy_global_std), np.array(total_hamming_xy_global)-np.array(total_hamming_xy_global_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[3])

    plt.plot(x, np.array(total_hamming_pe3_global), label='L=3', color=colors[0])
    plt.fill_between(x, np.array(total_hamming_pe3_global)+np.array(total_hamming_pe3_global_std), np.array(total_hamming_pe3_global)-np.array(total_hamming_pe3_global_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[0])

    plt.plot(x, np.array(total_hamming_pe6_global), label='L=6', color=colors[1])
    plt.fill_between(x, np.array(total_hamming_pe6_global)+np.array(total_hamming_pe6_global_std), np.array(total_hamming_pe6_global)-np.array(total_hamming_pe6_global_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[1])

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
