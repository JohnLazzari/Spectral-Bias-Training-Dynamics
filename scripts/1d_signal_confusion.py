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

def compute_confusion(model, optim, x_0, x_1, y_0, y_1):
    # Get confusion between the gradients for both inputs
    criterion = nn.MSELoss()

    optim.zero_grad()
    output = model(x_0)
    loss = criterion(output, y_0)
    loss.backward()

    grad_1 = torch.cat([torch.flatten(model.l1.weight.grad), torch.flatten(model.l2.weight.grad), torch.flatten(model.l3.weight.grad), torch.flatten(model.l4.weight.grad), torch.flatten(model.l5.weight.grad),
                        torch.flatten(model.l1.bias.grad), torch.flatten(model.l2.bias.grad), torch.flatten(model.l3.bias.grad), torch.flatten(model.l4.bias.grad), torch.flatten(model.l5.bias.grad),
                        ])

    optim.zero_grad()
    output = model(x_1)
    loss = criterion(output, y_1)
    loss.backward()
    grad_2 = torch.cat([torch.flatten(model.l1.weight.grad), torch.flatten(model.l2.weight.grad), torch.flatten(model.l3.weight.grad), torch.flatten(model.l4.weight.grad), torch.flatten(model.l5.weight.grad),
                        torch.flatten(model.l1.bias.grad), torch.flatten(model.l2.bias.grad), torch.flatten(model.l3.bias.grad), torch.flatten(model.l4.bias.grad), torch.flatten(model.l5.bias.grad),
                        ])

    # get inner products of gradients
    confusion = torchmetrics.functional.pairwise_cosine_similarity(grad_1.unsqueeze(dim=0), 
                                                                grad_2.unsqueeze(dim=0)).cpu()
    
    optim.zero_grad()

    return confusion.item()

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

            confusion = compute_confusion(model, optim, patch[rand_1], patch[rand_2], patch_target[rand_1], patch_target[rand_2])
            confusion_local.append(confusion)

    return confusion_local

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

        confusion = compute_confusion(model, optim, point_1, point_2, point_1_target, point_2_target)
        confusion_between.append(confusion)

    return confusion_between

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

    confusion_in_region = []
    confusion_between_region = []

    for epoch in range(args.epochs):

        optim.zero_grad()
        out = model(x)
        loss = criterion(out, target)
        print('Epoch {} Loss: {}'.format(epoch, loss.item()))
        loss.backward()
        optim.step()

        if epoch > args.epochs-2:
            confusion_in_region = hamming_within_regions(model, optim, x, target, 50)
            confusion_between_region = hamming_between_regions(model, optim, x, target, 10000)

    return confusion_in_region, confusion_between_region

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--neurons', type=int, default=128, help='Number of neurons per layer')

    args = parser.parse_args()

    runs = [1, 2, 3, 4, 5]

    total_confusion_xy_local = []
    total_confusion_xy_global = []

    total_confusion_pe4_local = []
    total_confusion_pe4_global = []

    total_confusion_pe8_local = []
    total_confusion_pe8_global = []

    for run in runs:

        # Define model and other training parameters
        model_raw = Net(1, args.neurons).to(device)
        optim_raw = optim.Adam(model_raw.parameters(), lr=.001)

        model_pe_4 = Net(8, args.neurons).to(device)
        optim_pe_4 = optim.Adam(model_pe_4.parameters(), lr=.001)

        model_pe_8 = Net(14, args.neurons).to(device)
        optim_pe_8 = optim.Adam(model_pe_8.parameters(), lr=.001)

        criterion = nn.MSELoss()

        yt, x = make_phased_waves(opt)

        # make data into tensors, and also add encoding for data, L=5 

        encoding_4 = np.stack((np.sin(np.pi*x), np.cos(np.pi*x), 
                                np.sin(2*np.pi * x), np.cos(2*np.pi * x),
                                np.sin(2**2 * np.pi * x), np.cos(2**2 * np.pi * x),
                                np.sin(2**3 * np.pi * x), np.cos(2**3 * np.pi * x)), axis=1)

        encoding_8 = np.stack((np.sin(np.pi*x), np.cos(np.pi*x), 
                                np.sin(2*np.pi * x), np.cos(2*np.pi * x),
                                np.sin(2**2 * np.pi * x), np.cos(2**2 * np.pi * x),
                                np.sin(2**3 * np.pi * x), np.cos(2**3 * np.pi * x),
                                np.sin(2**4 * np.pi * x), np.cos(2**4 * np.pi * x),
                                np.sin(2**5 * np.pi * x), np.cos(2**5 * np.pi * x),
                                np.sin(2**6 * np.pi * x), np.cos(2**6 * np.pi * x)), axis=1)
        
        x = torch.Tensor(x).to(device)
        x = x.unsqueeze(dim=1)

        encoding_l4 = torch.Tensor(encoding_4).to(device)
        encoding_l8 = torch.Tensor(encoding_8).to(device)

        target = torch.Tensor(yt).to(device)
        target = target.unsqueeze(dim=1)

        # Start training loop, will need to evaluate the network at every other iteration to determine the spectrogram and
        # evaluate through training
        confusion_within_xy, confusion_between_xy = train(model_raw, optim_raw, criterion, x, target, args)
        confusion_within_pe4, confusion_between_pe4 = train(model_pe_4, optim_pe_4, criterion, encoding_l4, target, args)
        confusion_within_pe8, confusion_between_pe8 = train(model_pe_8, optim_pe_8, criterion, encoding_l8, target, args)

        total_confusion_xy_local.append(confusion_within_xy)
        total_confusion_xy_global.append(confusion_between_xy)

        total_confusion_pe4_local.append(confusion_within_pe4)
        total_confusion_pe4_global.append(confusion_between_pe4)

        total_confusion_pe8_local.append(confusion_within_pe8)
        total_confusion_pe8_global.append(confusion_between_pe8)

    total_confusion_xy_local = np.array(total_confusion_xy_local).flatten()
    total_confusion_xy_global = np.array(total_confusion_xy_global).flatten()

    total_confusion_pe4_local = np.array(total_confusion_pe4_local).flatten()
    total_confusion_pe4_global = np.array(total_confusion_pe4_global).flatten()

    total_confusion_pe8_local = np.array(total_confusion_pe8_local).flatten()
    total_confusion_pe8_global = np.array(total_confusion_pe8_global).flatten()

    sns.kdeplot(total_confusion_xy_local, fill=True, label='Coordinates', color=colors[3])
    sns.kdeplot(total_confusion_pe4_local, fill=True, label='L=3', color=colors[0])
    sns.kdeplot(total_confusion_pe8_local, fill=True, label='L=6', color=colors[1])
    plt.legend()
    plt.show()

    sns.kdeplot(total_confusion_xy_global, fill=True, label='Coordinates', color=colors[3])
    sns.kdeplot(total_confusion_pe4_global, fill=True, label='L=3', color=colors[0])
    sns.kdeplot(total_confusion_pe8_global, fill=True, label='L=6', color=colors[1])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
