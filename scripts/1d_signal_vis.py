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

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')

    args = parser.parse_args()
    runs = [1, 2, 3, 4, 5]

    for run in runs:

        # Define model and other training parameters
        model_raw = Net(1, 128).to(device)
        optim_raw = optim.Adam(model_raw.parameters(), lr=.001)
        
        model_neg = Net(1, 128).to(device)
        optim_neg = optim.Adam(model_neg.parameters(), lr=.001)

        model_pe3 = Net(8, 128).to(device)
        optim_pe3 = optim.Adam(model_pe3.parameters(), lr=.001)

        model_pe6 = Net(14, 128).to(device)
        optim_pe6 = optim.Adam(model_pe6.parameters(), lr=.001)

        criterion = nn.MSELoss()

        yt, x = make_phased_waves(opt)
        neg_x = np.arange(-1, 1, 1./(opt.N/2))

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

        neg_x = torch.Tensor(neg_x).to(device)
        neg_x = neg_x.unsqueeze(dim=1)

        encoding_3 = torch.Tensor(encoding_3).to(device)
        encoding_6 = torch.Tensor(encoding_6).to(device)

        target = torch.Tensor(yt).to(device)
        target = target.unsqueeze(dim=1)

        # Start training loop, will need to evaluate the network at every other iteration to determine the spectrogram and
        # evaluate through training
        train(model_raw, optim_raw, criterion, x, target, args)
        train(model_neg, optim_neg, criterion, neg_x, target, args)
        train(model_pe3, optim_pe3, criterion, encoding_3, target, args)
        train(model_pe6, optim_pe6, criterion, encoding_6, target, args)

        with torch.no_grad():
            y_xy = model_raw(x)
            y_neg = model_neg(neg_x)
            y_3 = model_pe3(encoding_3)
            y_6 = model_pe6(encoding_6)

        x = np.linspace(0, 1, 2048)

        plt.figure(figsize=(12,5))
        plt.plot(x, yt, label='Ground Truth', color=colors[7])
        plt.plot(x, y_xy.cpu().numpy(), label='Coordinates [0,1]', color=colors[3])
        plt.plot(x, y_neg.cpu().numpy(), label='Coordinates [-1,1]', color=colors[4])
        plt.plot(x, y_3.cpu().numpy(), label='Encoding L=3', color=colors[0])
        plt.plot(x, y_6.cpu().numpy(), label='Encoding L=6', color=colors[1])
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
