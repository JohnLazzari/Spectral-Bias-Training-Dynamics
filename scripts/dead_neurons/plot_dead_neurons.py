from enum import unique
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
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

dead_neurons_01 = np.loadtxt('dead_neurons_01.txt')
dead_neurons_01_std = np.loadtxt('dead_neurons_01_std.txt')

dead_neurons_11 = np.loadtxt('dead_neurons_-11.txt')
dead_neurons_11_std = np.loadtxt('dead_neurons_-11_std.txt')

dead_neurons_22 = np.loadtxt('dead_neurons_-22.txt')
dead_neurons_22_std = np.loadtxt('dead_neurons_-22_std.txt')

dead_neurons_33 = np.loadtxt('dead_neurons_-33.txt')
dead_neurons_33_std = np.loadtxt('dead_neurons_-33_std.txt')

x = np.linspace(0, 1000, 20)

plt.plot(x, dead_neurons_01, label='Coordinates [0,1]', linewidth=2)
plt.fill_between(x, np.array(dead_neurons_01)+np.array(dead_neurons_01_std), np.array(dead_neurons_01)-np.array(dead_neurons_01_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

plt.plot(x, dead_neurons_11, label='Coordinates [-1,1]', linewidth=2)
plt.fill_between(x, np.array(dead_neurons_11)+np.array(dead_neurons_11_std), np.array(dead_neurons_11)-np.array(dead_neurons_11_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

plt.plot(x, dead_neurons_22, label='Coordinates [-2,2]', linewidth=2)
plt.fill_between(x, np.array(dead_neurons_22)+np.array(dead_neurons_22_std), np.array(dead_neurons_22)-np.array(dead_neurons_22_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

plt.plot(x, dead_neurons_33, label='Coordinates [-3,3]', linewidth=2)
plt.fill_between(x, np.array(dead_neurons_33)+np.array(dead_neurons_33_std), np.array(dead_neurons_33)-np.array(dead_neurons_33_std), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

plt.legend()
plt.savefig('dead_relus')