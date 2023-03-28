from enum import unique
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from nerf2D import Positional_Encoding
from torch_network import Net
import random
import torchmetrics
import seaborn as sns
import scipy
from torch import linalg as LA
import itertools
import argparse
from matplotlib import cm

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--L', type=int, default=5, help='frequency of encoding')

    args = parser.parse_args()

    # compute gradients individually for each, not sure best way to do this yet
    im2arr = np.random.randint(0, 255, (64, 64, 3))

    # Get the training image
    trainimg = im2arr / 255.0  
    H, W, C = trainimg.shape

    # Get the encoding
    PE = Positional_Encoding(trainimg, 'sin_cos', training=True)
    encoding, _, _ = PE.get_dataset(args.L, negative=False)

    distances = np.empty((4096, 4096))

    for i in range(4096):
        for j in range(4096):
            dot = np.linalg.norm(encoding[i] - encoding[j])
            distances[i][j] = dot

    plt.imshow(distances, cmap='magma')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()