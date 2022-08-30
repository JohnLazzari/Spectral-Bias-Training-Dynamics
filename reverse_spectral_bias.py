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

def get_data(image, encoding, L=10, batch_size=2048, RFF=False, shuffle=True, reverse=False):

    # Get the training image
    trainimg= image
    trainimg = trainimg / 255.0  
    H, W, C = trainimg.shape

    # Get the encoding
    PE = Positional_Encoding(trainimg, encoding, training=True)
    inp_batch, inp_target, ind_vals = PE.get_dataset(L, RFF=False)

    inp_batch, inp_target = torch.Tensor(inp_batch), torch.Tensor(inp_target)
    orig_batch, orig_target = torch.Tensor(inp_batch), torch.Tensor(inp_target)
    inp_batch, inp_target = inp_batch.to('cuda:0'), inp_target.to('cuda:0')
    orig_batch, orig_target = orig_batch.to('cuda:0'), orig_target.to('cuda:0')

    if reverse:

        reversed = []
        orig = []

        inp_batch = torch.reshape(inp_batch, [256, 256, 2])
        inp_target = torch.reshape(inp_target, [256, 256, 2])

        for x in range(H):
            for y in range(W):
                # square grid for training set
                if x % 16 == 0 and y % 16 == 0:
                    reversed.append((inp_batch[x][y], inp_target[x][y]))

        index = 0
        reversed_target = inp_target[150:166][150:166]
        reversed_target = torch.flatten(reversed_target)
        for x in range(H):
            for y in range(W):
                if inp_batch[x][y] == reversed[index][0]:
                    inp_target[x][y] = reversed_target[index]
                    

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

    model_raw = Net(2, 128).to('cuda:0')
    optim_raw = torch.optim.Adam(model_raw.parameters(), lr=.001)
    criterion = nn.MSELoss()
    epochs = 1000

    im = Image.open(f'dataset/fractal.jpg')
    im2arr = np.array(im) 
    batch_size = 2048
    display = True

    train_inp_batch, train_inp_target = get_data(image=im2arr, encoding='raw_xy', L=0, batch_size=batch_size)

    raw_losses = []
    for epoch in range(epochs):

        running_loss = 0

        for i, pixel in enumerate(train_inp_batch):
            optim_raw.zero_grad()
            output = model_raw(pixel)
            loss = criterion(output, train_inp_target[i])
            running_loss += loss.item()
            loss.backward()
            optim_raw.step()
        epoch_loss = running_loss / (256*256 / batch_size)
        print('Loss at epoch {}: {}'.format(epoch, epoch_loss))
        raw_losses.append(epoch_loss)

    if display:
        with torch.no_grad():
            # This is to actually pass the image through the network
            im = Image.open(f'dataset/fractal.jpg')
            im2arr = np.array(im) 

            # Get the encoding
            testimg = im2arr 
            testimg = testimg / 255.0  
            H, W, C = testimg.shape

            PE = Positional_Encoding(testimg, 'raw_xy', training=False)

            # Get data in encoded format
            inp_batch, inp_target, ind_vals = PE.get_dataset(L=0)
            inp_batch = torch.Tensor(inp_batch).to('cuda:0')

            output = model_raw(inp_batch)
            output = output.cpu()
            # Display the image from the model
            predicted_image = np.zeros_like(testimg)

            # preprocess image before displaying
            indices = ind_vals.astype('int')
            indices = indices[:, 1] * testimg.shape[1] + indices[:, 0]

            np.put(predicted_image[:, :, 0], indices, np.clip(output[:, 0], 0, 1))
            np.put(predicted_image[:, :, 1], indices, np.clip(output[:, 1], 0, 1))
            np.put(predicted_image[:, :, 2], indices, np.clip(output[:, 2], 0, 1))

            plt.imshow(predicted_image)
            plt.show()

    torch.save(model_raw.state_dict(), 'raw_xy_reversed.pth')

if __name__ == "__main__":
    main()