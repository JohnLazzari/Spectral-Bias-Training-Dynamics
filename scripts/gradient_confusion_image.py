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
        self.l5= nn.Linear(hidden, 3)
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

def confusion_local(model, optim, inp_batch, inp_target, iterations):

    print('Getting confusion within local regions...')
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

        for j in range(500):

            ind1 = np.random.randint(0, 2499)
            ind2 = np.random.randint(0, 2499)

            confusion = compute_confusion(model, optim, patch[ind1], patch[ind2], patch_target[ind1], patch_target[ind2])
            confusion_local.append(confusion)

    return confusion_local

def confusion_global(model, optim, inp_batch, inp_target, iterations):

    print('Getting confusion across input space...')
    confusion_between = []
    # reshape the inputs to be in image format again
    shape = inp_batch.shape
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

        confusion = compute_confusion(model, optim, point_1, point_2, point_1_target, point_2_target)
        confusion_between.append(confusion)

    return confusion_between

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

    confusion_in_region = []
    confusion_between_region = []

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

        # This gets confusion at the end of training, specify the number of epochs
        if epoch > args.epochs-2:
            # 100, 50000
            confusion_in_region = confusion_local(model, optim, inp_batch, inp_target, 100)
            confusion_between_region = confusion_global(model, optim, inp_batch, inp_target, 50000)

    return confusion_in_region, confusion_between_region

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=512, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8192, help='make a training and testing set')
    parser.add_argument('--print_loss', type=bool, default=True, help='print training loss')
    parser.add_argument('--negative', type=bool, default=False, help='-1 to 1')
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
            print("\nBeginning Raw XY Training...")
            confusion_within_xy, confusion_between_xy = train(model_raw, optim_raw, criterion, im, 'raw_xy', 0, args)
        
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

                confusion_within_pe, confusion_between_pe = train(model_pe, optim_pe, criterion, im, 'sin_cos', l, args)

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

    fig1, ax1 = plt.subplots()
    sns.kdeplot(data=averaged_global_confusion_pe[f'4_val'], fill=True, label='L=4')
    sns.kdeplot(data=averaged_global_confusion_pe[f'8_val'], fill=True, label='L=8')
    sns.kdeplot(data=averaged_global_confusion_pe[f'16_val'], fill=True, label='L=16')
    sns.kdeplot(data=averaged_global_confusion_xy, fill=True, label='coordinates')
    fig1.legend()
    #fig1.savefig('confusion_images/confusion_global_epoch1000')

    fig2, ax2 = plt.subplots()
    sns.kdeplot(data=averaged_local_confusion_pe[f'4_val'], fill=True, label='L=4')
    sns.kdeplot(data=averaged_local_confusion_pe[f'8_val'], fill=True, label='L=8')
    sns.kdeplot(data=averaged_local_confusion_pe[f'16_val'], fill=True, label='L=16')
    sns.kdeplot(data=averaged_local_confusion_xy, fill=True, label='coordinates')
    fig2.legend()
    #fig2.savefig('confusion_images/confusion_local_epoch1000')

if __name__ == '__main__':
    main()

