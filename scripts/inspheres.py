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
from ortools.linear_solver import pywraplp
import ortools
import cvxpy as cp

sns.set_style('white')
colors = sns.color_palette('tab10')

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
            return act_1, act_2, act_3, act_4

        return out_5

def set_positive_values_to_one(tensor):
    # Create a mask that is 1 for positive values and 0 for non-positive values
    mask = tensor.gt(0.0)
    # Set all positive values to 1 using the mask
    tensor[mask] = 1
    return tensor

def get_inradius(model, inp, min_x, max_x):

    with torch.no_grad():

        l1_weight, l1_bias = model.l1.weight, model.l1.bias
    
        # Get the activaton pattern
        act1, act2, act3, act4 = model(inp, act=True)
        act1 = set_positive_values_to_one(act1)

    x = cp.Variable((inp.shape[0],))
    r = cp.Variable(1)

    l1_weight, l1_bias = l1_weight.cpu().detach().numpy(), l1_bias.cpu().detach().numpy()
    l1_weight, l1_bias = cp.Constant(l1_weight), cp.Constant(l1_bias)

    # add constraints
    constraints = []

    for i, neuron in enumerate(l1_weight):
        inner_product = cp.sum(cp.multiply(x, neuron))
        if act1[i] == 0:
            constraints.append(inner_product + (r*cp.norm(neuron)) + l1_bias[i] <= 0)
        else:
            constraints.append(inner_product - (r*cp.norm(neuron)) + l1_bias[i] >= 0)

    for i in range(inp.shape[0]):
        constraints.append(min_x + r <= x[i])
        constraints.append(max_x - r >= x[i])

    objective = cp.Maximize(r)
    problem = cp.Problem(objective, constraints)
    result = problem.solve()
    
    return r.value[0]

def get_data(image, encoding, L=10, batch_size=2048, negative=False, shuffle=True):

    # Get the training image
    trainimg= image
    trainimg = trainimg / 255.0  
    H, W, C = trainimg.shape

    # Get the encoding
    PE = Positional_Encoding(trainimg, encoding, training=True)
    inp_batch, inp_target, ind_vals = PE.get_dataset(L, negative=negative)

    inp_batch, inp_target = torch.Tensor(inp_batch), torch.Tensor(inp_target)
    inp_batch, inp_target = inp_batch.to('cuda:1'), inp_target.to('cuda:1')

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

def train(model, optim, criterion, im, encoding, L, args, negative=False):

    # Get the training data
    train_inp_batch, train_inp_target = get_data(image=im, encoding=encoding, L=L, batch_size=args.batch_size, negative=negative)
    inp_batch, inp_target = get_data(image=im, encoding=encoding, L=L, batch_size=1, shuffle=False, negative=negative)

    inp_batch = inp_batch.squeeze()
    min_x = -1
    max_x = 1

    inspheres = []
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

    # Get inradius
    for i in range(1000):
        rand_inp = 2 * torch.rand(inp_batch.shape[-1]).to('cuda:1') - 1 
        radius = get_inradius(model, rand_inp, min_x, max_x)
        #print(f'Iteration {i}: {radius}')
        inspheres.append(radius)

    return inspheres

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=512, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=2500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8192, help='make a training and testing set')
    parser.add_argument('--print_loss', type=bool, default=True, help='print training loss')

    args = parser.parse_args()

    # change image to any in image_demonstration
    test_data = np.load('test_data_div2k.npy')
    test_data = test_data[:5]

    L_vals = [4, 8, 12, 16, 20]
    # lists for all confusion for each image

    averaged_local_confusion_pe = {}
    averaged_global_confusion_pe = {}

    for l in L_vals:
        averaged_local_confusion_pe[f'{l}_val'] = []
        averaged_global_confusion_pe[f'{l}_val'] = []

    # Go through each image individually
    for im in test_data:

        #################################### Sin Cos #############################################

        print("\nBeginning Positional Encoding Training...")
        for l in L_vals:

            # Set up pe network
            model_pe = Net(l*4, args.neurons).to('cuda:1')
            optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)
            criterion = nn.MSELoss()

            inradius = train(model_pe, optim_pe, criterion, im, 'sin_cos', l, args)

            averaged_global_confusion_pe[f'{l}_val'].append(inradius)
    
    ##################################### Plotting Data ###########################################

    for l in L_vals:
        averaged_global_confusion_pe[f'{l}_val'] = np.array(averaged_global_confusion_pe[f'{l}_val']).flatten()

    fig1, ax1 = plt.subplots()
    sns.kdeplot(data=averaged_global_confusion_pe[f'4_val'], fill=True, label='Encoding L=4', linewidth=2)
    sns.kdeplot(data=averaged_global_confusion_pe[f'8_val'], fill=True, label='Encoding L=8', linewidth=2)
    sns.kdeplot(data=averaged_global_confusion_pe[f'12_val'], fill=True, label='Encoding L=12', linewidth=2)
    sns.kdeplot(data=averaged_global_confusion_pe[f'16_val'], fill=True, label='Encoding L=16', linewidth=2)
    sns.kdeplot(data=averaged_global_confusion_pe[f'20_val'], fill=True, label='Encoding L=20', linewidth=2)
    ax1.set_xlim(0, .2)
    ax1.legend(loc='best')
    fig1.savefig(f'insphere_im/inspheres_l1_{args.epochs}')

if __name__ == '__main__':
    main()