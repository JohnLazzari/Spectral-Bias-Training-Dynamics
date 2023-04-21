import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import transforms

import argparse
from argparse import Namespace

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
colors = sns.color_palette('tab10')

class Net(nn.Module):
    def __init__(self, input_dim, hidden, out):
        super(Net, self).__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.l4 = nn.Linear(hidden, hidden)
        self.l5 = nn.Linear(hidden, out)
        self.relu = nn.ReLU()

    def forward(self, x, act=False):

        out_1 = self.relu(self.l1(x))
        out_2 = self.relu(self.l2(out_1))
        out_3 = self.relu(self.l3(out_2))
        out_4 = self.relu(self.l4(out_3))
        out_5 = self.l5(out_4)

        if act:
            pattern = torch.cat((out_1, out_2, out_3, out_4), dim=0).squeeze()
            return pattern

        return out_5

def MNIST_Dataset(batch_size):

    # Define the transforms to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),   # convert the PIL Image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # normalize the tensor with mean and std
    ])

    # Load the MNIST dataset with train, validation, and test splits
    mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Define the data loaders for the MNIST dataset
    mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    mnist_fullbatch = torch.utils.data.DataLoader(mnist_trainset, batch_size=60000, shuffle=True)

    return mnist_fullbatch, mnist_trainloader

def CIFAR10_Dataset(batch_size):

    # Define the transforms to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),   # convert the PIL Image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # normalize the tensor with mean and std
    ])

    # Load the CIFAR-10 dataset with train, validation, and test splits
    cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Define the data loaders for the CIFAR-10 dataset
    cifar_trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size, shuffle=True)
    cifar_fullbatch = torch.utils.data.DataLoader(cifar_trainset, batch_size=50000, shuffle=True)

    return cifar_fullbatch, cifar_trainloader

def CIFAR100_Dataset(batch_size):

    # Define the transforms to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),   # convert the PIL Image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # normalize the tensor with mean and std
    ])

    # Load the CIFAR-10 dataset with train, validation, and test splits
    cifar_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    # Define the data loaders for the CIFAR-10 dataset
    cifar_trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size, shuffle=True)
    cifar_fullbatch = torch.utils.data.DataLoader(cifar_trainset, batch_size=50000, shuffle=True)

    return cifar_fullbatch, cifar_trainloader

def FashionMNIST_Dataset(batch_size):

    # Define the transforms to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),   # convert the PIL Image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # normalize the tensor with mean and std
    ])

    # Load the ImageNet dataset with train, validation, and test splits
    fashion_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    # Define the data loaders for the ImageNet dataset
    fashion_trainloader = torch.utils.data.DataLoader(fashion_trainset, batch_size=batch_size, shuffle=True)
    fashion_fullbatch = torch.utils.data.DataLoader(fashion_trainset, batch_size=60000, shuffle=True)

    return fashion_fullbatch, fashion_trainloader

def set_positive_values_to_one(tensor):
    # Create a mask that is 1 for positive values and 0 for non-positive values
    mask = tensor.gt(0.0)
    # Set all positive values to 1 using the mask
    tensor[mask] = 1
    return tensor

def compute_hamming(model, x_0, x_1):

    # hamming distance
    with torch.no_grad():
        pattern_1 = model(x_0, act=True)
        pattern_1 = set_positive_values_to_one(pattern_1)
        pattern_2 = model(x_1, act=True)
        pattern_2 = set_positive_values_to_one(pattern_2)

    hamming = torch.sum(torch.abs(pattern_1-pattern_2))

    return hamming.item()

def hamming_within_regions(model, optim, inp_batch, iterations):

    print('Starting hamming within local regions...')
    hamming_local = []
    # reshape the inputs to be in image format again
    train_inputs = inp_batch

    for i in range(iterations):

        num = np.random.randint(0, train_inputs.shape[0])
        data_point = train_inputs[num]
        points = torch.cat((train_inputs[:num], train_inputs[num+1:]), dim=0)
        
        # Get the furthest datapoint
        distances = torch.linalg.norm(data_point - points, dim=1).squeeze()
        neighbors = points[torch.topk(distances, k=25, largest=False)[1].tolist()]

        for j in range(25):

            hamming = compute_hamming(model, data_point, neighbors[j])
            hamming_local.append(hamming)

    return hamming_local

def hamming_between_regions(model, optim, inp_batch, iterations):

    print('Starting hamming across input space...')
    hamming_between = []
    # reshape the inputs to be in image format again
    train_inputs = inp_batch

    for i in range(iterations):

        num = np.random.randint(0, train_inputs.shape[0])
        data_point = train_inputs[num]
        
        # Get the furthest datapoint
        distances = torch.linalg.norm(data_point - train_inputs, dim=1).squeeze()
        distant_points = train_inputs[torch.topk(distances, k=25)[1].tolist()]

        for j in range(25):

            hamming = compute_hamming(model, data_point, distant_points[j])
            hamming_between.append(hamming)

    return hamming_between

def train_model(opt, model, train_data, train_loader):

    # Build loss
    loss_fn = nn.CrossEntropyLoss()
    # Build optim
    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)

    # Rec
    mean_hamming_neighbors = []
    mean_hamming_distant = []

    std_hamming_neighbors = []
    std_hamming_distant = []

    model.train()
    # Loop! 
    for iter_num in range(opt.NUM_ITER):

        for data, target in train_loader:
            data, target = data.to('cuda:0'), target.to('cuda:0')

            optim.zero_grad()
            y = model(data.view(data.shape[0], -1))
            loss = loss_fn(y, target)
            loss.backward()
            optim.step()
    
        # mean hamming distance during training
        if iter_num % 2 == 0:

            hamming_neighbors = hamming_within_regions(model, optim, train_data, 2500)
            mean_hamming_neighbors.append(np.mean(np.array(hamming_neighbors)))
            std_hamming_neighbors.append(np.std(np.array(hamming_neighbors)))

            hamming_distant = hamming_between_regions(model, optim, train_data, 2500)
            mean_hamming_distant.append(np.mean(np.array(hamming_distant)))
            std_hamming_distant.append(np.std(np.array(hamming_distant)))

    return mean_hamming_neighbors, mean_hamming_distant, std_hamming_neighbors, std_hamming_distant

def main():

    opt = Namespace()
    # Model Training
    opt.LR = 0.001              # <--- LR for the optimizer. 
    opt.NUM_ITER = 100         # <--- Number of training iterations
    opt.batch_size = 512         # <--- batch size

    datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

    distant_hamming = {}
    neighbors_hamming = {}

    distant_hamming_std = {}
    neighbors_hamming_std = {}

    for data in datasets:

        distant_hamming[f'{data}'] = []
        neighbors_hamming[f'{data}'] = []

        distant_hamming_std[f'{data}'] = []
        neighbors_hamming_std[f'{data}'] = []
    
    mnist_data, mnist_loader = MNIST_Dataset(opt.batch_size)
    fashion_data, fashion_loader = FashionMNIST_Dataset(opt.batch_size)
    cifar_data, cifar_loader = CIFAR10_Dataset(opt.batch_size)
    cifar100_data, cifar100_loader = CIFAR100_Dataset(opt.batch_size)

    ######################################## MNIST ###################################

    mnist_data = next(iter(mnist_data))
    mnist_data = mnist_data[0].view(mnist_data[0].shape[0], -1).to('cuda:0')

    model = Net(784, 512, 10).to('cuda:0')

    mean_hamming_neighbor, mean_hamming_distant, std_hamming_neighbor, std_hamming_distant = train_model(opt, model, mnist_data, mnist_loader)

    distant_hamming[f'mnist'].append(mean_hamming_distant)
    neighbors_hamming[f'mnist'].append(mean_hamming_neighbor)

    distant_hamming_std[f'mnist'].append(std_hamming_distant)
    neighbors_hamming_std[f'mnist'].append(std_hamming_neighbor)
    mnist_data = mnist_data.detach()

    ######################################## Fashion MNIST ###################################

    fashion_data = next(iter(fashion_data))
    fashion_data = fashion_data[0].view(fashion_data[0].shape[0], -1).to('cuda:0')

    model = Net(784, 512, 10).to('cuda:0')

    mean_hamming_neighbor, mean_hamming_distant, std_hamming_neighbor, std_hamming_distant = train_model(opt, model, fashion_data, fashion_loader)

    distant_hamming[f'fashion_mnist'].append(mean_hamming_distant)
    neighbors_hamming[f'fashion_mnist'].append(mean_hamming_neighbor)

    distant_hamming_std[f'fashion_mnist'].append(std_hamming_distant)
    neighbors_hamming_std[f'fashion_mnist'].append(std_hamming_neighbor)
    fashion_data = fashion_data.detach()

    ######################################## CIFAR10 ###################################

    cifar_data = next(iter(cifar_data))
    cifar_data = cifar_data[0].view(cifar_data[0].shape[0], -1).to('cuda:0')

    model = Net(3*32*32, 512, 10).to('cuda:0')

    mean_hamming_neighbor, mean_hamming_distant, std_hamming_neighbor, std_hamming_distant = train_model(opt, model, cifar_data, cifar_loader)

    distant_hamming[f'cifar10'].append(mean_hamming_distant)
    neighbors_hamming[f'cifar10'].append(mean_hamming_neighbor)

    distant_hamming_std[f'cifar10'].append(std_hamming_distant)
    neighbors_hamming_std[f'cifar10'].append(std_hamming_neighbor)
    cifar_data = cifar_data.detach()
    
    ######################################## CIFAR100 ###################################

    cifar100_data = next(iter(cifar100_data))
    cifar100_data = cifar100_data[0].view(cifar100_data[0].shape[0], -1).to('cuda:0')

    model = Net(3*32*32, 512, 100).to('cuda:0')

    mean_hamming_neighbor, mean_hamming_distant, std_hamming_neighbor, std_hamming_distant = train_model(opt, model, cifar100_data, cifar100_loader)

    distant_hamming[f'cifar100'].append(mean_hamming_distant)
    neighbors_hamming[f'cifar100'].append(mean_hamming_neighbor)

    distant_hamming_std[f'cifar100'].append(std_hamming_distant)
    neighbors_hamming_std[f'cifar100'].append(std_hamming_neighbor)
    cifar100_data = cifar100_data.detach()

    ####################################### Begin Plots ###############################

    x = np.linspace(0, 100, 50)

    # Get mean and std across images
    for data in datasets:

        distant_hamming[f'{data}'] = np.squeeze(np.array(distant_hamming[f'{data}']))
        distant_hamming_std[f'{data}'] = np.squeeze(np.array(distant_hamming_std[f'{data}']))

        neighbors_hamming[f'{data}'] = np.squeeze(np.array(neighbors_hamming[f'{data}']))
        neighbors_hamming_std[f'{data}'] = np.squeeze(np.array(neighbors_hamming_std[f'{data}']))

    # Global Hamming Distances plot

    fig1, ax1 = plt.subplots()
    ax1.plot(x, distant_hamming[f'mnist'], label='MNIST', linewidth=2, color=colors[0])
    ax1.fill_between(x, np.array(distant_hamming[f'mnist'])+np.array(distant_hamming_std[f'mnist']), np.array(distant_hamming[f'mnist'])-np.array(distant_hamming_std[f'mnist']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[0])

    ax1.plot(x, distant_hamming[f'fashion_mnist'], label='Fashion MNIST', linewidth=2, color=colors[3])
    ax1.fill_between(x, np.array(distant_hamming[f'fashion_mnist'])+np.array(distant_hamming_std[f'fashion_mnist']), np.array(distant_hamming[f'fashion_mnist'])-np.array(distant_hamming_std[f'fashion_mnist']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[3])

    ax1.plot(x, distant_hamming[f'cifar10'], label='CIFAR10', linewidth=2, color=colors[4])
    ax1.fill_between(x, np.array(distant_hamming[f'cifar10'])+np.array(distant_hamming_std[f'cifar10']), np.array(distant_hamming[f'cifar10'])-np.array(distant_hamming_std[f'cifar10']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[4])

    ax1.plot(x, distant_hamming[f'cifar100'], label='CIFAR100', linewidth=2, color=colors[1])
    ax1.fill_between(x, np.array(distant_hamming[f'cifar100'])+np.array(distant_hamming_std[f'cifar100']), np.array(distant_hamming[f'cifar100'])-np.array(distant_hamming_std[f'cifar100']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[1])

    ax1.legend(loc='best')
    fig1.savefig('hamming_real_data/hamming_distant_inputs')

    # Local hamming distances plot

    fig2, ax2 = plt.subplots()
    ax2.plot(x, neighbors_hamming[f'mnist'], label='MNIST', linewidth=2, color=colors[0])
    ax2.fill_between(x, np.array(neighbors_hamming[f'mnist'])+np.array(neighbors_hamming_std[f'mnist']), np.array(neighbors_hamming[f'mnist'])-np.array(neighbors_hamming_std[f'mnist']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[0])

    ax2.plot(x, neighbors_hamming[f'fashion_mnist'], label='Fashion MNIST', linewidth=2, color=colors[3])
    ax2.fill_between(x, np.array(neighbors_hamming[f'fashion_mnist'])+np.array(neighbors_hamming_std[f'fashion_mnist']), np.array(neighbors_hamming[f'fashion_mnist'])-np.array(neighbors_hamming_std[f'fashion_mnist']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[3])

    ax2.plot(x, neighbors_hamming[f'cifar10'], label='CIFAR10', linewidth=2, color=colors[4])
    ax2.fill_between(x, np.array(neighbors_hamming[f'cifar10'])+np.array(neighbors_hamming_std[f'cifar10']), np.array(neighbors_hamming[f'cifar10'])-np.array(neighbors_hamming_std[f'cifar10']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[4])

    ax2.plot(x, neighbors_hamming[f'cifar100'], label='CIFAR100', linewidth=2, color=colors[1])
    ax2.fill_between(x, np.array(neighbors_hamming[f'cifar100'])+np.array(neighbors_hamming_std[f'cifar100']), np.array(neighbors_hamming[f'cifar100'])-np.array(neighbors_hamming_std[f'cifar100']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True, color=colors[1])

    ax2.legend(loc='best')
    fig2.savefig('hamming_real_data/hamming_neighboring_inputs')

if __name__ == '__main__':
    main()