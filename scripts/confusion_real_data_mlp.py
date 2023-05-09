import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import transforms

import torchmetrics
import argparse
from argparse import Namespace

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
colors = sns.color_palette('tab10')
device = 'cuda:1'

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

    def forward(self, x):

        out_1 = self.relu(self.l1(x))
        out_2 = self.relu(self.l2(out_1))
        out_3 = self.relu(self.l3(out_2))
        out_4 = self.relu(self.l4(out_3))
        out_5 = self.l5(out_4)

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

def compute_confusion(model, optim, x_0, x_1, y_0, y_1):
    # Get confusion between the gradients for both inputs
    criterion = nn.CrossEntropyLoss()
    cos = torch.nn.CosineSimilarity(dim=0)

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
    confusion = cos(grad_1, grad_2).cpu()

    return confusion.item()

def confusion_within_regions(model, optim, inp_batch, inp_target, iterations):

    print('Starting confusion within local regions...')
    confusion_local = []
    # reshape the inputs to be in image format again
    train_inputs = inp_batch
    train_targets = inp_target

    for i in range(iterations):

        num = np.random.randint(0, train_inputs.shape[0])

        data_point = train_inputs[num]
        data_target = train_targets[num]

        points = torch.cat((train_inputs[:num], train_inputs[num+1:]), dim=0)
        targets = torch.cat((train_targets[:num], train_targets[num+1:]), dim=0)
        
        # Get the closest datapoint
        distances = torch.linalg.norm(data_point - points, dim=1).squeeze()

        # Need to ensure they are of different classes! Avoid mostly showing highly correlated gradients
        neighbors = points[torch.topk(distances, k=5000, largest=False)[1].tolist()]
        neighbors_target = targets[torch.topk(distances, k=5000, largest=False)[1].tolist()]

        data_point_label = data_target.item()
        indices_different_classes = torch.where(neighbors_target == data_point_label)[0].tolist()
        
        # Remove the elements at the specified indices
        neighbors_different_classes = torch.stack([neighbors[i] for i in range(neighbors.shape[0]) if i not in indices_different_classes], dim=0).to(device)
        targets_different_classes = torch.tensor([neighbors_target[i] for i in range(neighbors_target.shape[0]) if i not in indices_different_classes]).to(device)

        for j in range(25):

            confusion = compute_confusion(model, optim, data_point, neighbors_different_classes[j], data_target, targets_different_classes[j])
            confusion_local.append(confusion)

    return confusion_local

def confusion_between_regions(model, optim, inp_batch, inp_target, iterations):

    print('Starting confusion across input space...')
    confusion_between = []

    train_inputs = inp_batch
    train_targets = inp_target

    for i in range(iterations):

        num = np.random.randint(0, train_inputs.shape[0])
        data_point = train_inputs[num]
        data_target = train_targets[num]
        
        # Get the furthest datapoint
        distances = torch.linalg.norm(data_point - train_inputs, dim=1).squeeze()
        distant_points = train_inputs[torch.topk(distances, k=25)[1].tolist()]
        distant_points_target = train_targets[torch.topk(distances, k=25)[1].tolist()]

        for j in range(25):

            confusion = compute_confusion(model, optim, data_point, distant_points[j], data_target, distant_points_target[j])
            confusion_between.append(confusion)

    return confusion_between

def train_model(opt, model, train_data, train_targets, train_loader):

    # Build loss
    loss_fn = nn.CrossEntropyLoss()
    # Build optim
    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)

    model.train()
    # Loop! 
    for iter_num in range(opt.NUM_ITER):

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optim.zero_grad()
            y = model(data.view(data.shape[0], -1))
            loss = loss_fn(y, target)
            loss.backward()
            optim.step()
    
    # Maybe try 2500 inputs and 25 samples for each 
    confusion_neighbors = confusion_within_regions(model, optim, train_data, train_targets, 2500)
    confusion_distant = confusion_between_regions(model, optim, train_data, train_targets, 2500)

    return confusion_neighbors, confusion_distant

def main():

    opt = Namespace()
    # Model Training
    opt.LR = 0.001              # <--- LR for the optimizer. 
    opt.NUM_ITER = 25         # <--- Number of training iterations
    opt.batch_size = 512         # <--- batch size
    opt.dif_classes = False

    datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

    distant_confusion = {}
    neighbors_confusion = {}

    for data in datasets:

        distant_confusion[f'{data}'] = []
        neighbors_confusion[f'{data}'] = []
    
    mnist_data, mnist_loader = MNIST_Dataset(opt.batch_size)
    fashion_data, fashion_loader = FashionMNIST_Dataset(opt.batch_size)
    cifar_data, cifar_loader = CIFAR10_Dataset(opt.batch_size)
    cifar100_data, cifar100_loader = CIFAR100_Dataset(opt.batch_size)

    ######################################## MNIST ###################################

    mnist_data = next(iter(mnist_data))
    mnist_points = mnist_data[0].view(mnist_data[0].shape[0], -1).to(device)
    mnist_targets = mnist_data[1].to(device)

    model = Net(784, 512, 10).to(device)

    confusion_neighbor, confusion_distant = train_model(opt, model, mnist_points, mnist_targets, mnist_loader)

    distant_confusion[f'mnist'].append(confusion_distant)
    neighbors_confusion[f'mnist'].append(confusion_neighbor)
    mnist_points = mnist_points.detach()
    mnist_targets = mnist_targets.detach()

    ######################################## Fashion MNIST ###################################

    fashion_data = next(iter(fashion_data))
    fashion_points = fashion_data[0].view(fashion_data[0].shape[0], -1).to(device)
    fashion_targets = fashion_data[1].to(device)

    model = Net(784, 512, 10).to(device)

    confusion_neighbor, confusion_distant = train_model(opt, model, fashion_points, fashion_targets, fashion_loader)

    distant_confusion[f'fashion_mnist'].append(confusion_distant)
    neighbors_confusion[f'fashion_mnist'].append(confusion_neighbor)
    fashion_points = fashion_points.detach()
    fashion_targets = fashion_targets.detach()

    ######################################## CIFAR10 ###################################

    cifar_data = next(iter(cifar_data))
    cifar_points = cifar_data[0].view(cifar_data[0].shape[0], -1).to(device)
    cifar_targets = cifar_data[1].to(device)

    model = Net(3*32*32, 512, 10).to(device)

    confusion_neighbor, confusion_distant = train_model(opt, model, cifar_points, cifar_targets, cifar_loader)

    distant_confusion[f'cifar10'].append(confusion_distant)
    neighbors_confusion[f'cifar10'].append(confusion_neighbor)
    cifar_points = cifar_points.detach()
    cifar_targets = cifar_targets.detach()

    ######################################## CIFAR100 ###################################

    cifar100_data = next(iter(cifar100_data))
    cifar100_points = cifar100_data[0].view(cifar100_data[0].shape[0], -1).to(device)
    cifar100_targets = cifar100_data[1].to(device)

    model = Net(3*32*32, 512, 100).to(device)

    confusion_neighbor, confusion_distant = train_model(opt, model, cifar100_points, cifar100_targets, cifar100_loader)

    distant_confusion[f'cifar100'].append(confusion_distant)
    neighbors_confusion[f'cifar100'].append(confusion_neighbor)
    cifar100_points = cifar100_points.detach()
    cifar100_targets = cifar100_targets.detach()

    ###################################### Begin Plots #######################################

    for data in datasets:

        distant_confusion[f'{data}'] = np.squeeze(np.array(distant_confusion[f'{data}']))
        neighbors_confusion[f'{data}'] = np.squeeze(np.array(neighbors_confusion[f'{data}']))

    fig1, ax1 = plt.subplots()
    sns.kdeplot(data=distant_confusion[f'mnist'], fill=True, label='MNIST', linewidth=2, color=colors[0])
    sns.kdeplot(data=distant_confusion[f'fashion_mnist'], fill=True, label='Fashion MNIST', linewidth=2, color=colors[3])
    sns.kdeplot(data=distant_confusion[f'cifar10'], fill=True, label='CIFAR10', linewidth=2, color=colors[4])
    sns.kdeplot(data=distant_confusion[f'cifar100'], fill=True, label='CIFAR100', linewidth=2, color=colors[1])
    sns.despine()
    ax1.set_ylim(0, 10)
    ax1.set_xlim(-1, 1)
    ax1.legend(loc='best')
    fig1.savefig('confusion_real_data/confusion_distant_inputs_difclass_epoch25')

    # Local hamming distances plot
    fig2, ax2 = plt.subplots()
    sns.kdeplot(data=neighbors_confusion[f'mnist'], fill=True, label='MNIST', linewidth=2, color=colors[0])
    sns.kdeplot(data=neighbors_confusion[f'fashion_mnist'], fill=True, label='Fashion MNIST', linewidth=2, color=colors[3])
    sns.kdeplot(data=neighbors_confusion[f'cifar10'], fill=True, label='CIFAR10', linewidth=2, color=colors[4])
    sns.kdeplot(data=neighbors_confusion[f'cifar100'], fill=True, label='CIFAR100', linewidth=2, color=colors[1])
    sns.despine()
    ax2.set_ylim(0, 10)
    ax2.set_xlim(-1, 1)
    ax2.legend(loc='best')
    fig2.savefig('confusion_real_data/confusion_neighboring_inputs_difclass_epoch25')

if __name__ == '__main__':
    main()