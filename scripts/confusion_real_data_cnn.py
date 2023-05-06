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

class CNN(nn.Module):
    def __init__(self, inp_channels, classes, flattened):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=inp_channels, out_channels=32,
			kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
			kernel_size=(3, 3), stride=2) 
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96,
			kernel_size=(3, 3), stride=2)
		# initialize first (and only) set of FC => RELU layers
        self.fc1 = nn.Linear(in_features=flattened, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=classes)

        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.relu = nn.ReLU()

    def forward(self, x, act=False):

        out_1 = self.relu(self.conv1(x))
        out_2 = self.relu(self.conv2(out_1))
        out_3 = self.relu(self.conv3(out_2))

        flattened_out = out_3.view(out_3.shape[0], -1)

        out_4 = self.relu(self.fc1(flattened_out))
        out_5 = self.fc2(out_4)

        if act:
            # input should be a 1d tensor
            pattern = torch.cat((out_1.flatten(), out_2.flatten(), out_3.flatten(), out_4.flatten()), dim=0).squeeze()
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

def compute_confusion(model, optim, x_0, x_1, y_0, y_1):
    # Get confusion between the gradients for both inputs
    criterion = nn.CrossEntropyLoss()
    cos = torch.nn.CosineSimilarity(dim=0)

    for param in model.parameters():
        param.grad = None

    output = model(x_0)
    loss_1 = criterion(output, y_0)
    loss_1.backward()
    grad_1 = torch.cat([torch.flatten(param.grad) for param in model.parameters()])

    for param in model.parameters():
        param.grad = None

    output = model(x_1)
    loss_2 = criterion(output, y_1)
    loss_2.backward()
    grad_2 = torch.cat([torch.flatten(param.grad) for param in model.parameters()])

    # get inner products of gradients
    confusion = cos(grad_1, grad_2).cpu()

    return confusion.item()

def confusion_within_regions(model, optim, inp_batch, inp_target, iterations, dif_classes=False):

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
        distances = torch.linalg.norm(data_point.flatten() - points.view(points.shape[0], -1), dim=1).squeeze()

        if dif_classes == True:
            # Need to ensure they are of different classes! Avoid mostly showing highly correlated gradients
            top_neighbors = points[torch.topk(distances, k=5000, largest=False)[1].tolist()]
            top_neighbors_target = targets[torch.topk(distances, k=5000, largest=False)[1].tolist()]

            data_point_label = data_target.item()
            indices_different_classes = torch.where(top_neighbors_target == data_point_label)[0].tolist()
            
            # Remove the elements at the specified indices
            neighbors = torch.stack([top_neighbors[i] for i in range(top_neighbors.shape[0]) if i not in indices_different_classes], dim=0).to(device)
            neighbors_targets = torch.tensor([top_neighbors_target[i] for i in range(top_neighbors_target.shape[0]) if i not in indices_different_classes]).to(device)
        else:
            # Need to ensure they are of different classes! Avoid mostly showing highly correlated gradients
            neighbors = points[torch.topk(distances, k=25, largest=False)[1].tolist()]
            neighbors_targets = targets[torch.topk(distances, k=25, largest=False)[1].tolist()]

        for j in range(25):

            confusion = compute_confusion(model, optim, data_point.unsqueeze(0), neighbors[j].unsqueeze(0), data_target.unsqueeze(0), neighbors_targets[j].unsqueeze(0))
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
        distances = torch.linalg.norm(data_point.flatten() - train_inputs.view(train_inputs.shape[0], -1), dim=1).squeeze()
        distant_points = train_inputs[torch.topk(distances, k=25)[1].tolist()]
        distant_points_target = train_targets[torch.topk(distances, k=25)[1].tolist()]

        for j in range(25):

            confusion = compute_confusion(model, optim, data_point.unsqueeze(0), distant_points[j].unsqueeze(0), data_target.unsqueeze(0), distant_points_target[j].unsqueeze(0))
            confusion_between.append(confusion)

    return confusion_between

def train_model(opt, model, train_data, train_targets, train_loader):

    # Build loss
    loss_fn = nn.CrossEntropyLoss()
    # Build optim
    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)
    confusion_distant = []

    model.train()
    # Loop! 
    for iter_num in range(opt.NUM_ITER):
        running_loss = 0
        num_loops = 0

        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optim.zero_grad()
            y = model(data)
            loss = loss_fn(y, target)
            running_loss += loss.item()
            loss.backward()
            optim.step()
            num_loops = i

        print(f"Training Loss at Epoch {iter_num}: {running_loss/num_loops}")
    
    # Maybe try 2500 inputs and 25 samples for each 
    confusion_neighbors = confusion_within_regions(model, optim, train_data, train_targets, 2500, dif_classes=opt.dif_classes)
    #confusion_distant = confusion_between_regions(model, optim, train_data, train_targets, 2500)

    return confusion_neighbors, confusion_distant

def main():

    opt = Namespace()
    # Model Training
    opt.LR = 0.001              # <--- LR for the optimizer. 
    opt.NUM_ITER = 25         # <--- Number of training iterations
    opt.batch_size = 512         # <--- batch size
    opt.dif_classes = True         # <--- batch size

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

    # TODO
    # See the effect of batch size and whether or not this displays more confusion
    # Find the right epoch to display the confusion densities
    # Change the sampling as stated in train_model function
    # Then repeat the exact same thing for CNNs
    # Actually, change the sampling in the confusion one (and i guess hamming distance) to be the k=25 nearest neighbors corresponding to different classes
    # since many of them might be in the same class for like mnist and fashion mnist, which may be the cause for more positively correlated gradients 
    # for the neighboring inputs. The distant inputs can be kept the same.

    ######################################## MNIST ###################################

    mnist_data = next(iter(mnist_data))
    mnist_points = mnist_data[0].to(device)
    mnist_targets = mnist_data[1].to(device)

    model = CNN(1, 10, 384).to(device)

    confusion_neighbor, confusion_distant = train_model(opt, model, mnist_points, mnist_targets, mnist_loader)

    distant_confusion[f'mnist'].append(confusion_distant)
    neighbors_confusion[f'mnist'].append(confusion_neighbor)
    mnist_points = mnist_points.detach()
    mnist_targets = mnist_targets.detach()

    ######################################## Fashion MNIST ###################################

    fashion_data = next(iter(fashion_data))
    fashion_points = fashion_data[0].to(device)
    fashion_targets = fashion_data[1].to(device)

    model = CNN(1, 10, 384).to(device)

    confusion_neighbor, confusion_distant = train_model(opt, model, fashion_points, fashion_targets, fashion_loader)

    distant_confusion[f'fashion_mnist'].append(confusion_distant)
    neighbors_confusion[f'fashion_mnist'].append(confusion_neighbor)
    fashion_points = fashion_points.detach()
    fashion_targets = fashion_targets.detach()

    ######################################## CIFAR10 ###################################

    cifar_data = next(iter(cifar_data))
    cifar_points = cifar_data[0].to(device)
    cifar_targets = cifar_data[1].to(device)

    model = CNN(3, 10, 864).to(device)

    confusion_neighbor, confusion_distant = train_model(opt, model, cifar_points, cifar_targets, cifar_loader)

    distant_confusion[f'cifar10'].append(confusion_distant)
    neighbors_confusion[f'cifar10'].append(confusion_neighbor)
    cifar_points = cifar_points.detach()
    cifar_targets = cifar_targets.detach()

    ######################################## CIFAR100 ###################################

    cifar100_data = next(iter(cifar100_data))
    cifar100_points = cifar100_data[0].to(device)
    cifar100_targets = cifar100_data[1].to(device)

    model = CNN(3, 100, 864).to(device)

    confusion_neighbor, confusion_distant = train_model(opt, model, cifar100_points, cifar100_targets, cifar100_loader)

    distant_confusion[f'cifar100'].append(confusion_distant)
    neighbors_confusion[f'cifar100'].append(confusion_neighbor)
    cifar100_points = cifar100_points.detach()
    cifar100_targets = cifar100_targets.detach()

    ###################################### Begin Plots #######################################

    for data in datasets:

        #distant_confusion[f'{data}'] = np.squeeze(np.array(distant_confusion[f'{data}']))
        neighbors_confusion[f'{data}'] = np.squeeze(np.array(neighbors_confusion[f'{data}']))

    '''
    fig1, ax1 = plt.subplots()
    sns.kdeplot(data=distant_confusion[f'mnist'], fill=True, label='MNIST', linewidth=2, color=colors[0])
    sns.kdeplot(data=distant_confusion[f'fashion_mnist'], fill=True, label='Fashion MNIST', linewidth=2, color=colors[3])
    sns.kdeplot(data=distant_confusion[f'cifar10'], fill=True, label='CIFAR10', linewidth=2, color=colors[4])
    sns.kdeplot(data=distant_confusion[f'cifar100'], fill=True, label='CIFAR100', linewidth=2, color=colors[1])
    sns.despine()
    ax1.set_ylim(0, 10)
    ax1.set_xlim(-1, 1)
    ax1.legend(loc='best')
    fig1.savefig('confusion_real_data/confusion_distant_inputs_cnn')
    '''

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
    fig2.savefig('confusion_real_data/confusion_neighboring_inputs_cnn_difclass')

if __name__ == '__main__':
    main()