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

device = 'cuda:0'

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

    def forward(self, x):

        out_1 = self.relu(self.conv1(x))
        out_2 = self.relu(self.conv2(out_1))
        out_3 = self.relu(self.conv3(out_2))

        flattened_out = out_3.view(out_3.shape[0], -1)

        out_4 = self.relu(self.fc1(flattened_out))
        out_5 = self.fc2(out_4)

        return out_5

def MNIST_Dataset(batch_size):

    # Define the transforms to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),   # convert the PIL Image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # normalize the tensor with mean and std
    ])

    # Load the MNIST dataset with train, validation, and test splits
    mnist_trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # Define the data loaders for the MNIST dataset
    mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    mnist_testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)
    mnist_fullbatch = torch.utils.data.DataLoader(mnist_trainset, batch_size=60000, shuffle=True)

    return mnist_fullbatch, mnist_trainloader, mnist_testloader

def CIFAR10_Dataset(batch_size):

    # Define the transforms to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),   # convert the PIL Image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # normalize the tensor with mean and std
    ])

    # Load the CIFAR-10 dataset with train, validation, and test splits
    cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    cifar_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    # Define the data loaders for the CIFAR-10 dataset
    cifar_trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size, shuffle=True)
    cifar_testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=batch_size, shuffle=True)
    cifar_fullbatch = torch.utils.data.DataLoader(cifar_trainset, batch_size=50000, shuffle=True)

    return cifar_fullbatch, cifar_trainloader, cifar_testloader

def CIFAR100_Dataset(batch_size):

    # Define the transforms to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),   # convert the PIL Image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # normalize the tensor with mean and std
    ])

    # Load the CIFAR-10 dataset with train, validation, and test splits
    cifar_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    cifar_testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    # Define the data loaders for the CIFAR-10 dataset
    cifar_trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=batch_size, shuffle=True)
    cifar_testloader = torch.utils.data.DataLoader(cifar_testset, batch_size=batch_size, shuffle=True)
    cifar_fullbatch = torch.utils.data.DataLoader(cifar_trainset, batch_size=50000, shuffle=True)

    return cifar_fullbatch, cifar_trainloader, cifar_testloader

def FashionMNIST_Dataset(batch_size):

    # Define the transforms to be applied to the data
    transform = transforms.Compose([
        transforms.ToTensor(),   # convert the PIL Image to a tensor
        transforms.Normalize((0.5,), (0.5,))  # normalize the tensor with mean and std
    ])

    # Load the ImageNet dataset with train, validation, and test splits
    fashion_trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fashion_testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    # Define the data loaders for the ImageNet dataset
    fashion_trainloader = torch.utils.data.DataLoader(fashion_trainset, batch_size=batch_size, shuffle=True)
    fashion_testloader = torch.utils.data.DataLoader(fashion_testset, batch_size=batch_size, shuffle=True)
    fashion_fullbatch = torch.utils.data.DataLoader(fashion_trainset, batch_size=60000, shuffle=True)

    return fashion_fullbatch, fashion_trainloader, fashion_testloader

def train_model(opt, model, train_loader, test_loader, type='mlp'):

    # Build loss
    loss_fn = nn.CrossEntropyLoss()
    # Build optim
    optim = torch.optim.Adam(model.parameters(), lr=opt.LR)

    # Rec
    training_loss = []
    testing_loss = []
    best_test_acc = 0

    model.train()
    # Loop! 
    for iter_num in range(opt.NUM_ITER):
        running_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optim.zero_grad()
            if type == 'mlp':
                y = model(data.view(data.shape[0], -1))
            elif type == 'cnn':
                y = model(data)
            loss = loss_fn(y, target)
            running_loss += loss.item()
            loss.backward()
            optim.step()
        
        # epoch loss
        running_loss /= len(train_loader.dataset)
        print(f'\nEpoch {iter_num} Training Loss: {running_loss}')
        training_loss.append(running_loss)

        with torch.no_grad():
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                optim.zero_grad()
                if type == 'mlp':
                    y = model(data.view(data.shape[0], -1))
                elif type == 'cnn':
                    y = model(data)

                pred = y.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                loss = loss_fn(y, target)
                test_loss += loss.item()
        
        # get testing accuracy 
        test_acc = 100 * correct / len(test_loader.dataset)
        print(f'Epoch {iter_num} Testing Accuracy: {test_acc}')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        # get testing loss
        test_loss /= len(test_loader.dataset)
        print(f'Epoch {iter_num} Testing Loss: {test_loss}')
        testing_loss.append(test_loss)
    
    return training_loss, testing_loss, best_test_acc

def main():

    opt = Namespace()
    # Model Training
    opt.LR = 0.001              # <--- LR for the optimizer. 
    opt.NUM_ITER = 100         # <--- Number of training iterations
    opt.batch_size = 512         # <--- batch size

    datasets = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

    training_loss_cnn = {}
    training_loss_mlp = {}
    
    test_loss_cnn = {}
    test_loss_mlp = {}

    for data in datasets:

        training_loss_cnn[f'{data}'] = []
        training_loss_mlp[f'{data}'] = []

        test_loss_cnn[f'{data}'] = []
        test_loss_mlp[f'{data}'] = []
    
    _, mnist_loader, mnist_test = MNIST_Dataset(opt.batch_size)
    _, fashion_loader, fashion_test = FashionMNIST_Dataset(opt.batch_size)
    _, cifar_loader, cifar_test = CIFAR10_Dataset(opt.batch_size)
    _, cifar100_loader, cifar100_test = CIFAR100_Dataset(opt.batch_size)

    ######################################## MNIST ###################################

    model = Net(784, 512, 10).to(device)
    cnn = CNN(1, 10, 384).to(device)

    training_loss, testing_loss, mlp_acc_mnist = train_model(opt, model, mnist_loader, mnist_test, type='mlp')

    training_loss_mlp['mnist'].append(training_loss)
    test_loss_mlp['mnist'].append(testing_loss)

    training_loss, testing_loss, cnn_acc_mnist = train_model(opt, cnn, mnist_loader, mnist_test, type='cnn')

    training_loss_cnn['mnist'].append(training_loss)
    test_loss_cnn['mnist'].append(testing_loss)

    ######################################## Fashion MNIST ###################################

    model = Net(784, 512, 10).to(device)
    cnn = CNN(1, 10, 384).to(device)

    training_loss, testing_loss, mlp_acc_fashion = train_model(opt, model, fashion_loader, fashion_test, type='mlp')

    training_loss_mlp['fashion_mnist'].append(training_loss)
    test_loss_mlp['fashion_mnist'].append(testing_loss)

    training_loss, testing_loss, cnn_acc_fashion = train_model(opt, cnn, fashion_loader, fashion_test, type='cnn')

    training_loss_cnn['fashion_mnist'].append(training_loss)
    test_loss_cnn['fashion_mnist'].append(testing_loss)

    ######################################## CIFAR10 ###################################

    model = Net(3*32*32, 512, 10).to(device)
    cnn = CNN(3, 10, 864).to(device)

    training_loss, testing_loss, mlp_acc_cifar10 = train_model(opt, model, cifar_loader, cifar_test, type='mlp')
    
    training_loss_mlp['cifar10'].append(training_loss)
    test_loss_mlp['cifar10'].append(testing_loss)

    training_loss, testing_loss, cnn_acc_cifar10 = train_model(opt, cnn, cifar_loader, cifar_test, type='cnn')

    training_loss_cnn['cifar10'].append(training_loss)
    test_loss_cnn['cifar10'].append(testing_loss)
    
    ######################################## CIFAR100 ###################################

    model = Net(3*32*32, 512, 100).to(device)
    cnn = CNN(3, 100, 864).to(device)

    training_loss, testing_loss, mlp_acc_cifar100 = train_model(opt, model, cifar100_loader, cifar100_test, type='mlp')

    training_loss_mlp['cifar100'].append(training_loss)
    test_loss_mlp['cifar100'].append(testing_loss)

    training_loss, testing_loss, cnn_acc_cifar100 = train_model(opt, cnn, cifar100_loader, cifar100_test, type='cnn')

    training_loss_cnn['cifar100'].append(training_loss)
    test_loss_cnn['cifar100'].append(testing_loss)

    print(f'Test Accuracy mnist mlp: {mlp_acc_mnist}')
    print(f'Test Accuracy mnist cnn: {cnn_acc_mnist}')

    print(f'Test Accuracy fashion mnist mlp: {mlp_acc_fashion}')
    print(f'Test Accuracy fashion mnist cnn: {cnn_acc_fashion}')

    print(f'Test Accuracy cifar10 mlp: {mlp_acc_cifar10}')
    print(f'Test Accuracy cifar10 cnn: {cnn_acc_cifar10}')

    print(f'Test Accuracy cifar100 mlp: {mlp_acc_cifar100}')
    print(f'Test Accuracy cifar100 cnn: {cnn_acc_cifar100}')

    ####################################### Begin Plots ###############################

    x = np.linspace(0, 100, 100)

    # Get mean and std across images
    for data in datasets:

        training_loss_mlp[f'{data}'] = np.squeeze(np.array(training_loss_mlp[f'{data}']))
        test_loss_mlp[f'{data}'] = np.squeeze(np.array(test_loss_mlp[f'{data}']))
        training_loss_cnn[f'{data}'] = np.squeeze(np.array(training_loss_mlp[f'{data}']))
        test_loss_cnn[f'{data}'] = np.squeeze(np.array(test_loss_cnn[f'{data}']))

    # Training and testing plots

    fig1, ax1 = plt.subplots()
    ax1.plot(x, training_loss_mlp[f'mnist'], label='MNIST MLP', linewidth=2)
    ax1.plot(x, training_loss_mlp[f'fashion_mnist'], label='Fashion MNIST MLP', linewidth=2)
    ax1.plot(x, training_loss_mlp[f'cifar10'], label='CIFAR10 MLP', linewidth=2)
    ax1.plot(x, training_loss_mlp[f'cifar100'], label='CIFAR100 MLP', linewidth=2)
    ax1.legend(loc='best')
    fig1.savefig('training_plots_real_data/training_losses_mlp')

    fig2, ax2 = plt.subplots()
    ax2.plot(x, training_loss_cnn[f'mnist'], label='MNIST CNN', linewidth=2)
    ax2.plot(x, training_loss_cnn[f'fashion_mnist'], label='Fashion MNIST CNN', linewidth=2)
    ax2.plot(x, training_loss_cnn[f'cifar10'], label='CIFAR10 CNN', linewidth=2)
    ax2.plot(x, training_loss_cnn[f'cifar100'], label='CIFAR100 CNN', linewidth=2)
    ax2.legend(loc='best')
    fig2.savefig('training_plots_real_data/training_losses_cnn')

    fig3, ax3 = plt.subplots()
    ax3.plot(x, test_loss_mlp[f'mnist'], label='MNIST MLP', linewidth=2)
    ax3.plot(x, test_loss_mlp[f'fashion_mnist'], label='Fashion MNIST MLP', linewidth=2)
    ax3.plot(x, test_loss_mlp[f'cifar10'], label='CIFAR10 MLP', linewidth=2)
    ax3.plot(x, test_loss_mlp[f'cifar100'], label='CIFAR100 MLP', linewidth=2)
    ax3.legend(loc='best')
    fig3.savefig('training_plots_real_data/test_losses_mlp')

    fig4, ax4 = plt.subplots()
    ax4.plot(x, test_loss_cnn[f'mnist'], label='MNIST CNN', linewidth=2)
    ax4.plot(x, test_loss_cnn[f'fashion_mnist'], label='Fashion MNIST CNN', linewidth=2)
    ax4.plot(x, test_loss_cnn[f'cifar10'], label='CIFAR10 CNN', linewidth=2)
    ax4.plot(x, test_loss_cnn[f'cifar100'], label='CIFAR100 CNN', linewidth=2)
    ax4.legend(loc='best')
    fig4.savefig('training_plots_real_data/test_losses_cnn')

if __name__ == '__main__':
    main()