import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import numpy as np
import matplotlib.pyplot as plt
from nerf2D import Positional_Encoding
from PIL import Image
import argparse
import random
import torchmetrics
from math import log10, sqrt
from tqdm import tqdm as tq
import seaborn as sns

class ListModule(object):
    def __init__(self, module, prefix, *args):
        self.module = module
        self.prefix = prefix
        self.num_module = 0
        for new_module in args:
            self.append(new_module)

    def append(self, new_module):
        if not isinstance(new_module, nn.Module):
            raise ValueError('Not a Module')
        else:
            self.module.add_module(self.prefix + str(self.num_module), new_module)
            self.num_module += 1

    def __len__(self):
        return self.num_module

    def __getitem__(self, i):
        if i < 0 or i >= self.num_module:
            raise IndexError('Out of bound')
        return getattr(self.module, self.prefix + str(i))

class maxout_mlp(nn.Module):
    def __init__(self, num_units=2):
        super(maxout_mlp, self).__init__()
        self.fc1_list = ListModule(self, "fc1_")
        self.fc2_list = ListModule(self, "fc2_")
        for _ in range(num_units):
            self.fc1_list.append(nn.Linear(2, 256))
            self.fc2_list.append(nn.Linear(256, 3))

    def forward(self, x): 
        x = self.maxout(x, self.fc1_list)
        x = self.maxout(x, self.fc2_list)
        return torch.sigmoid(x)

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output


class Net(nn.Module):
    def __init__(self, input_dim, hidden):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.l4 = nn.Linear(hidden, hidden)
        self.l5 = nn.Linear(hidden, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.relu(self.l3(out))
        out = self.relu(self.l4(out))
        out = torch.sigmoid(self.l5(out))

        return out

class GaborSIREN(nn.Module):
    def __init__(self, input_dim, hidden):
        super(GaborSIREN, self).__init__()

        self.num_encoding = 1
        self.gamma = [nn.Parameter(torch.distributions.uniform.Uniform(.01, 1).sample((hidden*2,))).to('cuda:0') for i in range(self.num_encoding)]
        self.mu = [nn.Parameter(2 * torch.rand(hidden*2, input_dim) - 1).to('cuda:0') for _ in range(self.num_encoding)]
        self.linear = [nn.Parameter(torch.distributions.uniform.Uniform(-16*np.pi, 16*np.pi).sample((input_dim, hidden*2))).to('cuda:0') for i in range(self.num_encoding)]
        self.bias = [nn.Parameter(torch.distributions.uniform.Uniform(-np.pi, np.pi).sample((hidden*2,))).to('cuda:0') for _ in range(self.num_encoding)]

        self.l1 = nn.Linear(hidden*2, hidden)
        nn.init.normal_(self.l1.weight, 0, np.sqrt(6/(hidden*2)))
        self.l2 = nn.Linear(hidden, hidden)
        nn.init.xavier_uniform_(self.l2.weight)
        #nn.init.normal_(self.l2.weight, 0, np.sqrt(6/hidden))
        self.l3 = nn.Linear(hidden, hidden)
        nn.init.xavier_uniform_(self.l3.weight)
        #nn.init.normal_(self.l3.weight, 0, np.sqrt(6/hidden))
        self.l4 = nn.Linear(hidden, hidden)
        nn.init.xavier_uniform_(self.l4.weight)
        #nn.init.normal_(self.l4.weight, 0, np.sqrt(6/hidden))
        self.output = nn.Linear(hidden, 3)
        nn.init.uniform_(self.output.weight, -np.sqrt(6/hidden), np.sqrt(6/hidden))
        self.relu = nn.ReLU()

    def forward(self, x):

        out = 0
        for i in range(self.num_encoding):
            D = (
                (x ** 2).sum(-1)[..., None]
                + (self.mu[i] ** 2).sum(-1)[None, :]
                - 2 * x @ self.mu[i].T
            )
            out += torch.sin(x @ self.linear[i] + self.bias[i]) * torch.exp(-0.5 * D * self.gamma[i][None, :]**2) 

        out = self.relu(self.l1(out))
        out = self.relu(self.l2(out))
        out = self.relu(self.l3(out))
        out = self.relu(self.l4(out))
        output = torch.sigmoid(self.output(out))

        return output

class GaborNet(nn.Module):
    def __init__(self, input_dim, hidden):
        super(GaborNet, self).__init__()

        self.mu1 = 2 * torch.rand(hidden*2, input_dim) - 1
        self.mu1 = nn.Parameter(self.mu1)

        # ones for text, uniform(0, 1) for div2k
        self.gamma1 = nn.Parameter(torch.distributions.uniform.Uniform(.01, 1).sample((hidden*2,)))
        
        self.sin_network1 = nn.Linear(input_dim, hidden*2) 
        # 16 for div2k, 30 for text
        # learning rate of .0005 for text, .0009 for div2k
        nn.init.uniform_(self.sin_network1.weight, -30*np.pi, 30*np.pi)
        self.sin_network1.bias.data.uniform_(-np.pi, np.pi)

        self.l1 = nn.Linear(hidden*2, hidden)
        nn.init.xavier_uniform_(self.l1.weight)
        self.l2 = nn.Linear(hidden, hidden)
        nn.init.xavier_uniform_(self.l2.weight)
        self.l3 = nn.Linear(hidden, hidden)
        nn.init.xavier_uniform_(self.l3.weight)
        self.l4 = nn.Linear(hidden, hidden)
        nn.init.xavier_uniform_(self.l4.weight)
        self.l5 = nn.Linear(hidden, 3)
        nn.init.xavier_uniform_(self.l5.weight)
        self.relu = nn.ReLU()

    def forward(self, x):

        D1 = (
            (x ** 2).sum(-1)[..., None]
            + (self.mu1 ** 2).sum(-1)[None, :]
            - 2 * x @ self.mu1.T
        )

        encoding = torch.sin(self.sin_network1(x)) * torch.exp(-0.5 * D1 * self.gamma1[None, :]**2) 
        
        first_out = self.relu(self.l1(encoding)) 
        second_out = self.relu(self.l2(first_out))
        out = self.relu(self.l3(second_out))
        out = self.relu(self.l4(out))
        out = torch.sigmoid(self.l5(out))

        return out, first_out, second_out

class SIREN(nn.Module):
    def __init__(self, input_dim, hidden):
        super(SIREN, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden)
        self.l1.weight.data.uniform_(-6/np.sqrt(input_dim), 6/np.sqrt(input_dim))
        self.l2 = nn.Linear(hidden, hidden)
        self.l2.weight.data.uniform_(-6/np.sqrt(hidden), 6/np.sqrt(hidden))
        self.l3 = nn.Linear(hidden, hidden)
        self.l3.weight.data.uniform_(-6/np.sqrt(hidden), 6/np.sqrt(hidden))
        self.l4 = nn.Linear(hidden, hidden)
        self.l4.weight.data.uniform_(-6/np.sqrt(hidden), 6/np.sqrt(hidden))
        self.l5 = nn.Linear(hidden, 3)
        self.l5.weight.data.uniform_(-6/np.sqrt(hidden), 6/np.sqrt(hidden))

    def forward(self, x):
        out = torch.sin(30 * self.l1(x))
        out = torch.sin(self.l2(out))
        out = torch.sin(self.l3(out))
        out = torch.sin(self.l4(out))
        out = torch.sigmoid(self.l5(out))

        return out

def get_data(image, encoding, L=10, batch_size=2048, RFF=False):
    # This is to actually pass the image through the network

    # Get the training image
    trainimg= image
    trainimg = trainimg / 255.0  
    H, W, C = trainimg.shape

    # Get the encoding
    PE = Positional_Encoding(trainimg, encoding, training=True)
    inp_batch, inp_target, ind_vals = PE.get_dataset(L, RFF=RFF)

    inp_batch, inp_target = torch.Tensor(inp_batch), torch.Tensor(inp_target)
    inp_batch, inp_target = inp_batch.to(args.device), inp_target.to(args.device)

    inp_batch = torch.reshape(inp_batch, [H, W, inp_batch.shape[1]])
    inp_target = torch.reshape(inp_target, [H, W, inp_target.shape[1]])

    train_batch = []
    test_batch = []

    for x in range(H):
        for y in range(W):
            # square grid for training set
            if x % 2 == 0 and y % 2 == 0:
                train_batch.append((inp_batch[x][y], inp_target[x][y]))
            else:
                test_batch.append((inp_batch[x][y], inp_target[x][y]))

    # randomize test and training set
    random.shuffle(train_batch)
    random.shuffle(test_batch)

    train_inputs = []
    train_targets = []
    # get the training inputs and targets
    for i in range(len(train_batch)):
        train_inputs.append(train_batch[i][0])
        train_targets.append(train_batch[i][1])

    # stack to make training set
    train_inputs = torch.stack(train_inputs, dim=0)
    train_targets = torch.stack(train_targets, dim=0)

    train_batches = []
    train_batches_targets = []
    # get batches for the training set
    for i in range(0, len(train_batch), batch_size):
        train_batches.append(train_inputs[i:i+batch_size])
        train_batches_targets.append(train_targets[i:i+batch_size])

    test_inputs = []
    test_targets = []
    # get entire testing set, no need for batches
    for i in range(len(test_batch)):
        test_inputs.append(test_batch[i][0])
        test_targets.append(test_batch[i][1])
    
    # stack the batches
    train_batches = torch.stack(train_batches, dim=0).to(args.device)
    train_batches_targets = torch.stack(train_batches_targets, dim=0).to(args.device)
    test_inputs = torch.stack(test_inputs, dim=0).to(args.device)
    test_targets = torch.stack(test_targets, dim=0).to(args.device)

    return train_batches, train_batches_targets, test_inputs, test_targets

if __name__ == '__main__':

    # Begin training loop
    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=256, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=.0009, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to train network on')
    parser.add_argument('--batch_size', type=int, default=8192, help='make a training and testing set')
    parser.add_argument('--fourier_l', type=int, default=7, help='L value for Fourier')
    parser.add_argument('--gabor_l', type=int, default=7, help='L value for Gabor')
    parser.add_argument('--RFF', type=bool, default=False, help='sample from a gaussian')
    parser.add_argument('--encoding', type=str, default='raw_xy', help='encoding to test on')
    parser.add_argument('--model', type=str, default='gabor', help='type of model to train on (relu, sin, or siren)')
    parser.add_argument('--im_256', type=bool, default=False, help='256x256 images')
    parser.add_argument('--im_512', type=bool, default=False, help='512x512 images')
    parser.add_argument('--im_768', type=bool, default=False, help='768x768 images')
    parser.add_argument('--print_training_loss', type=bool, default=False, help='print training loss')
    parser.add_argument('--print_test_loss', type=bool, default=False, help='print testing loss')
    parser.add_argument('--print_psnr', type=bool, default=True, help='print test psnr')

    args = parser.parse_args()

    # lists to keep track of losses for both encodings
    both_encoding_test_loss = []

    # list of different encoding depths to train
    encoding = args.encoding
    fourier_L = args.fourier_l
    gabor_L = args.gabor_l
    # Determine whether random Gaussian or mip-Gabor will be used
    loss_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1)
    test_data = np.load('test_data_div2k.npy')

    gabor_psnr = []
    sin_cos_psnr = []
    raw_xy_psnr = []

    # Training Loop
    # loop through the amount of
    dist = []
    dist2 = []

    for im in test_data:

        # keep track of losses for each individual encoding
        test_psnr = []

        # Choose dataset depending on generalizing or reconstructing image
        if encoding == 'gauss': 
            inp_batch, inp_targets, test_batch, test_targets = get_data(image=im, encoding=encoding, L=args.gabor_l, batch_size=args.batch_size, RFF=args.RFF)
        elif encoding == 'sin_cos' or encoding == 'gauss_sin_cos': 
            inp_batch, inp_targets, test_batch, test_targets = get_data(image=im, encoding=encoding, L=args.fourier_l, batch_size=args.batch_size)
        elif encoding == 'raw_xy':
            inp_batch, inp_targets, test_batch, test_targets = get_data(image=im, encoding=encoding, L=0, batch_size=args.batch_size)
        elif encoding == 'gabor_2d':
            inp_batch, inp_targets, test_batch, test_targets = get_data(image=im, encoding=encoding, L=1, batch_size=args.batch_size, RFF=args.RFF)

        if args.model == 'relu':
            model = Net(inp_batch.shape[2], args.neurons).to(args.device)
        elif args.model == 'siren':
            model = SIREN(inp_batch.shape[2], args.neurons).to(args.device)
        elif args.model == 'gabor':
            model = GaborNet(inp_batch.shape[2], args.neurons).to(args.device)
        elif args.model == 'gsiren':
            model = GaborSIREN(inp_batch.shape[2], args.neurons).to(args.device)

        # Training criteria
        criterion = nn.MSELoss()
        lr = args.lr
        optimizer = optim.Adam(model.parameters(), lr=lr)
        epochs = args.epochs

        # individual training session lists
        best_psnr = -20
        for epoch in range(epochs):
            running_loss = 0
            # For a split training and test set
            for i, batch in enumerate(inp_batch):
                optimizer.zero_grad()
                output = model(batch)
                loss = criterion(output, inp_targets[i])
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                #dist.append(first_out.detach().cpu())
                #dist2.append(second_out.detach().cpu())
            
            #sns.kdeplot(data=torch.cat(dist).flatten(), fill=True)
            #sns.kdeplot(data=torch.cat(dist2).flatten(), fill=True)
            #plt.show()

            if args.print_training_loss:
                loss = running_loss / (inp_batch.shape[0])
                print("Epoch {} loss: {}".format(epoch, loss))

            # Get the testing results for this epoch
            with torch.no_grad():

                test_output = model(test_batch)
                test_loss = criterion(test_output, test_targets)
                test_output, test_targets = test_output.to('cpu'), test_targets.to('cpu')

                if args.print_test_loss:
                    print("Epoch {} test loss: {}".format(epoch, loss))

                psnr = loss_psnr(test_output, test_targets).item()

                if args.print_psnr:
                    print("Epoch {} psnr: {}".format(epoch, psnr))

                test_targets = test_targets.to('cuda:0')
                # keep track of best loss and psnr
                if psnr > best_psnr:
                    best_psnr = psnr

            # append the best psnr and loss after training to the lists
            test_psnr.append(best_psnr)

        print('{} max psnr {}: {}'.format(encoding, im, np.max(np.array(test_psnr))))

        if encoding == 'sin_cos' or encoding == 'gauss_sin_cos':
            sin_cos_psnr.append(np.max(np.array(test_psnr)))
        elif encoding == 'raw_xy':
            raw_xy_psnr.append(np.max(np.array(test_psnr)))

    print('total Gabor psnr: {}'.format(np.mean(np.array(raw_xy_psnr))))
    print('total Sin_Cos psnr: {}'.format(np.mean(np.array(sin_cos_psnr))))
    print('total raw_xy psnr: {}'.format(np.mean(np.array(raw_xy_psnr))))

