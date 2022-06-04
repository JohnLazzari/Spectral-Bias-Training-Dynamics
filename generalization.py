import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from nerf2D import Positional_Encoding
from PIL import Image
import argparse
import random
import torchmetrics
from math import log10, sqrt
from tqdm import tqdm as tq

class Net(nn.Module):
    def __init__(self, input_dim, hidden):
        super(Net, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.l3(out)

        return out

def get_data(image, encoding, L=10, batch_size=2048, RFF=False):
    # This is to actually pass the image through the network
    im = Image.open(f'dataset/{image}.jpg')
    im2arr = np.array(im) 

    # Get the training image
    trainimg= im2arr 
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
    parser.add_argument('--neurons', type=int, default=128, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=350, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=.005, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to train network on')
    parser.add_argument('--image', type=str, default='fractal', help='Image to learn')
    parser.add_argument('--batch_size', type=int, default=2048, help='make a training and testing set')
    parser.add_argument('--fourier_l', type=int, default=5, help='L value for Fourier')
    parser.add_argument('--gabor_l', type=int, default=5, help='L value for Gabor')
    parser.add_argument('--RFF', type=bool, default=False, help='sample from a gaussian')
    parser.add_argument('--print_training_loss', type=bool, default=False, help='print training loss')
    parser.add_argument('--print_test_loss', type=bool, default=False, help='print testing loss')
    parser.add_argument('--print_psnr', type=bool, default=False, help='print test psnr')

    args = parser.parse_args()

    # lists to keep track of losses for both encodings
    both_encoding_test_loss = []
    both_encoding_psnr = []

    # list of different encoding depths to train
    encodings = ['gauss', 'sin_cos']
    fourier_L = args.fourier_l
    gabor_L = args.gabor_l
    # Determine whether random Gaussian or mip-Gabor will be used
    loss_psnr = torchmetrics.PeakSignalNoiseRatio(data_range=1)
    average = 5

    # Training Loop
    # loop through the amount of
    for encoding in encodings:
        # keep track of losses for each individual encoding
        test_psnr = []

        # Choose dataset depending on generalizing or reconstructing image
        if encoding == 'gauss': 
            inp_batch, inp_targets, test_batch, test_targets = get_data(image=args.image, encoding=encoding, L=gabor_L, batch_size=args.batch_size, RFF=args.RFF)
        elif encoding == 'sin_cos' or encoding == 'gauss_sin_cos': 
            inp_batch, inp_targets, test_batch, test_targets = get_data(image=args.image, encoding=encoding, L=fourier_L, batch_size=args.batch_size)

        # currently averaging over 5 iterations
        for k in range(average):

            model = Net(inp_batch.shape[2], args.neurons).to(args.device)
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

        print('{} mean psnr: {}'.format(encoding, np.mean(np.array(test_psnr))))
        both_encoding_psnr.append(test_psnr)
        print(both_encoding_psnr)
