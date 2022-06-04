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

def get_data(image, encoding, L=10, batch_size=2048):
    # This is to actually pass the image through the network
    im = Image.open(f'dataset/{image}.jpg')
    im2arr = np.array(im) 

    # Get the training image
    trainimg= im2arr 
    trainimg = trainimg / 255.0  
    H, W, C = trainimg.shape

    # Get the encoding
    PE = Positional_Encoding(trainimg, encoding, training=True)
    inp_batch, inp_target, ind_vals = PE.get_dataset(L)

    inp_batch, inp_target = torch.Tensor(inp_batch), torch.Tensor(inp_target)
    inp_batch, inp_target = inp_batch.to(args.device), inp_target.to(args.device)

    full_batches = []
    batches = []
    batch_targets = []

    random_batches = []
    random_targets = []
    # make sure to choose one which can be evenly divided by 65536
    for i in range(H*W):
        full_batches.append((inp_batch[i], inp_target[i]))

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
    
if __name__ == '__main__':

    # Begin training loop
    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--neurons', type=int, default=128, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=.005, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to train network on')
    parser.add_argument('--image', type=str, default='fractal', help='Image to learn')
    parser.add_argument('--batch_size', type=int, default=256, help='make a training and testing set')

    args = parser.parse_args()

    # lists to keep track of losses for both encodings
    total_batch_losses = []

    # list of different encoding depths to train
    encodings = ['gauss', 'sin_cos']
    L = [8, 12, 16]
    # Training Loop
    # loop through the amount of
    for encoding in range(len(encodings)):
        # currently averaging over 5 iterations
        for k in range(len(L)):
            # Choose dataset depending on generalizing or reconstructing image
            inp_batch, inp_targets = get_data(image=args.image, encoding=encodings[encoding], L=L[k], batch_size=args.batch_size)
            model = Net(inp_batch.shape[2], args.neurons).to(args.device)
            # Training criteria
            criterion = nn.MSELoss()
            lr = args.lr
            optimizer = optim.Adam(model.parameters(), lr=lr)
            epochs = args.epochs

            # individual training session lists
            batch_losses = []
            for epoch in range(epochs):
                running_loss = 0
                # For a split training and test set
                for i, batch in enumerate(inp_batch):
                    optimizer.zero_grad()
                    output = model(batch)
                    loss = criterion(output, inp_targets[i])
                    batch_losses.append(loss.item())
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                loss = running_loss / (inp_batch.shape[0])
                print("Epoch {} loss: {}".format(epoch, loss))
            # append the best psnr and loss after training to the lists
            total_batch_losses.append(batch_losses)

    # Gabor
    plt.plot(total_batch_losses[0], label='L: 8')
    plt.plot(total_batch_losses[1], label='L: 12')
    plt.plot(total_batch_losses[2], label='L: 16')
    plt.legend(loc='upper right')
    plt.ylim(top=.04)
    plt.ylim(bottom=.001)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.show()

    # Fourier
    plt.plot(total_batch_losses[3], label='L: 8')
    plt.plot(total_batch_losses[4], label='L: 12')
    plt.plot(total_batch_losses[5], label='L: 16')
    plt.legend(loc='upper right')
    plt.ylim(top=.04)
    plt.ylim(bottom=.001)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Error')
    plt.show()