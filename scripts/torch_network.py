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

class Net(nn.Module):
    def __init__(self, input_dim, hidden):
        super(Net, self).__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 3)
        self.relu = nn.ReLU()

    def forward(self, x, act=False, first_layer=False):
        out = self.relu(self.l1(x))

        if act:
            act_1 = out
            if first_layer:
                return act_1

        out = self.relu(self.l2(out))

        if act:
            act_2 = out

        out = self.l3(out)

        if act:
            pattern = torch.cat((act_1, act_2), dim=0).squeeze()
            return pattern

        return out

def get_data(image, encoding, L=10, batch_size=2048, RFF=False):
    # This is to actually pass the image through the network
    im = Image.open(f'{image}.jpg')
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

    # create batches to track batch loss and show it is more stable due to gabor encoding
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
    parser.add_argument('--neurons', type=int, default=256, help='Number of neurons per layer')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--encoding', type=str, default='raw_xy',
            help=f'Positional encoding')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to train network on')
    parser.add_argument('--image', type=str, default='fractal', help='Image to learn')
    parser.add_argument('--batch_size', type=int, default=2048, help='make a training and testing set')
    parser.add_argument('--L', type=int, default=10, help='L value for encoding')
    parser.add_argument('--RFF', type=bool, default=False, help='L value for encoding')

    args = parser.parse_args()

    # list of different encoding depths to train
    encoding = args.encoding
    L = args.L
    RFF = args.RFF

    # Choose dataset depending on generalizing or reconstructing image
    inp_batch, inp_targets = get_data(image=args.image, encoding=encoding, L=L, batch_size=args.batch_size, RFF=RFF)
    model = Net(inp_batch.shape[2], args.neurons).to(args.device)

    # Training criteria
    criterion = nn.MSELoss()
    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = args.epochs

    # individual training session lists
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

        loss = running_loss / (inp_batch.shape[0])
        print("Epoch {} loss: {}".format(epoch, loss))

    # Save the model based on the encoding
    torch.save(model.state_dict(), f'./saved_model/torch_{encoding}_{args.image}.pth')    
