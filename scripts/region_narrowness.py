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
import argparse

class Net(nn.Module):
    def __init__(self, input_dim, hidden):
        super(Net, self).__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 3)
        self.relu = nn.ReLU()

    def forward(self, x, pre=False):
        l1_pre = self.l1(x)
        l1_post = self.relu(l1_pre)
        l2_pre = self.l2(l1_post)
        l2_post = self.relu(l2_pre)
        out = self.l3(l2_post)
        if pre:
            return l1_pre, l2_pre
        else:
            return out

def get_data(image, encoding, shuffle, L=10,  batch_size=2048):

    # Get the training image
    trainimg= image
    trainimg = trainimg / 255.0  
    H, W, C = trainimg.shape

    # Get the encoding
    PE = Positional_Encoding(trainimg, encoding, training=True)
    inp_batch, inp_target, ind_vals = PE.get_dataset(L)

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

def main():

    parser = argparse.ArgumentParser(description='Blah.')
    parser.add_argument('--L', type=int, default=5, help='L value')
    parser.add_argument('--epochs', type=int, default=5000, help='epochs')
    args = parser.parse_args()

    model_pe = Net(args.L*4, 128).to('cuda:0')
    optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)

    criterion = nn.MSELoss()
    epochs = args.epochs
    distances = []
    std_distances = []

    im2arr = np.random.randint(0, 255, (64, 64, 3))
    #im = Image.open(f'image_demonstration/glasses.jpg')
    #im2arr = np.array(im) 

    # Get the encoding sin cos
    train_inp_batch, train_inp_target = get_data(image=im2arr, encoding='sin_cos', shuffle=True, L=args.L, batch_size=4096)
    inp_batch, inp_target = get_data(image=im2arr, encoding='sin_cos', shuffle=False, L=args.L, batch_size=1)

    for epoch in range(epochs):
        running_loss = 0
        for i, pixel in enumerate(train_inp_batch):
            optim_pe.zero_grad()
            output = model_pe(pixel)
            loss = criterion(output, train_inp_target[i])
            running_loss += loss.item()
            loss.backward()
            optim_pe.step()
        running_loss /= 16
        print('Epoch {} Loss: {}'.format(epoch, running_loss))

        # for each input, get its distance to the nearest boundary
        if epoch % 50 == 0:

            with torch.no_grad():

                cur_dist = []
                for i, pixel in enumerate(inp_batch):
                    l1, l2 = model_pe(pixel, pre=True)

                    l1 = torch.abs(l1)
                    l2 = torch.abs(l2)

                    l1_norms = torch.linalg.norm(model_pe.l1.weight, dim=1)
                    l2_norms = torch.linalg.norm(model_pe.l2.weight, dim=1)

                    l1_dist = l1 / l1_norms
                    l2_dist = l2 / l2_norms

                    dist = torch.min(torch.cat((l1_dist, l2_dist)))
                    cur_dist.append(dist.item())

                mean = sum(cur_dist) / len(cur_dist)
                distances.append(mean)
                std_distances.append(sum([((x - mean) ** 2) for x in cur_dist]) / len(cur_dist))
        
if __name__ == '__main__':
    main()