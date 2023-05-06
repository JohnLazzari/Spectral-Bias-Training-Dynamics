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

sns.set_style('darkgrid')
colors = sns.color_palette()

class Net(nn.Module):
    def __init__(self, input_dim, hidden):
        super(Net, self).__init__()
        self.hidden = hidden
        self.l1 = nn.Linear(input_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, hidden)
        self.l4 = nn.Linear(hidden, hidden)
        self.l5 = nn.Linear(hidden, 3)
        self.relu = nn.ReLU()

    def forward(self, x, pre=False):

        pre_out_1 = self.l1(x)
        out_1 = self.relu(pre_out_1)

        pre_out_2 = self.l2(out_1)
        out_2 = self.relu(pre_out_2)

        pre_out_3 = self.l3(out_2)
        out_3 = self.relu(pre_out_3)

        pre_out_4 = self.l4(out_3)
        out_4 = self.relu(pre_out_4)

        out_5 = self.l5(out_4)

        if pre:
            return pre_out_1, pre_out_2, pre_out_3, pre_out_4

        return out_5

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
    parser.add_argument('--epochs', type=int, default=2500, help='epochs')
    parser.add_argument('--batch_size', type=int, default=512*512, help='epochs')
    args = parser.parse_args()

    criterion = nn.MSELoss()
    epochs = args.epochs

    test_data = np.load('test_data_div2k.npy')
    test_data = test_data[:5]

    L_vals = [4, 8, 12, 16, 20]

    averaged_distances_pe = {}
    averaged_distances_pe_std = {}

    for l in L_vals:
        averaged_distances_pe[f'{l}_val'] = []
        averaged_distances_pe_std[f'{l}_val'] = []

    for im in test_data:

        for l in L_vals:

            model_pe = Net(l*4, 512).to('cuda:0')
            optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)

            # Get the encoding sin cos
            inp_batch, inp_target = get_data(image=im, encoding='sin_cos', shuffle=True, L=l, batch_size=args.batch_size)
            train_inp_batch, train_inp_target = get_data(image=im, encoding='sin_cos', shuffle=True, L=l, batch_size=8192)

            cur_dist = []

            for epoch in range(epochs):
                running_loss = 0
                for i, pixel in enumerate(train_inp_batch):
                    optim_pe.zero_grad()
                    output = model_pe(pixel)
                    loss = criterion(output, train_inp_target[i])
                    running_loss = loss.item()
                    loss.backward()
                    optim_pe.step()
                print('Epoch {} Loss: {}'.format(epoch, running_loss))

                # for each input, get its distance to the nearest boundary
                if epoch % 10 == 0:

                    with torch.no_grad():

                        for i, pixels in enumerate(inp_batch):
                            l1, l2, l3, l4 = model_pe(pixels, pre=True)

                            l1 = torch.abs(l1)
                            l2 = torch.abs(l2)
                            l3 = torch.abs(l3)
                            l4 = torch.abs(l4)

                            l1_norms = torch.linalg.norm(model_pe.l1.weight, dim=1)
                            l2_norms = torch.linalg.norm(model_pe.l2.weight, dim=1)
                            l3_norms = torch.linalg.norm(model_pe.l3.weight, dim=1)
                            l4_norms = torch.linalg.norm(model_pe.l4.weight, dim=1)

                            l1_dist = torch.div(l1, l1_norms)
                            l2_dist = torch.div(l2, l2_norms)
                            l3_dist = torch.div(l3, l3_norms)
                            l4_dist = torch.div(l4, l4_norms)

                            dist = torch.min(torch.cat((l1_dist, l2_dist, l3_dist, l4_dist), dim=-1), dim=-1)
                            cur_dist.append(torch.mean(dist[0]).item())

            averaged_distances_pe[f'{l}_val'].append(cur_dist)
    
    for l in L_vals:

        averaged_distances_pe[f'{l}_val'] = np.array(averaged_distances_pe[f'{l}_val'])
        averaged_distances_pe_std[f'{l}_val'] = np.std(averaged_distances_pe[f'{l}_val'], axis=0)
        averaged_distances_pe[f'{l}_val'] = np.mean(averaged_distances_pe[f'{l}_val'], axis=0)
    
    x = np.linspace(0, 2500, 250)

    fig1, ax1 = plt.subplots()
    ax1.plot(x, averaged_distances_pe[f'4_val'], label='Encoding L=4', linewidth=2)
    ax1.fill_between(x, np.array(averaged_distances_pe[f'4_val'])+np.array(averaged_distances_pe_std[f'4_val']), np.array(averaged_distances_pe[f'4_val'])-np.array(averaged_distances_pe_std[f'4_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_distances_pe[f'8_val'], label='Encoding L=8', linewidth=2)
    ax1.fill_between(x, np.array(averaged_distances_pe[f'8_val'])+np.array(averaged_distances_pe_std[f'8_val']), np.array(averaged_distances_pe[f'8_val'])-np.array(averaged_distances_pe_std[f'8_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_distances_pe[f'12_val'], label='Encoding L=12', linewidth=2)
    ax1.fill_between(x, np.array(averaged_distances_pe[f'12_val'])+np.array(averaged_distances_pe_std[f'12_val']), np.array(averaged_distances_pe[f'12_val'])-np.array(averaged_distances_pe_std[f'12_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_distances_pe[f'16_val'], label='Encoding L=16', linewidth=2)
    ax1.fill_between(x, np.array(averaged_distances_pe[f'16_val'])+np.array(averaged_distances_pe_std[f'16_val']), np.array(averaged_distances_pe[f'16_val'])-np.array(averaged_distances_pe_std[f'16_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)

    ax1.plot(x, averaged_distances_pe[f'20_val'], label='Encoding L=20', linewidth=2)
    ax1.fill_between(x, np.array(averaged_distances_pe[f'20_val'])+np.array(averaged_distances_pe_std[f'20_val']), np.array(averaged_distances_pe[f'20_val'])-np.array(averaged_distances_pe_std[f'20_val']), alpha=0.2, linewidth=2, linestyle='dashdot', antialiased=True)
        
    ax1.legend()
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Distance to nearest Boundary")
    fig1.savefig('region_distances/region_dist_test')
    
if __name__ == '__main__':
    main()