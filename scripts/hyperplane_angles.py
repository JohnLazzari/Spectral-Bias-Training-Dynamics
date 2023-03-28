import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from nerf2D import Positional_Encoding
from torch_network import Net
import random
import torchmetrics
import seaborn as sns

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

    epochs = 5000
    L_val = 5
    input_size = L_val * 4

    model_pe = Net(input_size, 128).to('cuda:0')
    optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)

    criterion = nn.MSELoss()

    im2arr = np.random.randint(0, 255, (64, 64, 3))

    # Get the encoding sin cos
    train_inp_batch, train_inp_target = get_data(image=im2arr, encoding='sin_cos', shuffle=True, L=L_val, batch_size=256)

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

    # this is only using two layer network from main paper
    # change to natural images if needed
    with torch.no_grad():

        l1_weights = model_pe.l1.weight
        mask = torch.zeros([128, 128-input_size]).to('cuda:0')
        l1_weights = torch.cat((l1_weights, mask), dim=1)

        l2_weights = model_pe.l2.weight

        all_weights = torch.cat((l1_weights, l2_weights), dim=0)
        angle_matrix_pe = torch.empty([256, 256]).to('cuda')

        # get inner products of gradients for sin_cos
        for i in range(all_weights.shape[0]):
            angle_matrix_pe[i] = torch.flatten(torchmetrics.functional.pairwise_cosine_similarity(all_weights[i].unsqueeze(dim=0), all_weights).cpu())

    plt.matshow(angle_matrix_pe.cpu().numpy(), cmap='seismic', vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()