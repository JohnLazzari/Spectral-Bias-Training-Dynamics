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

    def forward(self, x):

        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.relu(self.l3(out))
        out = self.relu(self.l4(out))
        out = self.l5(out)

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

    epochs = 5000
    L_val = 8
    input_size = L_val * 4

    model_pe = Net(input_size, 512).to('cuda:0')
    optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)

    criterion = nn.MSELoss()

    test_data = np.load('test_data_div2k.npy')
    test_data = test_data[0]

    # Get the encoding sin cos
    train_inp_batch, train_inp_target = get_data(image=test_data, encoding='sin_cos', shuffle=True, L=L_val, batch_size=8192)

    for epoch in range(epochs):
        running_loss = 0
        for i, pixel in enumerate(train_inp_batch):
            optim_pe.zero_grad()
            output = model_pe(pixel)
            loss = criterion(output, train_inp_target[i])
            running_loss += loss.item()
            loss.backward()
            optim_pe.step()
        running_loss /= 64
        print('Epoch {} Loss: {}'.format(epoch, running_loss))

    with torch.no_grad():

        l1_weights = model_pe.l1.weight
        mask = torch.zeros([512, 512-input_size]).to('cuda:0')
        l1_weights = torch.cat((l1_weights, mask), dim=1)

        l2_weights = model_pe.l2.weight
        l3_weights = model_pe.l3.weight
        l4_weights = model_pe.l4.weight

        all_weights = torch.cat((l1_weights, l2_weights, l3_weights, l4_weights), dim=0)
        angle_matrix_pe = torch.empty([2048, 2048]).to('cuda')

        # get inner products of gradients for sin_cos
        for i in range(all_weights.shape[0]):
            angle_matrix_pe[i] = torch.flatten(torchmetrics.functional.pairwise_cosine_similarity(all_weights[i].unsqueeze(dim=0), all_weights).cpu())

    plt.matshow(angle_matrix_pe.cpu().numpy(), cmap='seismic', vmin=-1, vmax=1)
    plt.colorbar()
    plt.tick_params(left = False, right = False , top = False, labelleft = False ,
                labelbottom = False, labeltop=False, bottom = False)
    plt.savefig('hyperplane_angles/epoch5000_l8.png')
    plt.show()

if __name__ == '__main__':
    main()