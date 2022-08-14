import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from nerf2D import Positional_Encoding
from torch_network import Net, SIN, SIREN
import random
import torchmetrics
import seaborn as sns
from mfn import FourierNet, GaborNet
import scipy
from torch import linalg as LA
import itertools

def get_activation_regions(model, input):
    patterns = []
    difference = []
    for i, pixel in enumerate(input):
        act_pattern = model(pixel, act=True)
        patterns.append(list(act_pattern))
        if i > 0:
            count = torch.sum(torch.abs(torch.tensor(patterns[i]) - torch.tensor(patterns[i-1])))
            difference.append(count.item())
    unique_patterns = []
    for pattern in patterns:
        if pattern not in unique_patterns:
            unique_patterns.append(pattern)
    return len(unique_patterns), np.mean(np.array(difference))

def get_data(image, encoding, L=10, batch_size=2048, RFF=False):

    # Get the training image
    trainimg= image
    trainimg = trainimg / 255.0  
    H, W, C = trainimg.shape

    # Get the encoding
    PE = Positional_Encoding(trainimg, encoding, training=True)
    inp_batch, inp_target, ind_vals = PE.get_dataset(L, RFF=False)

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

model_raw = Net(2, 784).to('cuda:0')
optim_raw = torch.optim.Adam(model_raw.parameters(), lr=.001)
model_pe = Net(32, 128).to('cuda:0')
optim_pe = torch.optim.Adam(model_pe.parameters(), lr=.001)
model_gabor = Net(176, 128).to('cuda:0')
optim_gabor = torch.optim.Adam(model_gabor.parameters(), lr=.001)
criterion = nn.MSELoss()
epochs = 500

# compute gradients individually for each, not sure best way to do this yet
im2arr = np.random.randint(0, 255, (28, 28, 3))

#################################### Raw xy ###############################################

# Get the encoding raw_xy
train_inp_batch, train_inp_target = get_data(image=im2arr, encoding='raw_xy', L=0, batch_size=28)
inp_batch, inp_target = get_data(image=im2arr, encoding='raw_xy', L=0, batch_size=1)
full_batch, full_target = get_data(image=im2arr, encoding='raw_xy', L=0, batch_size=96*96)

raw_grad_similarities = []
raw_param_norms = []
raw_num_patterns = []

layer_1_norms = []
layer_3_norms = []
layer_2_norms = []

total_grad_norm_raw = []

losses = []
for epoch in range(epochs):
    dead_neuron_count = 0
    running_loss = 0
    for i, pixel in enumerate(train_inp_batch):
        optim_raw.zero_grad()
        output = model_raw(pixel)
        loss = criterion(output, train_inp_target[i])
        running_loss += loss.item()
        loss.backward()
        optim_raw.step()
    epoch_loss = running_loss / 28
    print(epoch_loss)
    losses.append(epoch_loss)

    # This is to find parameter norms for lipschitz constant
    with torch.no_grad():
        # Go through each layer and add up the norms
        U, S, V = torch.linalg.svd(model_raw.l1.weight)
        norm_1 = max(S)
        layer_1_norms.append(norm_1.item())
        U, S, V = torch.linalg.svd(model_raw.l2.weight)
        norm_2 = max(S)
        layer_2_norms.append(norm_2.item())
        U, S, V = torch.linalg.svd(model_raw.l3.weight)
        norm_3 = max(S)
        layer_3_norms.append(norm_3.item())
        total_norm = norm_1 * norm_2 * norm_3
        raw_param_norms.append(total_norm.item())

    '''
    # this is to find gradient norms to see if this is cause for rise in raw_xy param norm
    optim_raw.zero_grad()
    output = model_raw(full_batch)
    loss = criterion(output, full_target)
    loss.backward()
    # check the amount of dead neurons
    all_grads = torch.cat([torch.flatten(model_raw.l1.weight.grad), torch.flatten(model_raw.l2.weight.grad), torch.flatten(model_raw.l3.weight.grad)])
    # Check amount of weight gradients that are zero
    for grad in all_grads:
        if grad == 0.0:
            dead_neuron_count += 1
    print('num dead weights: {}'.format(dead_neuron_count))
    '''

    if epoch > 490:
        raw_gradients = torch.empty([784, 620147]).to('cuda:0')
        for i, pixel in enumerate(inp_batch):
            optim_raw.zero_grad()
            output = model_raw(pixel)
            loss = criterion(output, inp_target[i])
            running_loss += loss.item()
            loss.backward()
            grads = torch.cat([torch.flatten(model_raw.l1.weight.grad), torch.flatten(model_raw.l2.weight.grad), torch.flatten(model_raw.l3.weight.grad), torch.flatten(model_raw.l1.bias.grad), torch.flatten(model_raw.l2.bias.grad), torch.flatten(model_raw.l3.bias.grad)])
            raw_gradients[i] = grads
        running_loss /= inp_batch.shape[0]
        losses.append(running_loss)
        # get inner products of gradients for raw xy
        for i in range(raw_gradients.shape[0]):
            #print(torchmetrics.functional.pairwise_cosine_similarity(raw_gradients[i].unsqueeze(dim=0), raw_gradients[j].unsqueeze(dim=0)).cpu().item())
            raw_grad_similarities.append(torch.flatten(torchmetrics.functional.pairwise_cosine_similarity(raw_gradients[i].unsqueeze(dim=0), raw_gradients[i+1:])).cpu())

'''
plt.plot(layer_1_norms, label='layer1')
plt.plot(layer_2_norms, label='layer2')
plt.plot(layer_3_norms, label='layer3')
plt.legend()
plt.show()
'''

'''
# Get num regions for raw_xy
with torch.no_grad():
    raw_regions, differences = get_activation_regions(model_raw, inp_batch)
    print(raw_regions)
    print("how the regions differ: {}".format(differences))
    raw_num_patterns.append(raw_regions)
'''

#plt.plot(raw_num_patterns)
#plt.show()

######################## Positional Encoding ######################################

# Get the encoding sin cos
train_inp_batch, train_inp_target = get_data(image=im2arr, encoding='sin_cos', L=8, batch_size=28)
inp_batch, inp_target = get_data(image=im2arr, encoding='sin_cos', L=8, batch_size=1)
full_batch, full_target = get_data(image=im2arr, encoding='sin_cos', L=8, batch_size=96*96)

sin_cos_grad_similarity = []
sin_cos_norms = []

layer_1_norms_pe = []
layer_3_norms_pe = []
layer_2_norms_pe = []

total_grad_norm_pe = []

pe_num_patterns = []
pe_losses = []
for epoch in range(epochs):
    dead_neuron_count = 0
    running_loss = 0
    for i, pixel in enumerate(train_inp_batch):
        optim_pe.zero_grad()
        output = model_pe(pixel)
        loss = criterion(output, train_inp_target[i])
        running_loss += loss.item()
        loss.backward()
        optim_pe.step()
    epoch_loss = running_loss / 28
    print(epoch_loss)

    # Get param norms for lipschitz sin_cos
    with torch.no_grad():
        U, S, V = torch.linalg.svd(model_pe.l1.weight)
        norm_1 = max(S)
        layer_1_norms_pe.append(norm_1.item())
        U, S, V = torch.linalg.svd(model_pe.l2.weight)
        norm_2 = max(S)
        layer_2_norms_pe.append(norm_2.item())
        U, S, V = torch.linalg.svd(model_pe.l3.weight)
        norm_3 = max(S)
        layer_3_norms_pe.append(norm_3.item())
        total_norm = norm_1 * norm_2 * norm_3
        sin_cos_norms.append(total_norm.item())
    
    '''
    # Get gradient norms for sin cos
    optim_pe.zero_grad()
    output = model_pe(full_batch)
    loss = criterion(output, full_target)
    loss.backward()
    all_grads = torch.cat([torch.flatten(model_pe.l1.weight.grad), torch.flatten(model_pe.l2.weight.grad), torch.flatten(model_pe.l3.weight.grad)])
    # Check amount of weight gradients which are zero
    for grad in all_grads:
        if grad == 0:
            dead_neuron_count += 1
    print('num dead weights: {}'.format(dead_neuron_count))
    '''

    # get confusion sin_cos
    if epoch > 490:
        pe_gradients = torch.empty([784, 21123]).to('cuda:0')
        for i, pixel in enumerate(inp_batch):
            optim_pe.zero_grad()
            output = model_pe(pixel)
            loss = criterion(output, inp_target[i])
            running_loss += loss.item()
            loss.backward()
            grads = torch.cat([torch.flatten(model_pe.l1.weight.grad), torch.flatten(model_pe.l2.weight.grad), torch.flatten(model_pe.l3.weight.grad), torch.flatten(model_pe.l1.bias.grad), torch.flatten(model_pe.l2.bias.grad), torch.flatten(model_pe.l3.bias.grad)])
            pe_gradients[i] = grads
        running_loss /= inp_batch.shape[0]
        pe_losses.append(running_loss)
        # get inner products of gradients for sin_cos
        for i in range(pe_gradients.shape[0]):
            sin_cos_grad_similarity.append(torch.flatten(torchmetrics.functional.pairwise_cosine_similarity(pe_gradients[i].unsqueeze(dim=0), pe_gradients[i+1:]).cpu()))

'''
# plotting the param norms between layers for sin cos
plt.plot(layer_1_norms_pe, label='layer1')
plt.plot(layer_2_norms_pe, label='layer2')
plt.plot(layer_3_norms_pe, label='layer3')
plt.legend()
plt.show()
'''

'''
# Get the number of activation regions for sin_cos
with torch.no_grad():
    pe_regions, differences = get_activation_regions(model_pe, inp_batch)
    print(pe_regions)
    print('how pe regions differ: {}'.format(differences))
    pe_num_patterns.append(pe_regions)
'''

# plot the total param norms for raw xy and sin cos
plt.plot(sin_cos_norms, label='Fourier Features')
plt.plot(raw_param_norms, label='raw_xy')
plt.legend()
plt.show()

sns.kdeplot(data=torch.cat(sin_cos_grad_similarity).cpu(), fill=True, label='Fourier Features')
sns.kdeplot(data=torch.cat(raw_grad_similarities).cpu(), fill=True, label='raw_xy')
plt.legend()
plt.show()