import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from nerf2D import Positional_Encoding

trainimg = np.empty([28, 28, 3])
# Get the encoding sin cos
PE = Positional_Encoding(trainimg, 'sin_cos', training=True)
inp_batch, inp_target, ind_vals = PE.get_dataset(L=1)

sin_cos_kernel = np.empty([784, 784])

for i in range(np.shape(inp_batch)[0]):
    for j in range(np.shape(inp_batch)[0]):
        sin_cos_kernel[i][j] = np.dot(inp_batch[i], inp_batch[j])

plt.matshow(sin_cos_kernel)
plt.colorbar()
plt.show()

# Get the encoding raw_xy
PE = Positional_Encoding(trainimg, 'raw_xy', training=True)
inp_batch, inp_target, ind_vals = PE.get_dataset(L=5)

raw_xy_kernel = np.empty([784, 784])

for i in range(np.shape(inp_batch)[0]):
    for j in range(np.shape(inp_batch)[0]):
        raw_xy_kernel[i][j] = np.dot(inp_batch[i], inp_batch[j])

plt.matshow(raw_xy_kernel)
plt.colorbar()
plt.show()

# Get the encoding raw_xy
PE = Positional_Encoding(trainimg, 'gauss', training=True)
inp_batch, inp_target, ind_vals = PE.get_dataset(L=5)

gabor_kernel = np.empty([784, 784])

for i in range(np.shape(inp_batch)[0]):
    for j in range(np.shape(inp_batch)[0]):
        gabor_kernel[i][j] = np.dot(inp_batch[i], inp_batch[j])

plt.matshow(gabor_kernel)
plt.colorbar()
plt.show()