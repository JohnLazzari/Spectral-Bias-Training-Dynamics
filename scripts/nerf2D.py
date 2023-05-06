import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import fft, ifft
import argparse
import torch
import os
from scipy.stats import multivariate_normal

class Positional_Encoding(object):
    def __init__(self, image, encoding, training=True, signal=None):
        super().__init__()

        self.image = image
        self.encoding = encoding
        self.training = training
        self.signal = signal


    def get_dataset(self, L=10, negative=False):
        height, width, _ = self.image.shape

        # Format positions to range [-1, 1)
        x_linspace = np.linspace(0, 1, width)
        y_linspace = np.linspace(0, 1, width)

        channels = 3

        x_encoding = []
        y_encoding = []
        x_encoding_hf = []
        y_encoding_hf = []

        for l in range(L):

            val = 2 ** l

            # Gamma encoding described in (4) of NeRF paper.
            if self.encoding == 'sin_cos':

                x = np.sin(val * np.pi * x_linspace)
                x_encoding.append(x)

                x = np.cos(val * np.pi * x_linspace)
                x_encoding_hf.append(x)

                y = np.sin(val * np.pi * y_linspace)
                y_encoding.append(y)

                y = np.cos(val * np.pi * y_linspace)
                y_encoding_hf.append(y)

            elif self.encoding == 'gauss_sin_cos':

                x = np.linspace(0, 1, (height)+1)[:-1]
                x = np.stack(np.meshgrid(x, x), axis=-1)
                x = x.reshape(width*height, 2)
                x = torch.Tensor(x).to('cuda:0')

                bvals = np.random.normal(size=(256, 2)) * 14
                bvals = torch.Tensor(bvals).to('cuda:0')
                inputs = torch.cat([torch.sin((2.*np.pi*x) @ bvals.T), 
                                    torch.cos((2.*np.pi*x) @ bvals.T)], dim=-1)
                inputs = np.array(inputs.cpu())
                
                inputs = inputs.reshape(width * height, inputs.shape[-1])
                outputs = self.image.reshape(width * height, channels)
                indices = np.array([[x, y] for x in range(width) for y in range(height)])
                return inputs, outputs, indices
            
        inputs, outputs, indices = [], [], []

        for y in range(height):
            for x in range(width):
                # i.e. passing raw coordinates instead of positional encoding 
                if self.encoding == 'raw_xy':
                    if negative:
                        xdash = 2 * (x/width) - 1
                        ydash = 2 * (y/height) - 1
                        p_enc = [xdash, ydash]
                    else:
                        xdash = (x/width)
                        ydash = (y/height)
                        p_enc = [xdash, ydash]

                else:
                    # Concatenate positional encoding.
                    p_enc = []
                    for l in range(L):
                        p_enc.append(x_encoding[l][x])
                        p_enc.append(x_encoding_hf[l][x])

                        p_enc.append(y_encoding[l][y])
                        p_enc.append(y_encoding_hf[l][y])

                inputs.append(p_enc)
                outputs = self.image.reshape(width * height, channels)
                indices.append([float(x), float(y)])

        return np.asarray(inputs), np.asarray(outputs), np.asarray(indices)
