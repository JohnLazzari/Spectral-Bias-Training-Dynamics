import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import fft, ifft
import argparse
import torch
import os
from scipy.stats import multivariate_normal

def gaussian(x, mu=0, stddev=1):
    return (1. / (stddev * np.sqrt(2. * np.pi))) * np.exp((-(x - mu)**2) / (2. * (stddev ** 2.)))

class Positional_Encoding(object):
    available_encodings = ['sin_cos', 'repeat_xy']
    def __init__(self, image, encoding, training=True, signal=None):
        super().__init__()

        self.image = image
        self.encoding = encoding
        self.training = training
        self.signal = signal


    def get_dataset(self, L=10, RFF=False):
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

                bvals = np.random.normal(size=(256, 2)) * 10
                bvals = torch.Tensor(bvals).to('cuda:0')
                inputs = torch.cat([torch.sin((2.*np.pi*x) @ bvals.T), 
                                    torch.cos((2.*np.pi*x) @ bvals.T)], dim=-1)
                for k in range(0):
                    bvals = np.random.normal(size=(256, 2)) * 10
                    bvals = torch.Tensor(bvals).to('cuda:0')
                    encoding = torch.cat([torch.sin((2.*np.pi*x) @ bvals.T), 
                                          torch.cos((2.*np.pi*x) @ bvals.T)], dim=-1)
                    inputs += encoding
                inputs = np.array(inputs.cpu())
                
                inputs = inputs.reshape(width * height, inputs.shape[-1])
                outputs = self.image.reshape(width * height, channels)
                indices = np.array([[x, y] for x in range(width) for y in range(height)])
                return inputs, outputs, indices
            
            elif self.encoding == 'gabor_2d':

                # make this gabor perform better then do a sum of sinusoids to potentially improve performance
                # this worked for Gaussian Fourier Features

                x = np.linspace(-1, 1, (height)+1)[:-1]
                x = np.stack(np.meshgrid(x, x), axis=-1)
                x = torch.Tensor(x).to('cuda:0')
                flattened_input = x.reshape(width * height, x.shape[-1]) 

                #gaussians = torch.empty(256, 2).to('cuda:0')
                #gaussians = gaussians.uniform_(-1, 1)
                #variance = torch.empty(256).to('cuda:0')
                #variance =  torch.distributions.gamma.Gamma(2, 1).sample((256,)).to('cuda:0')
                #eye = torch.eye(2).to('cuda:0')
                #variance *= eye

                encodings = []

                for k in range(1):

                    dims = 128
                    bvals = np.random.normal(size=(dims, 2)) * 8
                    bvals = torch.Tensor(bvals).to('cuda:0')
                    sin_component = torch.sin((2.*np.pi*x) @ bvals.T)
                    cos_component = torch.cos((2.*np.pi*x) @ bvals.T)

                    #sin_component = sin_component.reshape(width * height, sin_component.shape[-1])
                    #cos_component = cos_component.reshape(width * height, cos_component.shape[-1])
                    gaussian_inputs = []

                    gaussians = np.random.uniform(-1, 1, size=(dims, 2))
                    gaussians = torch.Tensor(gaussians).to('cuda:0')

                    #variance =  torch.distributions.gamma.Gamma(6, 2).sample((256,)).to('cuda:0')
                    variance = torch.empty(dims)
                    variance = variance.uniform_(.15, .45).to('cuda:0')

                    D = (
                            (x ** 2).sum(-1)[..., None]
                            + (gaussians ** 2).sum(-1)[None, :]
                            - 2 * x @ gaussians.T
                        )
                    gaussian_encoding = torch.exp(-.5 * D * variance[None, :])

                    inputs = torch.cat([sin_component * gaussian_encoding,
                                        cos_component * gaussian_encoding], dim=-1)
                    
                    inputs = np.array(inputs.cpu())
                    inputs = np.reshape(inputs, [width*height, dims*2])
                    encodings.append(inputs)

                encodings = np.stack(encodings, axis=2)
                encodings = np.sum(encodings, axis=2)

                outputs = self.image.reshape(width * height, channels)
                indices = np.array([[x, y] for x in range(width) for y in range(height)])
                return encodings, outputs, indices
                
            elif self.encoding == "gauss":

                stddev = .9
                # Scale output to [0, 1).
                if np.max(self.image) > 1:
                    self.image = self.image / 255.
                
                 # Generate gaussian encodings.
                g_enc = []
                l = np.linspace(-1, 1, width)
                regions = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                for s in range(1, L + 1):
                    for g in range(regions[s-1]):
                        mu = 2 * g / s - 1
                        g_l_sin = gaussian(l, mu, stddev / (regions[s-1]-1)) * np.sin((2 * np.random.normal(0, 10)) * np.pi * l)
                        g_l_cos = gaussian(l, mu, stddev / (regions[s-1]-1)) * np.cos((2 * np.random.normal(0, 10)) * np.pi * l)

                        g_enc.append(g_l_sin)
                        g_enc.append(g_l_cos)

                g_enc = np.array(g_enc)
                # Format as inputs.
                g_enc_len = len(g_enc)
                inputs = np.zeros((width, height, 2 * g_enc_len))

                for x in range(width):
                    for y in range(height):
                        d = 0
                        for s in range(0, 2*g_enc_len, 4):
                            inputs[x, y, s:s+2] = g_enc[d:d+2, x]
                            inputs[x, y, s+2:s+4] = g_enc[d:d+2, y]
                            d += 2

                inputs = inputs.reshape(width * height, inputs.shape[-1])
                outputs = self.image.reshape(width * height, channels)
                indices = np.array([[x, y] for x in range(width) for y in range(height)])
                break
        
        if self.encoding != "gauss":
            inputs, outputs, indices = [], [], []

            for y in range(height):
                for x in range(width):
                    # i.e. passing raw coordinates instead of positional encoding 
                    if self.encoding == 'raw_xy':

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
