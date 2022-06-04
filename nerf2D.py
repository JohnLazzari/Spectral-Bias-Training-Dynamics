import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.fft import fft, ifft
import argparse
import os

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
        x_linspace = np.linspace(-1, 1, width)
        y_linspace = np.linspace(-1, 1, width)

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

                rand = np.random.normal(0, 10)
                rand_y = np.random.normal(0, 10)

                x = np.sin(2 * np.pi * rand * x_linspace)
                x_encoding.append(x)

                x = np.cos(2 * np.pi * rand * x_linspace)
                x_encoding_hf.append(x)

                y = np.sin(2 * np.pi * rand_y * y_linspace)
                y_encoding.append(y)

                y = np.cos(2 * np.pi * rand_y * y_linspace)
                y_encoding_hf.append(y)


            # Deep Learning Group proposed encoding.
            elif self.encoding == 'repeat_xy':

                x_encoding.append(x_linspace)

                x_encoding_hf.append(x_linspace)

                y_encoding.append(y_linspace)

                y_encoding_hf.append(y_linspace)

            elif self.encoding == "gauss":

                stddev = .9
                # Scale output to [0, 1).
                if np.max(self.image) > 1:
                    self.image = self.image / 255.
                
                # Generate gaussian encodings.
                g_enc = []
                l = np.linspace(-1, 1, width)
                stop = 25
                for s in range(1, L + 1):
                    if s <= stop:
                        for g in range(s+1):
                            mu = 2 * g / s - 1
                            rand = np.random.normal(0, 10)
                            if RFF:
                                g_l_sin = gaussian(l, mu, stddev / (s)) * np.sin((2 * rand) * np.pi * l)
                                g_l_cos = gaussian(l, mu, stddev / (s)) * np.cos((2 * rand) * np.pi * l)
                            else:
                                g_l_sin = gaussian(l, mu, stddev / (s)) * np.sin((2 ** ((s-1))) * np.pi * l)
                                g_l_cos = gaussian(l, mu, stddev / (s)) * np.cos((2 ** ((s-1))) * np.pi * l)

                            g_enc.append(g_l_sin)
                            g_enc.append(g_l_cos)
                    else:
                        for g in range(stop+1):
                            mu = 2 * g / stop - 1
                            g_l_sin = gaussian(l, mu, stddev / s) * np.sin((2 ** (s-1)) * np.pi * l)
                            g_l_cos = gaussian(l, mu, stddev / s) * np.cos((2 ** (s-1)) * np.pi * l)

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

                        xdash = (x/width)*2 -1
                        ydash = (y/height)*2 -1
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