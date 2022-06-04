from scipy import signal, special 
from PIL import Image
from nerf2D import Positional_Encoding
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
import torch
import torch.nn as nn
from torch_network import Net
import argparse

def main():

	parser = argparse.ArgumentParser(description='Blah.')
	parser.add_argument('--encoding', type=str, default='gauss',
            help=f'Positional encoding, one of {Positional_Encoding.available_encodings}')
	parser.add_argument('--neurons', type=int, default=128)
	parser.add_argument('--image', type=str, default='fractal', help='Image to learn')
	parser.add_argument('--save_img', type=int, default=0, help='save jpg')
	parser.add_argument('--save_fourier', type=int, default=0, help='save fourier transform image')
	parser.add_argument('--L', type=int, default=10, help='save fourier transform image')
	args = parser.parse_args()	

	# Code is a mess right now, this line is just for displaying the image
	orig_image = Image.open(f'dataset/{args.image}.jpg')
	orig_image = np.asarray(orig_image)
	orig_image = orig_image / 255.0

	# This is to actually pass the image through the network
	im = Image.open(f'dataset/{args.image}.jpg')
	im2arr = np.array(im) 

	# Get the encoding
	testimg = im2arr 
	testimg = testimg / 255.0  
	H, W, C = testimg.shape

	PE = Positional_Encoding(testimg, args.encoding, training=False)

	# Get data in encoded format
	inp_batch, inp_target, ind_vals = PE.get_dataset(L=args.L)
	inp_batch = torch.Tensor(inp_batch)

	# send through model
	if args.encoding == 'gauss':
		model = Net(inp_batch.shape[1], args.neurons)
		model.load_state_dict(torch.load(f'./saved_model/torch_{args.encoding}_{args.image}.pth'))
		with torch.no_grad():
			output = model(inp_batch)
	elif args.encoding == 'sin_cos':
		model = Net(inp_batch.shape[1], args.neurons)
		model.load_state_dict(torch.load(f'./saved_model/torch_{args.encoding}_{args.image}.pth'))
		with torch.no_grad():
			output = model(inp_batch)

	# Display the image from the model
	predicted_image = np.zeros_like(orig_image)

	# preprocess image before displaying
	indices = ind_vals.astype('int')
	indices = indices[:, 1] * orig_image.shape[1] + indices[:, 0]

	np.put(predicted_image[:, :, 0], indices, np.clip(output[:, 0], 0, 1))
	np.put(predicted_image[:, :, 1], indices, np.clip(output[:, 1], 0, 1))
	np.put(predicted_image[:, :, 2], indices, np.clip(output[:, 2], 0, 1))
	# Gaussian encoding has different preprocessing
	if args.encoding == 'gauss':
		predicted_image = np.transpose(predicted_image, (1, 0, 2))

	if args.save_img:
		plt.imsave(f'{args.model}_{args.image}.jpg', predicted_image)
		print("saved image")
	# Display predicted image
	plt.imshow(predicted_image)
	plt.show()

	# gray scale the images
	grey_scale_image = rgb2gray(predicted_image)
	grey_scale_orig = rgb2gray(orig_image)

	plt.imshow(grey_scale_image, cmap='gray')
	plt.show()

	plt.imshow(grey_scale_orig, cmap='gray')
	plt.show()

	# fourier transform
	fourier_image = np.fft.fftshift(np.fft.fft2(grey_scale_image))
	fourier_orig = np.fft.fftshift(np.fft.fft2(grey_scale_orig))

	plt.imshow(np.log(abs(fourier_image)), cmap='gray')
	plt.show()

	plt.imshow(np.log(abs(fourier_orig)), cmap='gray')
	plt.show()

	print("Average frequency error: ")
	print(np.sum(np.abs(np.log(abs(fourier_image)) - np.log(abs(fourier_orig))))/(256*256))

	if args.save_fourier:
		plt.imsave(f'{args.model}_{args.image}_fourier_transform.jpg', np.log(abs(fourier_image)), cmap='gray')
		plt.imsave(f'{args.image}_fourier_transform.jpg', np.log(abs(fourier_orig+1e-06)), cmap='gray')
		print("saved fourier images")

if __name__ == '__main__':
	main()