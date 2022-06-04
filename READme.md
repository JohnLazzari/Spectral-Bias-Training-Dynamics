# Gabor Positional Encoding

This project contains the code necessary to replicate the experiments for the Gabor encoding proposed in "Joint Localization Reduces Artifacts of Neural Networks in Low Dimensional Domains". For most scripts there are arguments that can be passed in to change the batch size, learning rate, neurons, encoding, and image to train on. The script `nerf2D.py` contains the positional encoding object used for all experiments, and `kernel_visualization.py` can be run to visualize the kernels on a 28x28 grid. The Fourier features are referred to as "sin_cos" and the gabor encoding is referred to as "gauss", so these are the names given when selecting which encoding to use in the scripts. Note that the default parameters are set to those used in the paper, and instructions on how to replicate the results are given, although different sized networks, batches, learning rates, images, and L values can be tested for each task as well.

## Images

In this paper, we demonstrate the benefits of the Gabor encoding through the image regression task, in which we predict the RGB value by passing in the corresponding coordinates through an MLP. Our dataset contains 15 different images with 5 at each scale, the list is below.

### 256x256
* fractal.jpg https://github.com/ankurhanda/nerf2D
* robot.jpg https://github.com/ankurhanda/nerf2D
* salad.jpg https://github.com/ankurhanda/nerf2D
* cool_cows.jpg http://hof.povray.org/vaches.html
* glasses.jpg http://hof.povray.org/glasses.html

### 512x512
* jet.jpg https://sipi.usc.edu/database/database.php?volume=misc
* splash.jpg https://sipi.usc.edu/database/database.php?volume=misc
* peppers.jpg https://sipi.usc.edu/database/database.php?volume=misc
* forest.jpg https://www.timeforkids.com/k1/fantastic-forests/
* ocean.jpg https://www.istockphoto.com/photos/ocean

### 768x768
* ball.jpg https://www.setaswall.com/1024x1024-wallpapers/
* fruit.jpg https://www.setaswall.com/1024x1024-wallpapers/
* food.jpg https://www.setaswall.com/1024x1024-wallpapers/
* plant.jpg https://www.setaswall.com/1024x1024-wallpapers/
* flower.jpg https://www.setaswall.com/1024x1024-wallpapers/

## Overfitting

To replicate the overfitting experiment done in the paper (Figure 5), run `overfitting.py` using the default batch size, learning rate, epochs, and neurons with the selected images `--image fractal` (a), `--image peppers` (b), `--image ball` (c).

## Batch Loss

In Figure 3, we demonstrate the training stabilization the Gabor encoding induces. To replicate these plots, run `batch_loss.py` using the default arguments, which trains on the fractal image.

## Generalization

For the generalization task, we train on each image individually then take the average PSNR for each encoding across all images at the respective scale. To get PSNR for an image, run `generalization.py --image [image]`, using `--fourier_l 6` and `--gabor_l 5` for 256x256 images,  `--fourier_l 7` and `--gabor_l 6` for 512x512 images, and `--fourier_l 8` and `--gabor_l 7` for 768x768 images. These can be shown to work the best for each encoding through the overfitting experiment. Note that in this case, overfitting does not depend on the target signal but rather how fast we can sample in the normalized interval [-1, 1], thus these L values should work for any image. To run the encodings in which the frequencies were sampled from a normal distribution, change sin_cos in the script to gauss_sin_cos, and use the flag `--RFF True` for the gabor encoding. These were tested each using an L value of 10, which in this case does not determine the frequency but rather the amount of frequency components (amount of sinusoidal components). For both encodings sampled from a normal distribution, the best standard deviation for the gaussian was found to be 10, and is consistent for each image (can be changed in nerf2D.py). The overall results should be similar to the paper, where the standard deviation of the PSNR is taken across each image.

|                       | 256x256 PSNR     | 512x512 PSNR     | 768x768 PSNR     |
| --------------------- | ---------------- | ---------------- | ---------------- |
| No Mapping            | 17.39 $\pm$ 1.56 | 22.51 $\pm$ 2.48 | 22.45 $\pm$ 6.3  |
| Fourier               | 20.81 $\pm$ 1.41 | 26.55 $\pm$ 1.8  | 26.47 $\pm$ 3.95 |
| Fourier-$\mathcal{N}$ | 21.13 $\pm$ 1.61 | 27.64 $\pm$ 1.45 | 28.20 $\pm$ 6.39 |
| Gabor                 | 21.86 $\pm$ 1.9  | 28.63 $\pm$ 1.58 | 30.08 $\pm$ 6.38 |
| Gabor-$\mathcal{N}$   | 21.20 $\pm$ 1.87 | 28.90 $\pm$ 1.75 | 30.43 $\pm$ 6.16 |

## Visualization

To train a model and save it, run `torch_network.py` using the arguments of choice, which will save the model as 'torch_{encoding}_{image}.pth' (should take less than a minute on GPU). To then visualize the model's output, run `torch_activations.py` selecting the appropriate encoding `--encoding [encoding]`, image `--image [image]` and L value used when training `--L [frequency]` to then display the model's reconstructed image. This is done as a demonstration of the reduction of artifacts and overall faster convergence to high frequency components when using the Gabor encoding, not in terms of generalization, and the reconstructed images shown in the paper (Figure 1) were trained using batch gradient descent with an L value of 10 on these scripts for both encodings.

# Requirements

* `PyTorch 1.11`
* `Matplotlib 3.5.1`
* `PIL 9.0.1`
* `skimage 0.19.2`
* `scipy 1.7.3`
