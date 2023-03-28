# Understanding the Spectral Bias of Coordinate Based MLPs Via Training Dynamics

## Main Text

This project contains the code necessary to replicate the experiments for the paper "Understanding the Spectral Bias of Coordinate Based MLPs Via Training Dynamics". In the main text, we used a 64x64 random image to conduct all experiments, although this can be manually changed if wished. To visualize the activation regions, run `gradient_confusion.py --visualize_regions True`, and to count the the number of regions as training progresses, run `gradient_confusion.py --act_patterns True`. Most experiments come from this file, and to get the mean hamming distance and confusion densities locally and globally, run `--mean_hamming_in_region True`, `--mean_hamming_between_region True`, `--confusion_in_region True`, and `--confusion_between_region True`. The hamming distance will be calculated every 100 epochs, and the confusion densities are at the final epoch, therefore the number of epochs specified will give the confusion at that moment in training. The default network width, epochs, batch size, learning rate, and L value are those used in the main text, but can be changed through setting `--epochs [num]`, `--batch_size [num]`, `--neurons [num]`, and `--L [num]`. The depth and learning rate can be changed manually within the scripts if needed. To change the normalizing interval of coordinates from [0,1] to [-1,1], use `--negative True`. For parameter norms and dead ReLU neurons, use `--param_norms True` and `--dead_neurons True`, although the learning rate used for the parameter norms in the main text is .005 instead of .001.
For the hyperplane angle experiment, run `python hyperplane_angles.py`, and change the L value and epochs if needed within the script. To get the 2D slice of activation regions, run `python cross_section.py` and specify the components using `--frequency_slice [str]`, `--L [num]`, and `--first_layer [bool]`. Lastly, running `python region_narrowness.py` will automatically run the distance to boundary experiment with full batch gradient descent, and the L value will need to be specified. Note that the plots from these experiments will need to be generated manually in the script, or saved to a file and plotted elsewhere to generate standard deviations after multiple runs. 

## Appendix

For experiments done in the appendix, there is a script `gradient_confusion_image.py` that follows the same settings as above but is taylored for natural images. The different images provided and which one to use can be changed within the script. Lastly, `coordinate_networks.py` is similar to the script above as well, but allows for the specification of layers within the network for the relative hamming distances. Again, in order to actually plot the results from the scripts, this will need to be done manually using matplotlib or sns.kdeplot for confusion.

## Additional Information

The script `torch_network.py` contains the network used by the gradient confusion script. This can also be used to run a model on a natural image and save the network. You can then load in the results from this network in `torch_activations.py` to visualize its performance. The script `nerf2D.py` contains the encodings and gathers the coordinates and target values from the signal provided so the batches can be generated.

# Requirements

* `PyTorch 1.11`
* `Matplotlib 3.5.1`
* `PIL 9.0.1`
* `skimage 0.19.2`
* `scipy 1.7.3`
