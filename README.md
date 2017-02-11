# Variational Autoencoder

This is an implementation of a simple variational autoencoder which trains
on the MNIST dataset and generates similar images of digits. 

## Requirements:

1. CUDA toolkit 7.5
2. cuDNN v5
3. TensorFlow (https://github.com/tensorflow/tensorflow/tree/r0.10)
	from r0.10 branch, select binary compatible with the above.

## Instructions to run:
Run `python trainScriptClass.py`. It will train a simple 2 layer VAE generator with 2 layer encoder for training. The parameters are defined below:

1. `batch_size`: size of the training and testing batch (for batch size of 20, it nearly reached 11 GB on NVIDIA-Titan X Maxwell) 
2. `X_size`: size of the input (here it is the total number of pixels)
3. `hidden_enc_1_size`: hidden layer 1 size in the encoder
4. `hidden_enc_2_size`: hidden layer 2 size in the encoder
5. `hidden_gen_1_size`: hidden layer 1 size in the generator
6. `hidden_gen_2_size`: hidden layer 2 size in the generator
7. `z_size`: size of the latent variable 

The model trains with a default learning rate of 1e-4 using the adam optimizer.

The model is trained for 200000 iterations and 20 randomly generated samples and the checkpoint of the corresponding model are saved in `generated_class/' directory after every 50000 iterations.

## References:

1. This was implemented based on the Carl Doersch's tutorial available at: https://arxiv.org/abs/1606.05908
2. Another useful reference for implementing VAEs is: https://jmetzen.github.io/2015-11-27/vae.html

## Changelog:

Feb 10, 2017

* Added convolution + deconvolution based VAE
* Batch size is again fixed at initialization, have to alter the technique.

======================================

* Added support for a Beta weighting term in the KL Divergence loss
* Batch size is no longer fixed at initialization
* Added functions to encode a given x and decode a given z, and also to
  perform both these operations to generate an image "like" the given one.
* beta_trainScriptClass_conditional.py adds visualization of latent
  features