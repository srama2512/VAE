import tensorflow as tf
import pdb
import numpy as np
import os
import scipy.misc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vae_model', default='mlp', help='[ mlp | cnn ]')
parser.add_argument('--model_path', required=True)

commandline_params = vars(parser.parse_args())
model_choice = commandline_params['vae_model']
if model_choice == 'mlp':
    import betavae_mlp as vae
elif model_choice == 'cnn':
    import betavae_cnn as vae
else:
    raise ValueError('Wrong model choice for VAE!')

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)
sess = tf.InteractiveSession()

params = {}
params['z_size'] = 20
params['beta'] = 1
params['batch_size'] = 20
if model_choice == 'mlp':
    params['X_size'] = 784
    params['hidden_enc_1_size'] = 500
    params['hidden_enc_2_size'] = 200
    params['z_size'] = 20
    params['hidden_gen_1_size'] = 200
    params['hidden_gen_2_size'] = 500
else:
    params['X_size'] = [28,28]


params_generated = params

VAE = vae.vae(params)
VAE._create_network_()

try:
    sess.run(tf.global_variables_initializer())
except AttributeError:
    # note @San: I've changed the `except` to `except AttributeError` so that it doesnt bypass future errors. In case this fails, change it back to `except`
    sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
saver.restore(sess, commandline_params['model_path'])

# Examining latent variables
n_test_imgs = 100
# Sample some random images from the validtion set
rand_sample, rand_labels = mnist.validation.next_batch(n_test_imgs)
rand_labels = np.array([np.argmax(x) for x in rand_labels])
# Get a z for each image and then decode it back to an image
encoded = VAE.encode_ML(sess,np.reshape(rand_sample,(-1, 28, 28, 1)))
decoded = VAE.decode(sess,encoded)
#decoded = VAE.generateSampleConditional_ML(sess,rand_sample)
os.system('mkdir -p generated_class/latent')
latent_max = np.amax(np.abs(np.vstack(encoded)), axis=0)
# Organize generated images by label
for digit in range(10) :
    imlist = [np.zeros((1,28+28+7*params['z_size']))]
    for idx in np.where(rand_labels==digit)[0] :
        reshaped_image = rand_sample[idx].reshape(28, 28)
        decoded_image = decoded[idx].reshape(28,28)
        # Divide each latent feature by the maximum observed among the
        # encoded versions of the sample images and take abs to
        # normalize it to [0,1] range to visualize as a grayscale colour
        latent_vis = [ x*np.ones((28,7)) for x in abs(encoded[idx])/latent_max]
        imlist.append(np.concatenate([reshaped_image,decoded_image]+latent_vis,axis=1))
    scipy.misc.toimage(np.concatenate(imlist,axis=0), cmin=0.0, cmax=1.0).save('generated_class/latent/%d.png'%(digit))

