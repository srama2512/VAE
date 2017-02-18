# params['batch_size'] is changed

import argparse
import h5py
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import sys
sys.path.append("misc/")
import pdb
from MNIST_loader import MNIST_loader

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='generated/iter_050000/')
parser.add_argument('--input_h5', default='data/augmented_mnist.h5')
parser.add_argument('--vae_model', default='mlp', help='[ mlp | cnn ]')
parser.add_argument('--num_rotations', default=20, type=int)
parser.add_argument('--beta', default=4, type=float)

commandline_params = vars(parser.parse_args())
model_choice = commandline_params['vae_model']
if model_choice == 'mlp':
    import betavae_mlp as vae
elif model_choice == 'cnn':
    import betavae_cnn_aug as vae
else:
    raise ValueError('Wrong model choice for VAE!')

params = {}
params['z_size'] = 20
params['beta'] = commandline_params['beta']
params['batch_size'] = 30
if model_choice == 'mlp':
    params['X_size'] = 1600
    params['hidden_enc_1_size'] = 500
    params['hidden_enc_2_size'] = 200
    params['z_size'] = 20
    params['hidden_gen_1_size'] = 200
    params['hidden_gen_2_size'] = 500

# Import MNIST data. For testing, reading sequentially without shuffling
loader = MNIST_loader({'h5_file':commandline_params['input_h5'], 'batch_size':params['batch_size'], 'shufflle': 0})

sess = tf.Session()

VAE = vae.vae(params)
VAE._create_network_()

try:
    sess.run(tf.global_variables_initializer())
except AttributeError:
    # note @San: I've changed the `except` to `except AttributeError` so that it doesnt bypass future errors. In case this fails, change it back to `except`
    sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
chkpt = tf.train.get_checkpoint_state(commandline_params['checkpoint_path'])

if chkpt and chkpt.model_checkpoint_path:
    saver.restore(sess, chkpt.model_checkpoint_path)
else:
    print('No checkpoint found')


n_iter = 500
std_devs = np.zeros((n_iter, params['z_size']))
z_mean_theta = np.zeros((commandline_params['num_rotations'], params['z_size'])) # Computes mean z values at each angle
z_var = np.zeros((n_iter, params['z_size']))
print(std_devs.shape)
for iter in range(n_iter):

    batch = loader.next_batch()
    latent_mean, latent_variance = VAE.getMeanVariance(sess=sess, x=batch)

    for idx in range(params['batch_size']):
        #plt.imshow(batch[idx][:,:,0])
        #plt.show()
        #reshaped_input = np.reshape(batch[idx], (-1, 40*40))
        #latent_var_activation[idx] = VAE.encode(sess=sess, x=batch[idx])
        z_var[idx] += latent_variance[idx,:]
        z_mean_theta[idx] += latent_mean[idx,:]

    std_devs[iter,:] = np.std(latent_mean, axis=0)

z_mean_theta /= n_iter
z_var = np.mean(z_var, axis=0)/n_iter
plt.style.use('ggplot')
thetas = np.linspace(-180, 180, commandline_params['num_rotations'])
plt.figure(1)
mean_std_devs = np.mean(std_devs, axis=0)
for i in range(params['z_size']):
    z_across_theta = z_mean_theta[:, i]
    plt.subplot(params['z_size']/4, 4, i+1)
    plt.plot(thetas, z_across_theta)
    plt.ylabel('z%d: %.3f'%(i+1, z_var[i]))

plt.xlabel('rotation angle in degrees')
plt.savefig('rotation_mean_beta_{:}.png'.format(params['beta']))
plt.figure(2)
plt.plot(mean_std_devs)
plt.savefig('beta_{:}.png'.format(params['beta']))
plt.show()
