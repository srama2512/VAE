import tensorflow as tf
import pdb
import numpy as np
import os
import scipy.misc
import argparse
import h5py
import sys
sys.path.append("misc/")
from MNIST_loader import MNIST_loader

parser = argparse.ArgumentParser()
parser.add_argument('--vae_model', default='mlp', help='[ mlp | cnn ]')
parser.add_argument('--save_dir', default='generated')
parser.add_argument('--input_h5', default='data/augmented_mnist.h5')
parser.add_argument('--beta', default=4, type=float)
parser.add_argument('--learning_rate', default=1e-3, type=float)
commandline_params = vars(parser.parse_args())
model_choice = commandline_params['vae_model']
if model_choice == 'mlp':
    import betavae_mlp as vae
elif model_choice == 'cnn':
    import betavae_cnn_aug as vae
else:
    raise ValueError('Wrong model choice for VAE!')


sess = tf.InteractiveSession()

tr_iters = 1000000

params = {}
params['z_size'] = 20
params['beta'] = commandline_params['beta']
params['batch_size'] = 20
if model_choice == 'mlp':
    params['X_size'] = 1600
    params['hidden_enc_1_size'] = 500
    params['hidden_enc_2_size'] = 200
    params['z_size'] = 20
    params['hidden_gen_1_size'] = 200
    params['hidden_gen_2_size'] = 500

# Import MNIST data
loader = MNIST_loader({'h5_file':commandline_params['input_h5'], 'batch_size':params['batch_size'], 'shufflle': 1})

params_generated = params

VAE = vae.vae(params)
VAE._create_network_()

# train_step = tf.train.AdamOptimizer(commandline_params['learning_rate']).minimize(VAE.total_loss)
train_step = tf.train.RMSPropOptimizer(commandline_params['learning_rate']).minimize(VAE.total_loss)

try:
    sess.run(tf.global_variables_initializer())
except AttributeError:
    # note @San: I've changed the `except` to `except AttributeError` so that it doesnt bypass future errors. In case this fails, change it back to `except`
    sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

for i in range(0, tr_iters):
    
    batch = loader.next_batch()
    if model_choice == 'cnn':
        _, loss_val = sess.run([train_step, VAE.total_loss], \
                               feed_dict = {VAE.X_placeholder: batch})
    else:
        inputReshaped = np.reshape(batch, (-1, 40*40))
        _, loss_val = sess.run([train_step, VAE.total_loss], \
                               feed_dict = {VAE.X_placeholder: inputReshaped})

    if i % 1000 == 0:
        print('Iter %.4d  Train Loss: %6.3f data.iter: %.6d data.total: %.6d'%(i+1, loss_val, loader.iterator, loader.total_size ))

    if (i+1) % 10000 == 0 or i == 0:
        generated = VAE.generateSample(sess, n_samples=params['batch_size'])
        os.system('mkdir -p %s/iter_%.6d'%(commandline_params['save_dir'], i+1))
        for im in range(params['batch_size']):
            reshaped_image = generated[im]
            reshaped_image = reshaped_image.reshape(40, 40)
            scipy.misc.toimage(reshaped_image, cmin=0.0, cmax=1.0).save('%s/iter_%.6d/img%.3d.png'%(commandline_params['save_dir'],i+1,im))

        save_path = saver.save(sess, '%s/iter_%.6d/checkpoint.ckpt'%(commandline_params['save_dir'],i+1))
        print('Saved model to %s'%(save_path))
