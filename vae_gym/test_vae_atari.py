import tensorflow as tf
import pdb
import numpy as np
import numpy.random as rn
import os
import scipy.misc
import argparse
import pickle
import betavae_cnn_84 as vae

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default='output/generated_conv/iter_010000/')
parser.add_argument('--aux_data', default='output/aux_data.pkl')
parser.add_arugment('--batch_size', default=-1, type=int) # -1 implies it will use train batch size itself 
parser.add_argument('--dump_path', default='output')

commandline_params = vars(parser.parse_args())
dump_path = commandline_params['dump_path']

def sample_batch(frameslist, perm, n=1) :
    return [frameslist[perm[x]]/np.float32(255) for x in rn.choice(len(perm),n)]

aux_data = pickle.load(commandline_params['aux_data'])
magic_seed_number = aux_data['magic_seed_number']
rn.seed(magic_seed_number) # Seed chosen by die rolls. Guaranteed to be random

frames = pickle.load(open('frames.pkl','rb'))
n_total = len(frames)
n_valid = int(.2*n_total)
#perm = rn.choice(n_total,n_total)
#perm_train = perm[:-n_valid]
#perm_valid = perm[-n_valid:]
perm_train = aux_data['perm_train']
perm_valid = aux_data['perm_valid']

sess = tf.InteractiveSession()

tr_iters = 2000

params = aux_data['params']
if commandline_params['batch_size'] != -1:
    params['batch_size'] = commandline_params['batch_size']

params_generated = params

VAE = vae.vae(params)
VAE._create_network_()

try:
    sess.run(tf.global_variables_initializer())
except AttributeError:
    sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
chkpt = tf.train.get_checkpoint_state(commandline_params['checkpoint_path'])

if chkpt and chkpt.model_checkpoint_path:
    saver.restore(sess, chkpt.model_checkpoint_path)
else:
    print 'No checkpoint found'

generated = VAE.generateSample(sess, n_samples=30)
for im in range(30):
    reshaped_image = generated[im]
    reshaped_image = reshaped_image.reshape(84, 84)
    scipy.misc.toimage(reshaped_image, cmin=0.0, cmax=1.0).save(commandline_params['checkpoint_path']+'img%.3d.png'%(im))

# Examining latent variables
n_test_imgs_per_batch = 200
n_test_batches = 50
# Sample some random images from the validtion set
#rand_samples = [sample_batch(frames,n_test_imgs_per_batch) for i in range(n_test_batches)]
# Get a z for each image and then decode it back to an image
encoded=np.concatenate([VAE.encode_ML(sess,np.reshape(sample_batch(frames, perm_valid, n_test_imgs_per_batch),(-1, 84, 84, 1))) for i in range(n_test_batches)], axis = 0)
#encoded_stacked = (np.vstack(encoded))
latent_variance = np.var(encoded, axis=0)
print np.sort(latent_variance)

covmat = np.cov(encoded,rowvar=False)
scipy.misc.toimage(covmat, cmin=0.0, cmax=1.0).save('%s/covariance.png'%(dump_path))
