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
commandline_params = vars(parser.parse_args())


def sample_batch(frameslist, perm, n=1) :
	return [frameslist[perm[x]]/np.float32(255) for x in rn.choice(len(perm),n)]

frames = pickle.load(open('frames.pkl','rb'))
n_total = len(frames)
n_valid = int(.2*n_total)
perm = rn.choice(n_total,n_total)
perm_train = perm[:-n_valid]
perm_valid = perm[-n_valid:]

sess = tf.InteractiveSession()

tr_iters = 2000


params = {}
params['batch_size'] = 20
params['X_size'] = [84,84]
params['hidden_enc_1_size'] = 500
params['hidden_enc_2_size'] = 200
params['z_size'] = 30
params['hidden_gen_1_size'] = 200
params['hidden_gen_2_size'] = 500
params['beta'] = 1.28

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
scipy.misc.toimage(covmat, cmin=0.0, cmax=1.0).save('output/covariance.png')
