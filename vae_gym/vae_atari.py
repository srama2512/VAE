import tensorflow as tf
import pdb
import numpy as np
import numpy.random as rn
import os
import scipy.misc
import argparse
import pickle
import betavae_cnn_84 as vae

def sample_batch(frameslist,n=1) :
	return [frameslist[x] for x in rn.choice(len(frameslist),n)]

frames = pickle.load(open('frames.pkl','rb'))

sess = tf.InteractiveSession()

tr_iters = 200000


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

train_step = tf.train.AdamOptimizer(1e-4).minimize(VAE.total_loss)

try:
    sess.run(tf.global_variables_initializer())
except AttributeError:
    # note @San: I've changed the `except` to `except AttributeError` so that it doesnt bypass future errors. In case this fails, change it back to `except`
    sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

for i in range(tr_iters):
    
    batch = zip(*[(frames[x],x) for x in rn.choice(len(frames),params['batch_size'])])
    inputReshaped = np.reshape(batch[0], (-1, 84, 84, 1))
    _, loss_val = sess.run([train_step, VAE.total_loss], \
                               feed_dict = {VAE.X_placeholder: inputReshaped})

    if i % 1000 == 0:
        print('Iteration %.4d  Train Loss: %6.3f'%(i+1, loss_val))

    if (i+1) % 20000 == 0 or i == 0:
        generated = VAE.generateSample(sess, n_samples=10)
        os.system('mkdir -p output/generated_conv/iter_%.6d'%(i+1))
        for im in range(10):
            reshaped_image = generated[im]
            reshaped_image = reshaped_image.reshape(84, 84)
            scipy.misc.toimage(reshaped_image, cmin=0.0, cmax=1.0).save('output/generated_conv/iter_%.6d/img%.3d.png'%(i+1,im))

        #save_path = saver.save(sess, 'generated_conv/iter_%.6d/checkpoint.ckpt'%(i+1))
        #print('Saved model to %s'%(save_path))

# Examining latent variables
n_test_imgs_per_batch = 200
n_test_batches = 50
# Sample some random images from the validtion set
#rand_samples = [sample_batch(frames,n_test_imgs_per_batch) for i in range(n_test_batches)]
# Get a z for each image and then decode it back to an image
encoded=np.concatenate([VAE.encode_ML(sess,np.reshape(sample_batch(frames,n_test_imgs_per_batch),(-1, 84, 84, 1))) for i in range(n_test_batches)], axis = 0)
#encoded_stacked = (np.vstack(encoded))
latent_variance = np.var(encoded, axis=0)
print np.sort(latent_variance)

covmat = np.cov(encoded,rowvar=False)
scipy.misc.toimage(covmat, cmin=0.0, cmax=1.0).save('output/covariance.png')
