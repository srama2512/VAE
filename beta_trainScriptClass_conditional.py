import tensorflow as tf
import betavae_mlp as vae
import pdb
import numpy as np
import os
import scipy.misc

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)
sess = tf.InteractiveSession()

tr_iters = 100000

params = {}
params['batch_size'] = 20
params['X_size'] = 784
params['hidden_enc_1_size'] = 500
params['hidden_enc_2_size'] = 200
params['z_size'] = 20
params['hidden_gen_1_size'] = 200
params['hidden_gen_2_size'] = 500
params['beta'] = 4

params_generated = params

VAE = vae.vae(params)
VAE._create_network_()

train_step = tf.train.AdamOptimizer(1e-4).minimize(VAE.total_loss)

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

for i in range(tr_iters):
    
    batch = mnist.train.next_batch(params['batch_size'])
    _, loss_val = sess.run([train_step, VAE.total_loss], \
                           feed_dict = {VAE.X_placeholder: batch[0]})

    if i % 1000 == 0:
        print('Iteration %.4d  Train Loss: %6.3f'%(i+1, loss_val))

    if (i+1) % 100000 == 0:
		# Generate some samples from the current model and store
        generated = VAE.generateSample(sess,params['batch_size'])
        os.system('mkdir -p generated_class/iter_%.6d'%(i+1))
        for im in range(params['batch_size']):
            reshaped_image = generated[im]
            reshaped_image = reshaped_image.reshape(28, 28)
            scipy.misc.toimage(reshaped_image, cmin=0.0, cmax=1.0).save('generated_class/iter_%.6d/img%.3d.png'%(i+1,im))
        save_path = saver.save(sess, 'generated_class/iter_%.6d/checkpoint.ckpt'%(i+1))
        print('Saved model to %s'%(save_path))


# Examining latent variables
n_test_imgs = 100
# Sample some random images from the validtion set
rand_sample, rand_labels = mnist.validation.next_batch(n_test_imgs)
rand_labels = np.array([np.argmax(x) for x in rand_labels])
# Get a z for each image and then decode it back to an image
encoded = VAE.encode_ML(sess,rand_sample)
decoded = VAE.decode(sess,encoded)
#decoded = VAE.generateSampleConditional_ML(sess,rand_sample)
os.system('mkdir -p generated_class_conditional/latent')
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
	scipy.misc.toimage(np.concatenate(imlist,axis=0), cmin=0.0, cmax=1.0).save('generated_class_conditional/latent/%d.png'%(digit))

