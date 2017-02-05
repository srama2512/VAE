import tensorflow as tf
import vae
import pdb
import numpy as np
import os
import scipy.misc

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

params = {}
params['batch_size'] = 20
params['X_size'] = 784
params['hidden_enc_1_size'] = 500
params['hidden_enc_2_size'] = 200
params['z_size'] = 20
params['hidden_gen_1_size'] = 200
params['hidden_gen_2_size'] = 500

params_generated = params

VAE = vae.vae(params)
VAE._create_network_()

train_step = tf.train.AdamOptimizer(1e-4).minimize(VAE.total_loss)

sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

for i in range(200000):
    
    batch = mnist.train.next_batch(params['batch_size'])
    _, loss_val = sess.run([train_step, VAE.total_loss], \
                           feed_dict = {VAE.X_placeholder: batch[0]})

    if i % 1000 == 0:
        print('Iteration %.4d  Train Loss: %6.3f'%(i+1, loss_val))

    if (i+1) % 50000 == 0:
        generated = VAE.generateSample(sess)
        os.system('mkdir -p generated_class/iter_%.6d'%(i+1))
        for im in range(params['batch_size']):
            reshaped_image = generated[im]
            reshaped_image = reshaped_image.reshape(28, 28)
            scipy.misc.toimage(reshaped_image, cmin=0.0, cmax=1.0).save('generated_class/iter_%.6d/img%.3d.jpg'%(i+1,im))

        save_path = saver.save(sess, 'generated_class/iter_%.6d/checkpoint.ckpt'%(i+1))
        print('Saved model to %s'%(save_path))
