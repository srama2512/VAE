import tensorflow as tf
import pdb
import numpy as np
import os
import scipy.misc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--vae_model', default='mlp', help='[ mlp | cnn ]')

commandline_params = vars(parser.parse_args())
model_choice = commandline_params['vae_model']
if model_choice == 'mlp':
    import betavae_mlp as vae
elif model_choice == 'cnn':
    import betavae_cnn as vae
else:
    raise ValueError('Wrong model choice for VAE!')

tr_iters = 200000

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
    
    batch = mnist.train.next_batch(params['batch_size'])
    if model_choice == 'cnn':
        inputReshaped = np.reshape(batch[0], (-1, 28, 28, 1))
        _, loss_val = sess.run([train_step, VAE.total_loss], \
                               feed_dict = {VAE.X_placeholder: inputReshaped})
    else:
        _, loss_val = sess.run([train_step, VAE.total_loss], \
                               feed_dict = {VAE.X_placeholder: batch[0]})

    if i % 1000 == 0:
        print('Iteration %.4d  Train Loss: %6.3f'%(i+1, loss_val))

    if (i+1) % 50000 == 0 or i == 0:
        generated = VAE.generateSample(sess, n_samples=params['batch_size'])
        os.system('mkdir -p generated_conv/iter_%.6d'%(i+1))
        for im in range(params['batch_size']):
            reshaped_image = generated[im]
            reshaped_image = reshaped_image.reshape(28, 28)
            scipy.misc.toimage(reshaped_image, cmin=0.0, cmax=1.0).save('generated_conv/iter_%.6d/img%.3d.png'%(i+1,im))

        save_path = saver.save(sess, 'generated_conv/iter_%.6d/checkpoint.ckpt'%(i+1))
        print('Saved model to %s'%(save_path))
