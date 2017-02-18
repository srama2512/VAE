import tensorflow as tf
import pdb
import numpy as np
import numpy.random as rn
import os
import scipy.misc
import argparse
import pickle
import betavae_cnn_84 as vae

def sample_batch(frameslist, perm, n=1) :
	return [frameslist[perm[x]]/np.float32(255) for x in rn.choice(len(perm),n)]

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', default='frames.pkl')
commandline_params = vars(parser.parse_args())

frames=[]
with open(commandline_params['data_file'], 'r') as f:
    frames = pickle.load(f)
f.close()

n_total = len(frames)
n_valid = int(.2*n_total)
perm = rn.choice(n_total,n_total)
perm_train = perm[:-n_valid]
perm_valid = perm[-n_valid:]

sess = tf.InteractiveSession()

tr_iters = 10000


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
    sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()

for i in range(tr_iters):
    
    batch = sample_batch(frames, perm_train, params['batch_size'])
    inputReshaped = np.reshape(batch, (-1, 84, 84, 1))
    _, loss_val = sess.run([train_step, VAE.total_loss], \
                               feed_dict = {VAE.X_placeholder: inputReshaped})

    if i % 1000 == 0:
        print('Iteration %.4d  Train Loss: %6.3f'%(i+1, loss_val))

    if (i+1) % 50000 == 0 or i == 0:
        generated = VAE.generateSample(sess, n_samples=30)
        os.system('mkdir -p output/generated_conv/iter_%.6d'%(i+1))
        save_path = saver.save(sess, 'output/generated_conv/iter_%.6d/checkpoint.ckpt'%(i+1))
        #print('Saved model to %s'%(save_path))
os.system('mkdir -p output/generated_conv/iter_%.6d'%(i+1))        
save_path = saver.save(sess, 'output/generated_conv/iter_%.6d/checkpoint.ckpt'%(i+1))
