import tensorflow as tf
import numpy as np

def getWeight(shape):
    return tf.Variable(tf.random_uniform(shape, minval=-0.08, maxval=0.08, dtype=tf.float32))

def getBias(shape):
    # Initialize biases as 0
    return tf.Variable(tf.constant(0, shape=shape, dtype=tf.float32), dtype=tf.float32)

class vae(object):
    
    def __init__(self, params):
        # params consists of:
        # (1) X_size: number of image pixels
        # (2) z_size: latent variable size
        
        self.X_size = params['X_size']
        self.z_size = params['z_size']
        self.beta = params['beta']
        
    def _create_network_(self):
    
        self.X_placeholder = tf.placeholder(tf.float32,[None]+self.X_size+ [1])

        self.fc_size = 32*20*20 # Size to flatten to after 2nd convolution 
        self.batch_size = tf.shape(self.X_placeholder)[0]
        self.num_channels = 1#tf.shape(self.X_placeholder)[3]
        self.getEncoder()
        self.getLatentSampler()
        self.getGenerator()
        self.getReconstructionLoss()
        self.getKLDLoss()
        # average the total loss over all samples
        self.total_loss = tf.reduce_mean(self.rec_loss + self.kl_loss)

    def generateSample(self, sess, n_samples=20):
        z_rng = np.random.normal(size=[n_samples, self.z_size])
        return sess.run(self.output, feed_dict={self.z_sample: z_rng})
    
    def generateSampleConditional(self, sess, x) :
        return sess.run(self.output, feed_dict={self.X_placeholder: x})
    
    def encode(self,sess,x) :
        return sess.run(self.z_sample, feed_dict={self.X_placeholder: x})
    
    def get_conditional_params(self,sess,x) :
        return sess.run([self.mu_X, tf.exp(self.log_Sigma_X_diag)],feed_dict={self.X_placeholder: x})
    
    def encode_ML(self,sess,x) :
        '''
        Encode a given value of x to z by using the most likely z from Q(z|x)
        '''
        return sess.run(self.mu_X, feed_dict={self.X_placeholder: x})
    
    def decode(self,sess,z) :
        return sess.run(self.output, feed_dict={self.z_sample: z})

    def getEncoder(self):
            
        # Convolution layer 1
        self.conv_w_1 = getWeight([4, 4, 1, 64])
        self.conv_b_1 = getBias([64])
        conv_1 = tf.nn.relu(tf.nn.conv2d(self.X_placeholder, self.conv_w_1, \
                     strides=[1, 2, 2, 1], padding='VALID') + self.conv_b_1)
        
        # Convolution layer 2
        self.conv_w_2 = getWeight([3, 3, 64, 32])
        self.conv_b_2 = getBias([32])
        self.conv_2 = tf.nn.relu(tf.nn.conv2d(conv_1, self.conv_w_2, \
                     strides=[1, 2, 2, 1], padding='VALID')+self.conv_b_2)
        # Projection layer
        conv_2_reshaped = tf.reshape(self.conv_2, [-1, self.fc_size])

        # Get mu(X) and log_Sigma_diag(X)
        self.mu_weight = getWeight([self.fc_size, self.z_size])
        self.mu_bias = getBias([self.z_size])
        self.mu_X = tf.matmul(conv_2_reshaped, self.mu_weight) + self.mu_bias

        self.Sigma_weight = getWeight([self.fc_size, self.z_size])
        self.Sigma_bias = getBias([self.z_size])
        self.log_Sigma_X_diag = tf.matmul(conv_2_reshaped, self.Sigma_weight) + self.Sigma_bias

    def getLatentSampler(self):

        self.eps = tf.random_normal([self.batch_size, self.z_size], 0, 1, dtype=tf.float32)
        if tf.__version__ == '0.10.0':
            self.z_sample = tf.mul(tf.sqrt(tf.exp(self.log_Sigma_X_diag)), self.eps) + self.mu_X
        else:
            self.z_sample = tf.multiply(tf.sqrt(tf.exp(self.log_Sigma_X_diag)), self.eps) + self.mu_X


    def getGenerator(self):
        # Up-projection layer
        self.gen_proj_w = getWeight([self.z_size, self.fc_size])
        self.gen_proj_b = getBias([self.fc_size])
        gen_proj = tf.matmul(self.z_sample, self.gen_proj_w) + self.gen_proj_b
        self.z_batch_size=tf.shape(self.z_sample)[0]
        proj_reshaped = tf.nn.relu(tf.reshape(gen_proj, [self.z_batch_size, 20, 20, 32]))
        # Transposed Convolution layer 1
        # Note: conv2d_transpose expects the filter size to be specified
        # as (height, width, OUT_channels, IN_channels)
        self.trans_conv_w_1 = getWeight([3, 3, 64, 32])
        self.trans_conv_b_1 = getBias([64])
        trans_conv_1 = tf.nn.relu(tf.nn.conv2d_transpose(proj_reshaped, self.trans_conv_w_1, output_shape=[self.z_batch_size, 41, 41, 64], strides=[1, 2, 2, 1], padding='VALID') + self.trans_conv_b_1)

        # Transposed Convolution layer 2
        # Note: conv2d_transpose expects the filter size to be specified
        # as (height, width, OUT_channels, IN_channels)
        self.trans_conv_w_2 = getWeight([4, 4, 1, 64])
        self.trans_conv_b_2 = getBias([1])
        trans_conv_2 = tf.nn.conv2d_transpose(trans_conv_1, self.trans_conv_w_2, output_shape=[self.z_batch_size, 84, 84, 1], strides=[1, 2, 2, 1], padding='VALID')+self.trans_conv_b_2

        # Generated output
        self.output = tf.nn.sigmoid(trans_conv_2)

    def getReconstructionLoss(self):
        # obtain the negative log likelihood for bernoulli distribution
        # for each sample in the batch
        X_flat = tf.reshape(self.X_placeholder, [self.batch_size, -1])
        out_flat = tf.reshape(self.output, [self.batch_size, -1])
        self.rec_loss = -tf.reduce_sum(X_flat*tf.log(1e-10 + out_flat) + \
                                (1-X_flat)*tf.log(1e-10 + 1 - out_flat), 1)

    def getKLDLoss(self):
        # obtain the latent encoding error using KL Divergence
        # between the distribution over z = N(mu(X, Sigma(X)) and 
        # N(0, I) for each element in batch
        self.kl_loss = self.beta*tf.reduce_sum(tf.exp(self.log_Sigma_X_diag) + \
                                                                 tf.square(self.mu_X) - 1 - self.log_Sigma_X_diag, 1)

