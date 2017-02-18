import tensorflow as tf
import numpy as np

def getBias(shape):
    # Initialize biases as 0
    return tf.Variable(tf.constant(0, shape=shape, dtype=tf.float32), dtype=tf.float32)

class vae(object):
    
    def __init__(self, params):
        # params consists of:
        # (1) hidden_enc_1_size: 1st hidden layer size of encoder
        # (2) hidden_enc_2_size: 2nd hidden layer size of encoder
        # (3) hidden_gen_1_size: 1st hidden layer size of generator
        # (4) hidden_gen_2_size: 2nd hidden layer size of generator
        # (5) X_size: number of image pixels
        # (6) z_size: latent variable size
        # (7) batch_size: size of training batch

        self.hidden_enc_1_size = params['hidden_enc_1_size']
        self.hidden_enc_2_size = params['hidden_enc_2_size']
        self.hidden_gen_1_size = params['hidden_gen_1_size']
        self.hidden_gen_2_size = params['hidden_gen_2_size']
        self.X_size = params['X_size']
        self.z_size = params['z_size']
        #self.batch_size = params['batch_size']
        self.beta = params['beta']

    def _create_network_(self):

        self.X_placeholder = tf.placeholder(tf.float32,[None, self.X_size])
        self.batch_size = tf.shape(self.X_placeholder)[0]
        self.getEncoder()
        self.getLatentSampler()
        self.getGenerator()
        self.getReconstructionLoss()
        self.getKLDLoss()

        # average the total loss over all samples
        self.total_loss = tf.reduce_mean(self.rec_loss + self.kl_loss)

    def generateSample(self, sess, n_samples = 20):
        '''
        Choose random z and return decoded image(s)
        '''
        z_rng = np.random.normal(size=[n_samples, self.z_size])
        return sess.run(self.output, feed_dict={self.z_sample: z_rng})
    
    
    def generateSampleConditional(self, sess, x) :
        '''
        Sample z from Q(z|x) and return decoded image
        '''
        return sess.run(self.output, feed_dict={self.X_placeholder: x})
    
    
    def generateSampleConditional_ML(self, sess, x) :
        '''
        Use the most likely z for given x to generate a sample instead of
        sampling from Q(z|x)
        '''
        return self.decode(sess,self.encode_ML(sess,x))
    
    
    def encode(self,sess,x) :
        '''
        Encode a given value of x to z by sampling z from Q(z|x)
        '''
        return sess.run(self.z_sample, feed_dict={self.X_placeholder: x})
    
    
    def encode_ML(self,sess,x) :
        '''
        Encode a given value of x to z by using the most likely z from Q(z|x)
        '''
        return sess.run(self.mu_X, feed_dict={self.X_placeholder: x})
    
    
    def decode(self,sess,z) :
        '''
        Generate x for a given z
        '''
        return sess.run(self.output, feed_dict={self.z_sample: z})

    def getEncoder(self):
            
        # Hidden layer 1
        self.enc_h_1_w = tf.get_variable("enc_h1_w",shape=[self.X_size, self.hidden_enc_1_size], initializer=tf.contrib.layers.xavier_initializer())
        self.enc_h_1_b = getBias([self.hidden_enc_1_size])
        hidden_1 = tf.nn.relu(tf.matmul(self.X_placeholder, self.enc_h_1_w) \
                           + self.enc_h_1_b)

        # Hidden layer 2
        self.enc_h_2_w = tf.get_variable("enc_h2_w",shape=[self.hidden_enc_1_size, self.hidden_enc_2_size], initializer=tf.contrib.layers.xavier_initializer())
        self.enc_h_2_b = getBias([self.hidden_enc_2_size])
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, self.enc_h_2_w) + self.enc_h_2_b)

        # Get mu(X) and log_Sigma_diag(X)
        self.mu_weight = tf.get_variable("enc_mu_w", shape=[self.hidden_enc_2_size, self.z_size], initializer=tf.contrib.layers.xavier_initializer())
        self.mu_bias = getBias([self.z_size])
        self.mu_X = tf.matmul(hidden_2, self.mu_weight) +  self.mu_bias

        self.Sigma_weight = tf.get_variable("sigma_w", shape=[self.hidden_enc_2_size, self.z_size], initializer=tf.contrib.layers.xavier_initializer())
        self.Sigma_bias = getBias([self.z_size])
        self.log_Sigma_X_diag = tf.matmul(hidden_2, self.Sigma_weight) + self.Sigma_bias

    def getLatentSampler(self):
        self.eps = tf.random_normal([self.batch_size, self.z_size], 0, 1, dtype=tf.float32)
        if tf.__version__ == '0.10.0':
            self.z_sample = tf.mul(tf.sqrt(tf.exp(self.log_Sigma_X_diag)), self.eps) + self.mu_X
        else:
            self.z_sample = tf.multiply(tf.sqrt(tf.exp(self.log_Sigma_X_diag)), self.eps) + self.mu_X

    def getGenerator(self):
        # Hidden layer 1
        self.gen_h_1_w = tf.get_variable("gen_h1_w", shape=[self.z_size, self.hidden_gen_1_size], initializer=tf.contrib.layers.xavier_initializer())
        self.gen_h_1_b = getBias([self.hidden_gen_1_size])
        hidden_1 = tf.nn.relu(tf.matmul(self.z_sample, self.gen_h_1_w) + self.gen_h_1_b)

        # Hidden layer 2
        self.gen_h_2_w = tf.get_variable("gen_h2_w", shape=[self.hidden_gen_1_size, self.hidden_gen_2_size], initializer=tf.contrib.layers.xavier_initializer())
        self.gen_h_2_b = getBias([self.hidden_gen_2_size])
        hidden_2 = tf.nn.relu(tf.matmul(hidden_1, self.gen_h_2_w) + self.gen_h_2_b)

        # Generated output
        self.output_weight = tf.get_variable("gen_op_w", shape=[self.hidden_gen_2_size, self.X_size], initializer=tf.contrib.layers.xavier_initializer())
        self.output_bias = getBias([self.X_size])
        self.output = tf.nn.sigmoid(tf.matmul(hidden_2, self.output_weight) + self.output_bias)

    def getReconstructionLoss(self):
        '''
        Obtain the negative log likelihood for bernoulli distribution
        for each sample in the batch
        '''
        self.rec_loss = -tf.reduce_sum(self.X_placeholder*tf.log(1e-10 + self.output)  \
                                                + (1-self.X_placeholder)*tf.log(1e-10 + 1 - self.output), 1)

    def getKLDLoss(self):
        '''
        Obtain the latent encoding error using KL Divergence
        between the distribution over z = N(mu(X, Sigma(X)) and 
        N(0, I) for each element in batch
        '''
        self.kl_loss = self.beta*tf.reduce_sum(tf.exp(self.log_Sigma_X_diag) + \
                                                                 tf.square(self.mu_X) - 1 - self.log_Sigma_X_diag, 1)
    
    def getMeanVariance(self,sess,x) :
        mu, log_sigma = sess.run([self.mu_X, self.log_Sigma_X_diag], feed_dict={self.X_placeholder: x})
        return mu, np.exp(log_sigma)