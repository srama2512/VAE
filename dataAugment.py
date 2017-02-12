from skimage import transform
import numpy as np
from matplotlib import pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

IMAGE_DIM = 28

shift_y, shift_x = (IMAGE_DIM-1) / 2.,(IMAGE_DIM-1) / 2.   #Assumption: All images of same size
tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(60))
tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])

batch_size = 2

batch = mnist.train.next_batch(batch_size)
reshaped_input = np.reshape(batch[0], (-1, IMAGE_DIM, IMAGE_DIM, 1))
for i in  range(batch_size):
    image_rotated = transform.warp(reshaped_input[i,:,:,0], (tf_shift + (tf_rotate + tf_shift_inv)).inverse, order = 3)    
    plt.subplot(121)
    plt.imshow(reshaped_input[i,:,:,0])
    plt.subplot(122)
    plt.imshow(image_rotated)
    plt.show()

