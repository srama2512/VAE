from skimage import transform
import numpy as np
from matplotlib import pyplot as plt
import pdb
import argparse
import h5py
import sys

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

parser = argparse.ArgumentParser()
parser.add_argument('--debug_mode', default=0, type=int)
parser.add_argument('--test_mode', default=0, type=int)
parser.add_argument('--rotation', default=1, type=int)
parser.add_argument('--translation_x', default=1, type=int)
parser.add_argument('--translation_y', default=1, type=int)
parser.add_argument('--scaling', default=1, type=int)
parser.add_argument('--save_path', default='data/augmented_mnist.h5')

commandline_params = vars(parser.parse_args())

mnist = input_data.read_data_sets("./data/", one_hot=True)

IMAGE_DIM = 28
# Zero pad the sides of images to 40x40 to allow
# zooming, free rotations and translations
CANVAS_DIM = 40
x_center = int((CANVAS_DIM/2)-1)
y_center = int((CANVAS_DIM/2)-1)
w_half = int(IMAGE_DIM/2)
h_half = int(IMAGE_DIM/2)
shift_y, shift_x = (CANVAS_DIM-1) / 2.,(CANVAS_DIM-1) / 2.   #Assumption: All images of same size

rotations = np.linspace(-180, 180, 30)
translations = np.linspace(-6, 6, 15)
scaling = np.arange(0.6,1.5,0.1)

num_trans = 0
trans_array = np.zeros(4, dtype=np.uint8)
if commandline_params['rotation'] == 1:
    num_trans += rotations.shape[0]
    trans_array[0]=1
if commandline_params['translation_x'] == 1:    
    num_trans += translations.shape[0]
    trans_array[1]=1
if commandline_params['translation_y'] == 1:
    num_trans += translations.shape[0]
    trans_array[2]=1
if commandline_params['scaling'] == 1:
    num_trans += scaling.shape[0]
    trans_array[3]=1

np.savetxt("transformations.txt", trans_array, fmt='%s', header = "Row1: Whether rotation was performed. Row2: translation_x. Row 3: translation_y. Row 4: scaling")

if commandline_params['debug_mode'] == 1:
    batch_size = 1
elif commandline_params['test_mode'] == 1:
    batch_size = 1000
else:
    # Use 55000 images from train
    batch_size = 55000

batch = mnist.train.next_batch(batch_size)
reshaped_input = np.reshape(batch[0], (-1, IMAGE_DIM, IMAGE_DIM))
augmented_inputs = np.zeros((reshaped_input.shape[0]*num_trans, CANVAS_DIM, CANVAS_DIM))

aug_counter = 0
for i in  range(batch_size):
    canvas_image = np.zeros((CANVAS_DIM, CANVAS_DIM))
    canvas_image[(y_center-h_half):(y_center+h_half), (x_center-w_half):(x_center+w_half)] = reshaped_input[i,:,:].copy()
    
    if commandline_params['rotation'] == 1:
        # Rotations
        for r in range(rotations.shape[0]):
            tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(rotations[r]))
            tf_shift = transform.SimilarityTransform(translation=[-shift_x, -shift_y])
            tf_shift_inv = transform.SimilarityTransform(translation=[shift_x, shift_y])
            
            image_rotated = transform.warp(canvas_image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse, order = 3)    
            augmented_inputs[aug_counter,:,:] = image_rotated.copy()
            aug_counter += 1
            if commandline_params['debug_mode'] == 1:
                plt.subplot(121)
                plt.imshow(canvas_image)
                plt.subplot(122)
                plt.imshow(image_rotated)
                plt.show()
  
    if commandline_params['translation_x'] == 1:
        # X translation
        for t in range(translations.shape[0]):
            tf_shift = transform.SimilarityTransform(translation=[translations[t], 0])
            image_x_trans = transform.warp(canvas_image, tf_shift.inverse, order = 3)   
            augmented_inputs[aug_counter,:,:] = image_x_trans.copy()
            aug_counter += 1
            if commandline_params['debug_mode'] == 1:
                plt.subplot(121)
                plt.imshow(canvas_image)
                plt.subplot(122)
                plt.imshow(image_x_trans)
                plt.show()
    
    if commandline_params['translation_y'] == 1: 
        # Y translation
        for t in range(translations.shape[0]):
            tf_shift = transform.SimilarityTransform(translation=[0, translations[t]])
            image_y_trans = transform.warp(canvas_image, tf_shift.inverse, order = 3)    
            augmented_inputs[aug_counter,:,:] = image_y_trans.copy()
            aug_counter += 1
            if commandline_params['debug_mode'] == 1:
                plt.subplot(121)
                plt.imshow(canvas_image)
                plt.subplot(122)
                plt.imshow(image_y_trans)
                plt.show()
    
    if commandline_params['scaling'] == 1: 
        # Scaling
        for s in range(scaling.shape[0]):
            
            scaled_input = transform.rescale(reshaped_input[i,:,:], scaling[s])
            h_curr = scaled_input.shape[0]
            w_curr = scaled_input.shape[1]
            h_half_curr = int(scaled_input.shape[0]/2)
            w_half_curr = int(scaled_input.shape[1]/2)
            canvas_image_scaled = np.zeros((CANVAS_DIM, CANVAS_DIM))
            canvas_image_scaled[(y_center-h_half_curr):(y_center+h_curr-h_half_curr), (x_center-w_half_curr):(x_center+w_curr-w_half_curr)] = scaled_input.copy()
            augmented_inputs[aug_counter,:,:] = canvas_image_scaled.copy()
            aug_counter += 1
            if commandline_params['debug_mode'] == 1:
                plt.subplot(121)
                plt.imshow(canvas_image)
                plt.subplot(122)
                plt.imshow(canvas_image_scaled)
                plt.show()
    if i%100 == 0:
        sys.stdout.write('=====> Images augmented: {:}\r'.format(i))
        sys.stdout.flush()
sys.stdout.write('=====> Images augmented: {:}\n'.format(batch_size))

if commandline_params['debug_mode'] == 0:
    h5_file = h5py.File(commandline_params['save_path'], 'w')
    augmented_inputs = augmented_inputs.reshape(augmented_inputs.shape[0],augmented_inputs.shape[1], augmented_inputs.shape[2],1)
    h5_file.create_dataset('augmented', augmented_inputs.shape, data = augmented_inputs)
    h5_file.close()
    print('Augmented data written to {:}'.format(commandline_params['save_path']))
else:
    print('Debug mode complete. No data will be saved')
