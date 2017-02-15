import argparse
import h5py
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='data/augmented_mnist.h5')

params = vars(parser.parse_args())

h5_file = h5py.File(params['path'], 'r')
aug_data = h5_file['augmented'][:]
for idx in range(aug_data.shape[0]):
    plt.imshow(aug_data[idx][:,:,0])
    plt.show()