import h5py
import numpy as np

class MNIST_loader(object):

    def __init__(self, params):
        self.h5_file = h5py.File(params['h5_file'], 'r')
        self.batch_size = params.get('batch_size', 20)
        self.total_size = self.h5_file['augmented'].shape[0]
        self.shuffle = params.get('shuffle',  0)
        self.data = np.array(self.h5_file['augmented'])
        if self.shuffle == 1:
            # self.data = np.random.shuffle(self.data)
            self.data = self.data[np.random.permutation(np.arange(0,self.total_size,1))]

        self.shapes = self.data.shape[1:]
        self.shapes = (self.batch_size, self.shapes[0], self.shapes[1], self.shapes[2])
        self.iterator = 0

    def next_batch(self):
        data_next = np.zeros(self.shapes)
        if self.iterator + self.batch_size - 1 < self.total_size:
            data_next[:, :, :, :] = np.array(self.data[self.iterator:(self.iterator+self.batch_size), :, :, :])
            self.iterator += self.batch_size
            if self.iterator >= self.total_size:
                self.iterator = 0
        else:
            num_remains = self.batch_size - (self.total_size - self.iterator+1)
            data_next[:(self.total_size-self.iterator+1), :, :, :] = np.array(self.data[self.iterator:self.total_size, :, :, :])
            # Wrap around
            data_next[(self.total_size-self.iterator+1):, :, :, :] = np.array(self.data[0:num_remains, :, :, :])
            self.iterator = num_remains

        return data_next
        
