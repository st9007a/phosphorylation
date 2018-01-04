#!/usr/bin/python3
import numpy as np

src = 'data/baseline/'

class Batcher():

    file_x = src + 'train_x.npy'
    file_y = src + 'train_y.npy'

    def __init__(self, batch_size):
        self.times = 0

        self.x = np.load(Batcher.file_x)
        self.y = np.load(Batcher.file_y)
        self.batch_size = batch_size

        self.combined = np.c_[self.x.reshape(len(self.x), -1), self.y.reshape(len(self.y), -1)]
        self.x1 = self.combined[:, :self.x.size//len(self.x)].reshape(self.x.shape)
        self.y1 = self.combined[:, self.x.size//len(self.x):].reshape(self.y.shape)

    def next_batch(self):

        if self.times == 0:
            np.random.shuffle(self.combined)

        self.times += 1
        self.times %= (np.shape(self.y)[0]//self.batch_size)

        return self.x1[self.batch_size*self.times:self.batch_size*(self.times+1)], self.y1[self.batch_size*self.times:self.batch_size*(self.times+1)]

def get_test():

    x = np.load(src + 'test_x.npy')
    y = np.load(src + 'test_y.npy')

    return x, y

def test():
    ba = Batcher(batch_size = 100)
    for i in range (0, 1000):
        a, b = ba.next_batch()
        print(np.shape(a), np.shape(b))

