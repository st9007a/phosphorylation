#!/usr/bin/python3
import numpy as np

class Batcher():

    file_x = 'data/train_data_X_conv.npy'
    file_y = 'data/train_data_Y_conv.npy'

    def __init__(self, batch_size):
        self.times = 0

        self.x = np.load(Batcher.file_x)
        self.y = np.load(Batcher.file_y)
        self.batch_size = batch_size

        self.combined = np.c_[self.x.reshape(len(self.x), -1), self.y.reshape(len(self.y), -1)]
        self.x1 = self.combined[:, :self.x.size//len(self.x)].reshape(self.x.shape)
        self.y1 = self.combined[:, self.x.size//len(self.x):].reshape(self.y.shape)

    def next_batch(self):

        times = self.times
        x = self.x
        y = self.y
        batch_size = self.batch_size

        curr = (times + 1) % (np.shape(y)[0]//batch_size)
        if curr == 0:
            curr = np.shape(y)[0]//batch_size

        if curr == 1:
            np.random.shuffle(self.combined)

        #print("y.size = ", np.shape(y)[0]//batch_size, "times = ", times, "curr = ", curr)
        self.times += 1

        #print("batch ", batch_size*(curr-1), " to ", batch_size*curr)
        return self.x1[batch_size*(curr-1):batch_size*curr], self.y1[batch_size*(curr-1):batch_size*curr]

def get_validation():
    x = np.load('data/validation_data_X_conv.npy')
    y = np.load('data/validation_data_Y_conv.npy')

    return x, y

def test():
    ba = Batcher(batch_size = 100)
    for i in range (0, 1000):
        a, b = ba.next_batch()
        print(np.shape(a), np.shape(b))
