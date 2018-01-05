#!/usr/bin/python3
import numpy as np
import tensorflow as tf

src = 'data/mlb/'

class Batcher():

    file_x = src + 'train_x.npy'
    file_y = src + 'train_y.npy'

    def __init__(self, batch_size):
        self.times = 0

        self.x = np.load(Batcher.file_x)
        self.y = np.load(Batcher.file_y)
        self.batch_size = batch_size

        self.combined = np.c_[self.x.reshape(len(self.x), -1), self.y.reshape(len(self.y), -1)]
        self.x1 = self.combined[:, :self.x.size // len(self.x)].reshape(self.x.shape)
        self.y1 = self.combined[:, self.x.size // len(self.x):].reshape(self.y.shape)

    def next_batch(self):

        if self.times == 0:
            np.random.shuffle(self.combined)

        self.times += 1
        self.times %= (np.shape(self.y)[0] // self.batch_size)

        return self.x1[self.batch_size * self.times : self.batch_size * (self.times + 1)], self.y1[self.batch_size * self.times : self.batch_size * (self.times + 1)]

def get_test():

    x = np.load(src + 'test_x.npy')
    y = np.load(src + 'test_y.npy')

    return x, y

def test():
    ba = Batcher(batch_size = 100)
    for i in range (0, 1000):
        a, b = ba.next_batch()
        print(np.shape(a), np.shape(b))

def input_pipeline(files, batch_size, epochs, record_defaults):
    file_queue = tf.train.string_input_producer(files, num_epochs = epochs, shuffle = True)
    key, record = tf.TextLineReader().read(file_queue)
    data = tf.decode_csv(record, record_defaults = record_defaults)

    y_batch, x_batch = tf.train.shuffle_batch( \
        [data[0], data[1:]],                   \
        batch_size = batch_size,               \
        capacity = 1000 + 10 * batch_size,     \
        min_after_dequeue = 1000               \
    )

    return y_batch, x_batch
