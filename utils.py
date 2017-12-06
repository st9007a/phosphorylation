#!/usr/bin/python3
import numpy as np

def next_batch(x, y, batch_size, times):
    global z, x1, y1
    if times == 1:
        combined = np.c_[x.reshape(len(x), -1), y.reshape(len(y), -1)]
        x1 = combined[:, :x.size//len(x)].reshape(x.shape)
        y1 = combined[:, x.size//len(x):].reshape(y.shape)

    curr = times % (y.size//batch_size)
    if curr == 0:
        curr = y.size//batch_size

    if curr == 1:
        np.random.shuffle(combined)

    return x1[batch_size*(curr-1):batch_size*curr], y1[batch_size*(curr-1):batch_size*curr]

