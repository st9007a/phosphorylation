#!/usr/bin/python3
import numpy as np

src = 'raw_data/ML'
x_tr = np.load(src + '/train_data_X_conv.npy')
y_tr = np.load(src + '/train_data_Y_conv.npy')
x_te = np.load(src + '/test_data_X_conv.npy')
y_te = np.load(src + '/test_data_Y_conv.npy')
y = np.load('testY_new.npy')

print(x_tr.shape)
print(y_tr.shape)
print(x_te.shape)
print(y_te.shape)
print(y.shape)
