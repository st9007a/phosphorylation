#!/usr/bin/python3
import numpy as np

x_tr = np.load('data/train_data_X.npy')
y_tr = np.load('data/train_data_Y.npy')

x_va = np.load('data/validation_data_X.npy')
y_va = np.load('data/validation_data_Y.npy')

x_te = np.load('data/test_data_X.npy')
y_te = np.load('data/test_data_Y.npy')

print(x_tr.shape)
print(y_tr.shape)

print(x_va.shape)
print(y_va.shape)

print(x_te.shape)
print(y_te.shape)

