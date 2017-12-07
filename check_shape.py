#!/usr/bin/python3
import numpy as np

x_tr = np.load('data/train_data_X.npy')
x_tr_c = np.load('data/train_data_X_conv.npy')
y_tr = np.load('data/train_data_Y.npy')
y_tr_c = np.load('data/train_data_Y_conv.npy')

x_va = np.load('data/validation_data_X.npy')
x_va_c = np.load('data/validation_data_X_conv.npy')
y_va = np.load('data/validation_data_Y.npy')
y_va_c = np.load('data/validation_data_Y_conv.npy')

x_te = np.load('data/test_data_X.npy')
x_te_c = np.load('data/test_data_X_conv.npy')
y_te = np.load('data/test_data_Y.npy')
y_te_c = np.load('data/test_data_Y_conv.npy')

print(x_tr.shape)
print(x_tr_c.shape)
print(y_tr.shape)
print(y_tr_c.shape)

print(x_va.shape)
print(x_va_c.shape)
print(y_va.shape)
print(y_va_c.shape)

print(x_te.shape)
print(x_te_c.shape)
print(y_te.shape)
print(y_te_c.shape)

