#!/usr/bin/python3
import numpy as np

y_tr = np.load('data/train_data_Y.npy')
y_va = np.load('data/validation_data_Y.npy')
y_te = np.load('data/test_data_Y.npy')

y_tr_convert = []
for i in y_tr:
    if i == 1:
        y_tr_convert.append([1, 0])
    elif i == 0:
        y_tr_convert.append([0, 1])
    else:
        print("unexpected value: " + str(i))
        exit()

y_va_convert = []
for i in y_va:
    if i == 1:
        y_va_convert.append([1, 0])
    elif i == 0:
        y_va_convert.append([0, 1])
    else:
        print("unexpected value: " + str(i))
        exit()

y_te_convert = []
for i in y_te:
    if i == 1:
        y_te_convert.append([1, 0])
    elif i == 0:
        y_te_convert.append([0, 1])
    else:
        print("unexpected value: " + str(i))
        exit()

np.save('data/train_data_Y_conv', y_tr_convert)
np.save('data/validation_data_Y_conv', y_va_convert)
np.save('data/test_data_Y_conv', y_te_convert)

x_tr = np.load('data/train_data_X.npy')
x_va = np.load('data/validation_data_X.npy')
x_te = np.load('data/test_data_X.npy')

x_tr_conv = []

for i in x_tr:
    data = []
    for j in i:
        onehot = [0] * 21
        if j != 21:
            onehot[j] = 1
        data.append(onehot)
    x_tr_conv.append(data)


x_va_conv = []

for i in x_va:
    data = []
    for j in i:
        onehot = [0] * 21
        if j != 21:
            onehot[j] = 1
        data.append(onehot)
    x_va_conv.append(data)

x_te_conv = []

for i in x_te:
    data = []
    for j in i:
        onehot = [0] * 21
        if j != 21:
            onehot[j] = 1
        data.append(onehot)
    x_te_conv.append(data)

np.save('data/train_data_X_conv', x_tr_conv)
np.save('data/validation_data_X_conv', x_va_conv)
np.save('data/test_data_X_conv', x_te_conv)
