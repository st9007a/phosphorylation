#!/usr/bin/python3
import numpy as np

src = 'raw_data/T'
y_tr = np.load(src + '/train_y.npy')
# y_va = np.load(src + '/validation_data_Y.npy')
y_te = np.load(src + '/test_y.npy')

y_tr_convert = []
for i in y_tr:
    if i == 1:
        y_tr_convert.append([1, 0])
    elif i == 0:
        y_tr_convert.append([0, 1])
    else:
        print("unexpected value: " + str(i))
        exit()

# y_va_convert = []
# for i in y_va:
#     if i == 1:
#         y_va_convert.append([1, 0])
#     elif i == 0:
#         y_va_convert.append([0, 1])
#     else:
#         print("unexpected value: " + str(i))
#         exit()

y_te_convert = []
for i in y_te:
    if i == 1:
        y_te_convert.append([1, 0])
    elif i == 0:
        y_te_convert.append([0, 1])
    else:
        print("unexpected value: " + str(i))
        exit()

np.save(src + '/train_data_Y_conv', y_tr_convert)
# np.save(src + '/validation_data_Y_conv', y_va_convert)
np.save(src + '/test_data_Y_conv', y_te_convert)
del y_tr_convert
del y_te_convert
del y_tr
del y_te


x_tr = np.load(src + '/train_x.npy')
# x_va = np.load(src + '/validation_data_X.npy')
x_te = np.load(src + '/test_x.npy')

x_tr_conv = []

# for i in x_tr:
#     pair = []
#     for j in range(0, 4):
#         data = []
#         for k in i[j]:
#             onehot = [0] * 26
#             onehot[k] = 1
#             data.append(onehot)
#         pair.append(data)
#     x_tr_conv.append(pair)

for i in x_tr:
    data = []
    for j in i:
        print(j)
        onehot = [0] * 21
        onehot[j] = 1
        data.append(onehot)
    x_tr_conv.append(data)


x_te_conv = []

# for i in x_te:
#     pair = []
#     for j in range(0, 4):
#         data = []
#         for k in i[j]:
#             onehot = [0] * 26
#             onehot[k] = 1
#             data.append(onehot)
#         pair.append(data)
#     x_te_conv.append(pair)

for i in x_te:
    data = []
    for j in i:
        onehot = [0] * 21
        onehot[j] = 1
        data.append(onehot)
    x_te_conv.append(data)

np.save(src + '/train_data_X_conv', x_tr_conv)
# np.save(src + '/validation_data_X_conv', x_va_conv)
np.save(src + '/test_data_X_conv', x_te_conv)
