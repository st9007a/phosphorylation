#!/usr/bin/python3
import numpy as np
import sys

src = sys.argv[1]
dst = sys.argv[2]

encode_list = list('ARNDCEQGHILKMFPSTWYV*')

with open(src, 'r') as f:
    raw = f.readlines()

x = []
y = []

for line in raw:
    line = line.rstrip('\n').split(' ')

    if line[0] == '-1':
        y.append([0, 1])
    elif line[0] == '1':
        y.append([1, 0])

    data = []
    for c in line[2]:
        vec = [0] * 21

        if c != 'X':
            vec[encode_list.index(c)] = 1

        data.append(vec)
    x.append(data)

np.save(dst + '_x', np.array(x))
np.save(dst + '_y', np.array(y))

