#!/usr/bin/python3
import sys

src = sys.argv[1]
dst = sys.argv[2]

encode_list = list('ARNDCEQGHILKMFPSTWYV*X')

with open(src, 'r') as f:
    raw = f.readlines()

with open(dst, 'w+') as f:
    f.write('')

with open(dst, 'a') as f:
    for line in raw:
        line = line.rstrip('\n').split(' ')
        data = line[0]
        if data == '-1':
            data = '0'

        for c in line[2]:
            vec = [0] * 22
            vec[encode_list.index(c)] = 1

            data += ',' + ','.join(str(x) for x in vec)

        f.write(data + '\n')
