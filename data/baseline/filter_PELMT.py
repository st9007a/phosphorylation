#!/usr/bin/python3
import random
import sys

src = sys.argv[1]
dst = sys.argv[2]

count = 0
sample = []

with open(src, 'r') as f:
    all_lines = f.readlines()

    for line in all_lines:
        if line.split(' ')[0] == '1':
            count += 1
            sample.append(line)

    random.shuffle(all_lines)

    for line in all_lines:
        if count > 0 and line.split(' ')[0] == '-1':
            count -= 1
            sample.append(line)

with open(dst, 'w+') as f:
    f.write('')

with open(dst, 'a') as f:
    for line in sample:
        f.write(line)
