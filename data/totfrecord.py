#!/usr/bin/python3
import tensorflow as tf
import sys

src = sys.argv[1]
dst = sys.argv[2]
encode_list = list('ARNDCEQGHILKMFPSTWYV*X')

def int_to_byte(i):
    return bytes([i])

if __name__ == '__main__':
    with open(src, 'r') as text:
        raw_data = text.readlines()

    writer = tf.python_io.TFRecordWriter(dst)

    for line in raw_data:
        line = line.rstrip('\n').split(' ')

        y = int(line[0])
        if y == -1:
            y = 0
        y = int_to_byte(y)

        x = b''
        for c in line[2]:
            x = x + int_to_byte(encode_list.index(c))

        features = tf.train.Features(feature = {
            'x': tf.train.Feature(bytes_list = tf.train.BytesList(value = [x])),
            'y': tf.train.Feature(bytes_list = tf.train.BytesList(value = [y])),
        })

        example = tf.train.Example(features = features)
        writer.write(example.SerializeToString())

    writer.close()
