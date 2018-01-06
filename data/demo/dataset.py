#!/usr/bin/python3
import datetime
import tensorflow as tf
import sys

num_parallel_calls = 1

if len(sys.argv) > 1:
    num_parallel_calls = int(sys.argv[1])

def _parse_tfrecord(example_proto):
    features = {
        'x': tf.FixedLenFeature([], tf.string, default_value = ''),
        'y': tf.FixedLenFeature([], tf.string, default_value = ''),
    }

    parsed_features = tf.parse_single_example(example_proto, features, name = 'parse_feature')

    x = tf.decode_raw(parsed_features['x'], tf.uint8)
    x = tf.one_hot(x, depth = 22)
    x = tf.cast(x, tf.float32)

    y = tf.decode_raw(parsed_features['y'], tf.uint8)
    y = tf.one_hot(y, depth = 2)
    y = tf.cast(y, tf.float32)
    return x, y

with tf.device('/cpu:0'):
    dataset = tf.data.TFRecordDataset(['train.tfrecord'])
    # dataset = dataset.prefetch(1000)
    dataset = dataset.map(_parse_tfrecord, num_parallel_calls = num_parallel_calls)
    dataset = dataset.repeat(10)
    dataset = dataset.shuffle(buffer_size = 10000)
    dataset = dataset.batch(200)

    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

sess = tf.Session()
start = datetime.datetime.now()

while True:
    try:
        sess.run(next_batch)
    except tf.errors.OutOfRangeError:
        break

end = datetime.datetime.now()
print(end - start)
