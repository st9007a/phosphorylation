#!/usr/bin/python3
import datetime
import tensorflow as tf
import sys

num_threads = 1

if len(sys.argv) > 1:
    num_threads = int(sys.argv[1])

def _parse_tfrecord(example_proto):
    features = {
        'x': tf.FixedLenFeature([], tf.string, default_value = ''),
        'y': tf.FixedLenFeature([], tf.string, default_value = ''),
    }

    parsed_features = tf.parse_single_example(example_proto, features, name = 'parse_feature')

    x = tf.decode_raw(parsed_features['x'], tf.uint8)
    x = tf.one_hot(x, depth = 22)
    x = tf.reshape(x, [33, 22])
    x = tf.cast(x, tf.float32)

    y = tf.decode_raw(parsed_features['y'], tf.uint8)
    y = tf.one_hot(y, depth = 2)
    y = tf.reshape(y, [2])
    y = tf.cast(y, tf.float32)
    return y, x

def read_tfrecord(file_queue):
    reader = tf.TFRecordReader()
    key, record = reader.read(file_queue)

    return _parse_tfrecord(record)

def input_pipeline(files, batch_size, epochs):
    file_queue = tf.train.string_input_producer(files, num_epochs = epochs, shuffle = True)

    label, feature = read_tfrecord(file_queue)

    label_batch, feature_batch = tf.train.shuffle_batch( \
        [label, feature],                                \
        batch_size = batch_size,                         \
        num_threads = num_threads,                       \
        capacity = 10000 + 5 * batch_size,               \
        min_after_dequeue = 10000                        \
    )

    return label_batch, feature_batch

with tf.device('/cpu:0'):
    y, x = input_pipeline(['train.tfrecord'], batch_size = 200, epochs = 10)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess = tf.Session()
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord = coord, sess = sess)

start = datetime.datetime.now()
try:
    while not coord.should_stop():
        sess.run(y)

except tf.errors.OutOfRangeError as e:
    coord.request_stop(e)

finally:
    coord.request_stop()
end = datetime.datetime.now()
print(end - start)

coord.join(threads)
sess.close()
