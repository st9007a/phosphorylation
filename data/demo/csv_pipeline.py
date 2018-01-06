#!/usr/bin/python3
import tensorflow as tf
import sys

num_threads = 1

if len(sys.argv) > 1:
    num_threads = int(sys.argv[1])

def read_csv(file_queue):
    reader = tf.TextLineReader()

    key, record = reader.read(file_queue)
    data = tf.decode_csv(record, record_defaults = [[0.0]] * (21 * 33 + 1))
    # data = tf.decode_csv(record, record_defaults = [[1]] * 2)

    return data[0], data[1:]

def input_pipeline(files, batch_size, epochs):
    file_queue = tf.train.string_input_producer(files, num_epochs = epochs, shuffle = True)

    label, feature = read_csv(file_queue)

    label_batch, feature_batch = tf.train.shuffle_batch( \
        [label, feature],                                \
        batch_size = batch_size,                         \
        num_threads = num_threads,                       \
        capacity = 10000 + 5 * batch_size,               \
        min_after_dequeue = 10000                        \
    )

    return label_batch, feature_batch

with tf.device('/cpu:0'):
    y, x = input_pipeline(['train_onehot21.csv'], batch_size = 200, epochs = 5)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

sess = tf.Session()
sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord = coord, sess = sess)
i = 0

try:
    while not coord.should_stop():
        ty = sess.run(y)
        print(str(i))
        i += 1

except tf.errors.OutOfRangeError as e:
    print('Done')
    coord.request_stop(e)

finally:
    coord.request_stop()

coord.join(threads)
sess.close()
