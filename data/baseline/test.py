#!/usr/bin/python3
import tensorflow as tf

def read_csv(file_queue):
    reader = tf.TextLineReader()

    key, record = reader.read(file_queue)
    data = tf.decode_csv(record, record_defaults = [[0]] * (21 * 33 + 1))

    return data[0], data[1:]

def input_pipeline(files, batch_size, epochs):
    file_queue = tf.train.string_input_producer(files, num_epochs = None, shuffle = True)

    label, feature = read_csv(file_queue)

    label_batch, feature_batch = tf.train.shuffle_batch( \
        [label, feature],                                \
        batch_size = batch_size,                         \
        capacity = 1000 + 10 * batch_size,               \
        min_after_dequeue = 1000                         \
    )

    return label_batch, feature_batch

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord = coord, sess = sess)

y, x = input_pipeline(['train_onehot21.csv'], batch_size = 20, epochs = 1)
a = tf.reshape(x, [20, 21, 33, 1])

for i in range(100):
    sess.run(a)
    print(i)

coord.request_stop()
coord.join(threads)

sess.close()
