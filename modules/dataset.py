#!/usr/bin/python3
import tensorflow as tf

def _encode_onehot21(x):
    x = tf.one_hot(x, depth = 21)
    x = tf.reshape(x, [33, 21])
    x = tf.unstack(x, axis = 0)
    x = [tf.where(tf.equal(tf.reduce_sum(t), 0), tf.constant([0.05] * 20 + [0], tf.float32), t) for t in x]
    x = tf.stack(x, axis = 0)
    # x = tf.map_fn(
    #     lambda t: tf.where(tf.equal(tf.reduce_sum(t), 0), tf.constant([0.05] * 20 + [0], tf.float32), t), \
    #     elems = x, \
    #     dtype = tf.float32, \
    #     parallel_iterations = 10 \
    # )

    return x

def _encode_onehot22(x):
    x = tf.one_hot(x, depth = 22)
    x = tf.cast(x, tf.float32)

    return x

def _encode_onehot21_nox(x):
    x = tf.one_hot(x, depth = 21)
    x = tf.cast(x, tf.float32)

    return x

def _encode_onehot21_nopad(x):
    return x

def _encode_identity(x):
    return tf.cast(x, tf.float32)

encode_func_map = {
    'identity': _encode_identity,
    'onehot21': _encode_onehot21,
    'onehot22': _encode_onehot22,
    'onehot21_nox': _encode_onehot21_nox,
    'onehot21_nopad': _encode_onehot21_nopad,
}

if __name__ == '__main__':

    constant = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, \
                11, 12, 13, 14, 15, 16, 17, 18, 19, 20, \
                21,  1,  2,  3,  4,  5,  6,  7,  8,  9, \
                10, 11, 13]

    with tf.device('/cpu:0'):
        test = tf.constant(constant, tf.uint8)
        test = _encode_onehot21(test)
    sess = tf.Session()
    print(sess.run(test))


class Dataset():

    def __init__(self, encode, trainfiles, testfiles, batch_size, parallel_call = 1):

        self.trainfiles = trainfiles
        self.testfiles = testfiles
        self.encode_func = encode_func_map[encode]

        self.dataset = self.data_pipeline(batch_size, parallel_call)
        self.iterator = self.dataset.make_initializable_iterator()

    def _parse_tfrecord(self, example_proto):
        features = {
            'x': tf.FixedLenFeature([], tf.string, default_value = ''),
            'y': tf.FixedLenFeature([], tf.string, default_value = ''),
        }

        parsed_features = tf.parse_single_example(example_proto, features, name = 'parse_feature')

        x = tf.decode_raw(parsed_features['x'], tf.uint8)
        y = tf.decode_raw(parsed_features['y'], tf.uint8)
        y = tf.one_hot(y, depth = 2)
        y = tf.cast(y, tf.float32)
        return self.encode_func(x), y

    def data_pipeline(self, batch_size, parallel_call):

        with tf.device('/cpu:0'):

            self.files = tf.placeholder(tf.string, shape = [None])

            dataset = tf.data.TFRecordDataset(self.files)
            dataset = dataset.map(self._parse_tfrecord, num_parallel_calls = parallel_call)
            dataset = dataset.shuffle(buffer_size = 10000)
            dataset = dataset.batch(batch_size)

            return dataset
