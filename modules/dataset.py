#!/usr/bin/python3
import tensorflow as tf

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

class Dataset():

    def __init__(self, trainfiles, testfiles, batch_size, parallel_call = 1):

        self.trainfiles = trainfiles
        self.testfiles = testfiles

        self.dataset = self.data_pipeline(batch_size, parallel_call)
        self.iterator = self.dataset.make_initializable_iterator()


    def data_pipeline(self, batch_size, parallel_call):

        with tf.device('/cpu:0'):

            self.files = tf.placeholder(tf.string, shape = [None])

            dataset = tf.data.TFRecordDataset(self.files)
            dataset = dataset.map(_parse_tfrecord, num_parallel_calls = parallel_call)
            dataset = dataset.shuffle(buffer_size = 10000)
            dataset = dataset.batch(batch_size)

            return dataset
