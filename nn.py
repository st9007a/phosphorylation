#!/usr/bin/python3
import tensorflow as tf
from utils import Batcher, get_validation

def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_var(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

def attention(tensor, dim):
    input_dim = int(tensor.get_shape()[2])
    input_length = int(tensor.get_shape()[1])

    tensor_unpack = tf.unstack(tensor, axis = 1)

    w0 = weight_var([input_dim, dim])
    b0 = bias_var([dim])
    energy = [tf.matmul(t, w0) + b0 for t in tensor_unpack]

    w1 = weight_var([dim, 1])
    b1 = bias_var([1])
    energy = [tf.matmul(t, w1) + b1 for t in energy]

    energy = tf.stack(energy, axis = 1)
    energy = tf.reshape(energy, [-1, input_length])
    energy = tf.nn.softmax(energy)

    weighted_sum = tf.reduce_sum(tensor * tf.expand_dims(energy, -1), 1)
    return weighted_sum


def batch_normalization(entry_tensor):
    shape = entry_tensor.get_shape()
    trainable_var_shape = shape[1:]
    axis = [i for i in range(len(shape))]

    mean, variance = tf.nn.moments(entry_tensor, axis)
    offset = tf.Variable(tf.zeros(trainable_var_shape))
    scale = tf.Variable(tf.ones(trainable_var_shape))

    bn = tf.nn.batch_normalization(entry_tensor, \
                                   mean = mean, \
                                   variance = variance, \
                                   offset = offset, \
                                   scale = scale, \
                                   variance_epsilon = 1e-3)
    return bn

class Classifier():

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape = [None, 33, 21])
        self.y = tf.placeholder(tf.float32, shape = [None, 2])

        self.build()

    def build(self):
        x_reshape = tf.reshape(self.x, [-1, 33, 21, 1])

        w_conv1 = weight_var([1, 21, 1, 200])
        b_conv1 = bias_var([200])
        h_conv1 = tf.nn.conv2d(x_reshape, w_conv1, strides = [1, 1, 1, 1], padding = 'VALID')
        h_conv1 = batch_normalization(h_conv1)
        # h_conv1 = tf.nn.dropout(h_conv1, keep_prob = 0.75)
        h_conv1 = tf.nn.relu(h_conv1) + b_conv1

        w_conv2 = weight_var([9, 1, 200, 150])
        b_conv2 = bias_var([150])
        h_conv2 = tf.nn.conv2d(h_conv1, w_conv2, strides = [1, 1, 1, 1], padding = 'VALID')
        h_conv2 = batch_normalization(h_conv2)
        # h_conv2 = tf.nn.dropout(h_conv2, keep_prob = 0.75)
        h_conv2 = tf.nn.relu(h_conv2) + b_conv2

        w_conv3 = weight_var([10, 1, 150, 200])
        b_conv3 = bias_var([200])
        h_conv3 = tf.nn.conv2d(h_conv2, w_conv3, strides = [1, 1, 1, 1], padding = 'VALID')
        h_conv3 = batch_normalization(h_conv3)
        h_conv3 = tf.nn.relu(h_conv3) + b_conv3

        seq_domain_entry = tf.reshape(h_conv3, [-1, 16, 200])
        fm_domain_entry = tf.transpose(seq_domain_entry, perm = [0, 2, 1])

        merged = tf.concat([attention(seq_domain_entry, dim = 8), attention(fm_domain_entry, dim = 10)], axis = 1)

        w_merge1 = weight_var([200 + 16, 149])
        b_merge1 = bias_var([149])
        h_merge1 = tf.matmul(merged, w_merge1) + b_merge1
        h_merge1 = batch_normalization(h_merge1)
        h_merge1 = tf.nn.relu(h_merge1)
        h_merge1 = tf.nn.dropout(h_merge1, keep_prob = 0.298224)

        w_merge2 = weight_var([149, 8])
        b_merge2 = bias_var([8])
        h_merge2 = tf.matmul(h_merge1, w_merge2) + b_merge2
        h_merge2 = batch_normalization(h_merge2)
        h_merge2 = tf.nn.relu(h_merge2)

        w_output = weight_var([8, 2])
        b_output = bias_var([2])
        raw_output = tf.nn.relu(tf.matmul(h_merge2, w_output) + b_output)


        self.predict = tf.nn.softmax(raw_output)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = raw_output, labels = self.y)
        self.train_step = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(raw_output, 1), tf.argmax(self.y, 1)), \
                tf.float32 \
            )
        )

        self.auc = tf.metrics.auc(labels = self.y, predictions = tf.nn.softmax(raw_output))

    def train(self, batch_size, steps):

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5

        self.session = tf.Session(config = config)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init)

        batcher = Batcher(batch_size)
        x_eval, y_eval = get_validation()

        for i in range(steps):
            x_batch, y_batch = batcher.next_batch()
            self.session.run(self.train_step, feed_dict = {self.x: x_batch, self.y: y_batch})

            if i % 100 == 0:
                train_accuracy = self.session.run(self.accuracy, feed_dict = {self.x: x_batch, self.y: y_batch})
                test_accuracy = self.session.run(self.accuracy, feed_dict = {self.x: x_eval, self.y: y_eval})
                auc, _ = self.session.run(self.auc, feed_dict = {self.x: x_eval, self.y: y_eval})
                print("train acc: " + str(train_accuracy) + ", test acc: " + str(test_accuracy) + ", test auc: " + str(auc) + ", step: " + str(i))

        self.session.close()
