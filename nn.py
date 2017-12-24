#!/usr/bin/python3
import tensorflow as tf
from utils import Batcher, get_validation, get_test

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
    return weighted_sum, w0, w1


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

class CNN():

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape = [None, 10, 33])
        self.y = tf.placeholder(tf.float32, shape = [None, 2])

        self.build()

    def conv_layer(self, input_tensor, filter_size, channels, patch = 1):
        current_channels = int(input_tensor.get_shape()[3])

        w = weight_var(filter_size + [current_channels, channels])
        b = bias_var([channels])
        h = tf.nn.conv2d(input_tensor, w, strides = [1, patch, patch, 1], padding = 'SAME') + b
        h = batch_normalization(h)
        h = tf.nn.relu(h)

        return h

    def dense_layer(self, input_tensor, out_dim, act = tf.nn.relu):
        current_dim = int(input_tensor.get_shape()[1])

        w = weight_var([current_dim, out_dim])
        b = bias_var([out_dim])
        h = tf.matmul(input_tensor, w) + b
        h = batch_normalization(h)
        h = act(h)

        return h, w

    def build(self):
        self.dropout1 = tf.placeholder(tf.float32)
        self.dropout2 = tf.placeholder(tf.float32)
        x_reshape = tf.reshape(self.x, [-1, 10, 33, 1])

        h_conv1 = self.conv_layer(x_reshape, filter_size = [3, 3], channels = 64)
        h_conv2 = self.conv_layer(h_conv1, filter_size = [3, 3], channels = 64)

        h_reduce1 = self.conv_layer(h_conv2, filter_size = [2, 2], channels = 64, patch = 2)

        # size: 5 * 17
        h_conv3 = self.conv_layer(h_reduce1, filter_size = [3, 3], channels = 128)
        h_conv4 = self.conv_layer(h_conv3, filter_size = [3, 3], channels = 128)

        h_reduce2 = self.conv_layer(h_conv4, filter_size = [2, 2], channels = 128, patch = 2)

        # size: 3 * 9
        h_conv5 = self.conv_layer(h_reduce2, filter_size = [3, 3], channels = 256)
        h_conv6 = self.conv_layer(h_conv5, filter_size = [3, 3], channels = 256)
        h_conv7 = self.conv_layer(h_conv6, filter_size = [3, 3], channels = 256)

        h_reduce3 = self.conv_layer(h_conv7, filter_size = [2, 2], channels = 128, patch = 2)

        # size: 2 * 5
        h_conv8 = self.conv_layer(h_reduce3, filter_size = [3, 3], channels = 512)
        h_conv9 = self.conv_layer(h_conv8, filter_size = [3, 3], channels = 512)
        h_conv10 = self.conv_layer(h_conv9, filter_size = [3, 3], channels = 512)

        h_conv11 = self.conv_layer(h_conv10, filter_size = [3, 3], channels = 128)

        h_pool4 = tf.nn.avg_pool(h_conv11, ksize = [1, 2, 5, 1], strides = [1, 2, 5, 1], padding = 'SAME')

        flatten = tf.reshape(h_pool4, [-1, 128])

        h_fc, w_fc = self.dense_layer(flatten, out_dim = 1024)
        h_dropout = tf.nn.dropout(h_fc, keep_prob = self.dropout2)

        w_output = weight_var([1024, 2])
        b_output = bias_var([2])
        raw_output = tf.matmul(h_dropout, w_output) + b_output

        self.predict = tf.nn.softmax(raw_output)
        self.train_loss = tf.nn.softmax_cross_entropy_with_logits(logits = raw_output, labels = self.y)
        self.train_step = tf.train.AdamOptimizer(5e-4).minimize(self.train_loss)
        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(raw_output, 1), tf.argmax(self.y, 1)), \
                tf.float32 \
            )
        )

        self.auc = tf.metrics.auc(labels = self.y, predictions = tf.nn.softmax(raw_output))

    def train(self, batch_size, steps):

        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 1
        #
        # self.session = tf.Session(config = config)
        self.session = tf.Session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init)

        batcher = Batcher(batch_size)
        x_test, y_test = get_test()

        for i in range(1, steps + 1):
            x_batch, y_batch = batcher.next_batch()
            self.session.run(self.train_step, feed_dict = {self.x: x_batch, self.y: y_batch, self.dropout2: 0.8})

            if i % 100 == 0:
                train_accuracy = self.session.run(self.accuracy, feed_dict = {self.x: x_batch, self.y: y_batch, self.dropout2: 0.8})
                test_accuracy = self.session.run(self.accuracy, feed_dict = {self.x: x_test, self.y: y_test, self.dropout2: 1})
                test_auc, _ = self.session.run(self.auc, feed_dict = {self.x: x_test, self.y: y_test, self.dropout2: 1})
                print("train acc: " + str(train_accuracy) + ", test acc: " + str(test_accuracy) +  ", test auc: " + str(test_auc) + ", step: " + str(i))

        self.session.close()


class MusiteDeepClassifier():

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
        h_conv1 = tf.nn.dropout(h_conv1, keep_prob = 0.75)
        h_conv1 = tf.nn.relu(h_conv1) + b_conv1

        w_conv2 = weight_var([9, 1, 200, 150])
        b_conv2 = bias_var([150])
        h_conv2 = tf.nn.conv2d(h_conv1, w_conv2, strides = [1, 1, 1, 1], padding = 'VALID')
        h_conv2 = batch_normalization(h_conv2)
        h_conv2 = tf.nn.dropout(h_conv2, keep_prob = 0.75)
        h_conv2 = tf.nn.relu(h_conv2) + b_conv2

        w_conv3 = weight_var([10, 1, 150, 200])
        b_conv3 = bias_var([200])
        h_conv3 = tf.nn.conv2d(h_conv2, w_conv3, strides = [1, 1, 1, 1], padding = 'VALID')
        h_conv3 = batch_normalization(h_conv3)
        h_conv3 = tf.nn.relu(h_conv3) + b_conv3

        seq_domain_entry = tf.reshape(h_conv3, [-1, 16, 200])
        fm_domain_entry = tf.transpose(seq_domain_entry, perm = [0, 2, 1])

        seq_att, seq_att_w0, seq_att_w1 = attention(seq_domain_entry, dim = 8)
        fm_att, fm_att_w0, fm_att_w1 = attention(fm_domain_entry, dim = 10)
        merged = tf.concat([seq_att, fm_att], axis = 1)

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
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = raw_output, labels = self.y) \
            + 2 * tf.nn.l2_loss(seq_att_w0) \
            + 2 * tf.nn.l2_loss(seq_att_w1) \
            + 0.151948 * tf.nn.l2_loss(fm_att_w0) \
            + 0.151948 * tf.nn.l2_loss(fm_att_w1)

        self.train_step = tf.train.AdamOptimizer(5e-4).minimize(self.loss)
        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(raw_output, 1), tf.argmax(self.y, 1)), \
                tf.float32 \
            )
        )

        self.auc = tf.metrics.auc(labels = self.y, predictions = tf.nn.softmax(raw_output))

    def train(self, batch_size, steps):

        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8

        # self.session = tf.Session(config = config)
        self.session = tf.Session()
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init)

        batcher = Batcher(batch_size)
        # x_eval, y_eval = get_validation()
        x_test, y_test = get_test()

        for i in range(1, steps + 1):
            x_batch, y_batch = batcher.next_batch()
            self.session.run(self.train_step, feed_dict = {self.x: x_batch, self.y: y_batch})

            if i % 100 == 0:
                train_accuracy = self.session.run(self.accuracy, feed_dict = {self.x: x_batch, self.y: y_batch})
                # valid_accuracy = self.session.run(self.accuracy, feed_dict = {self.x: x_eval, self.y: y_eval})
                test_accuracy = self.session.run(self.accuracy, feed_dict = {self.x: x_test, self.y: y_test})
                # valid_auc = self.session.run(self.auc, feed_dict = {self.x: x_eval, self.y: y_eval})
                test_auc = self.session.run(self.auc, feed_dict = {self.x: x_test, self.y: y_test})
                # print("train acc: " + str(train_accuracy) + ", valid acc: " + str(valid_accuracy) + ", test acc: " + str(test_accuracy) + ", valid auc: " + str(valid_auc) + ", test auc: " + str(test_auc) + ", step: " + str(i))
                print("train acc: " + str(train_accuracy) + ", test acc: " + str(test_accuracy) +  ", test auc: " + str(test_auc) + ", step: " + str(i))

        self.session.close()
