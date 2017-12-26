#!/usr/bin/python3
import tensorflow as tf
from utils import Batcher, get_test

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

class MusiteDeepClassifier():

    def __init__(self):
        self.dist = 'log/PELMY-PPA3Y/'
        self.dir = self.dist + 'pssm-1-4/'
        self.x = tf.placeholder(tf.float32, shape = [None, 4, 33, 25])
        self.y = tf.placeholder(tf.float32, shape = [None, 2])

        self.build()

    def conv_layer(self, input_tensor, num_fms, filter_size, strides, dropout = None, act = tf.nn.relu):
        curr = int(input_tensor.get_shape()[3])
        w = weight_var(filter_size + [curr, num_fms])
        b = bias_var([num_fms])
        h = tf.nn.conv2d(input_tensor, w, strides = strides, padding = 'SAME') + b
        h = batch_normalization(h)

        dropout_layer = None
        if dropout is True:
            dropout_layer = tf.placeholder(tf.float32)
            h = tf.nn.dropout(h, keep_prob = dropout_layer)
        h = act(h)

        return h, dropout_layer

    def build(self):
        x_transpose = tf.transpose(self.x, perm = [0, 2, 3, 1])

        h_conv1, self.dropout1 = self.conv_layer(x_transpose, num_fms = 200, filter_size = [1, 25], strides = [1, 1, 25, 1], dropout = True)
        h_conv2, self.dropout2 = self.conv_layer(h_conv1, num_fms = 150, filter_size = [9, 1], strides = [1, 1, 1, 1], dropout = True)
        h_conv3, _ = self.conv_layer(h_conv2, num_fms = 200, filter_size = [10, 1], strides = [1, 1, 1, 1])

        h_conv3 = h_conv3 + h_conv1

        seq_domain_entry = tf.reshape(h_conv3, [-1, 33, 200])
        fm_domain_entry = tf.transpose(seq_domain_entry, perm = [0, 2, 1])

        seq_att, seq_att_w0, seq_att_w1 = attention(seq_domain_entry, dim = 8)
        fm_att, fm_att_w0, fm_att_w1 = attention(fm_domain_entry, dim = 10)
        merged = tf.concat([seq_att, fm_att], axis = 1)

        w_merge1 = weight_var([200 + 33, 149])
        b_merge1 = bias_var([149])
        h_merge1 = merged @ w_merge1 + b_merge1
        h_merge1 = batch_normalization(h_merge1)
        h_merge1 = tf.nn.relu(h_merge1)

        self.dropout4 = tf.placeholder(tf.float32)
        h_merge1 = tf.nn.dropout(h_merge1, keep_prob = self.dropout4)

        w_merge2 = weight_var([149, 8])
        b_merge2 = bias_var([8])
        h_merge2 = h_merge1 @ w_merge2 + b_merge2
        h_merge2 = batch_normalization(h_merge2)
        h_merge2 = tf.nn.relu(h_merge2)

        w_output = weight_var([8, 2])
        b_output = bias_var([2])
        raw_output = h_merge2 @ w_output + b_output


        self.predict = tf.nn.softmax(raw_output)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = raw_output, labels = self.y) \
            + 2 * tf.nn.l2_loss(seq_att_w0) \
            + 2 * tf.nn.l2_loss(seq_att_w1) \
            + 0.151948 * tf.nn.l2_loss(fm_att_w0) \
            + 0.151948 * tf.nn.l2_loss(fm_att_w1)
        self.mean_loss = tf.reduce_mean(self.loss)

        self.train_step = tf.train.AdamOptimizer(5e-4).minimize(self.loss)
        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(raw_output, 1), tf.argmax(self.y, 1)), \
                tf.float32 \
            )
        )

        self.auc = tf.metrics.auc(labels = self.y, predictions = tf.nn.softmax(raw_output))

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('AUC', self.auc[0])

        self.tb = tf.summary.merge_all()


    def train(self, batch_size, steps):


        # config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.8

        # self.session = tf.Session(config = config)
        self.session = tf.Session()
        train_writer = tf.summary.FileWriter(self.dir + '/train', self.session.graph)
        test_writer = tf.summary.FileWriter(self.dir + '/test')
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.session.run(init)


        batcher = Batcher(batch_size)
        x_test, y_test = get_test()

        def get_feed(opt = 'train'):
            if opt == 'train':
                x, y = batcher.next_batch()
                return {self.x: x, \
                        self.y: y, \
                        self.dropout1: 0.25, \
                        self.dropout2: 0.25, \
                        self.dropout3: 1 - 0.298224}
            elif opt == 'test':
                return {self.x: x_test, \
                        self.y: y_test, \
                        self.dropout1: 1, \
                        self.dropout2: 1, \
                        self.dropout3: 1}

        for i in range(1, steps + 1):
            x_batch, y_batch = batcher.next_batch()
            self.session.run(self.train_step, feed_dict = get_feed('train'))

            if i % 100 == 0:
                tb_train, train_accuracy = self.session.run([self.tb, self.accuracy], feed_dict = get_feed('train'))
                tb_test, test_accuracy = self.session.run([self.tb, self.accuracy], feed_dict = get_feed('test'))
                _, test_auc = self.session.run(self.auc, feed_dict = get_feed('test'))
                test_loss = self.session.run(self.mean_loss, feed_dict = get_feed('test'))
                print("train acc: " + str(train_accuracy) + ", test acc: " + str(test_accuracy) +  ", test auc: " + str(test_auc) + ", test loss: " + str(test_loss) + ", step: " + str(i))

                # train_writer.add_summary(tb_train, i)
                # test_writer.add_summary(tb_test, i)

        self.session.close()
