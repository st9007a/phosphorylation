#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from modules.tf_builtin import *

class MusiteDeepModel():

    def __init__(self, dataset, logdir = None):

        self.dataset = dataset
        self.logdir = logdir

        self.sess = None
        self.tb = None

        self.build_model()
        self.session_init()

    def build_model(self):
        data_pipeline = self.dataset.iterator.get_next()

        x = data_pipeline[0]
        y = tf.reshape(data_pipeline[1], [-1, 2])

        x_reshape = tf.reshape(x, [-1, 33, 22, 1])

        h_conv1, self.dropout1 = conv_layer(x_reshape, num_fms = 200, filter_size = [1, 22], strides = [1, 1, 22, 1], dropout = True)
        h_conv2, self.dropout2 = conv_layer(h_conv1, num_fms = 150, filter_size = [9, 1], strides = [1, 1, 1, 1], dropout = True)
        h_conv3, _ = conv_layer(h_conv2, num_fms = 200, filter_size = [10, 1], strides = [1, 1, 1, 1])

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

        self.dropout3 = tf.placeholder(tf.float32)
        h_merge1 = tf.nn.dropout(h_merge1, keep_prob = self.dropout3)

        w_merge2 = weight_var([149, 8])
        b_merge2 = bias_var([8])
        h_merge2 = h_merge1 @ w_merge2 + b_merge2
        h_merge2 = batch_normalization(h_merge2)
        h_merge2 = tf.nn.relu(h_merge2)

        w_output = weight_var([8, 2])
        b_output = bias_var([2])
        raw_output = h_merge2 @ w_output + b_output

        loss = tf.nn.softmax_cross_entropy_with_logits(logits = raw_output, labels = y) \
            + 2 * tf.nn.l2_loss(seq_att_w0) \
            + 2 * tf.nn.l2_loss(seq_att_w1) \
            + 0.151948 * tf.nn.l2_loss(fm_att_w0) \
            + 0.151948 * tf.nn.l2_loss(fm_att_w1)

        self.predict = tf.nn.softmax(raw_output)

        self.loss = tf.reduce_mean(loss)

        self.accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(raw_output, 1), tf.argmax(y, 1)), \
                tf.float32 \
            )
        )

        self.auc = tf.metrics.auc(labels = y[:,1], predictions = self.predict[:,1])

        self.train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)

        if self.logdir != None:
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.scalar('AUC', self.auc[0])

            self.tb = tf.summary.merge_all()

    def session_init(self):

        self.sess = tf.Session()

        if self.tb != None:
            self.train_writer = tf.summary.FileWriter(self.logdir + '/train', self.session.graph)
            self.test_writer = tf.summary.FileWriter(self.logdir + '/test')

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.sess.run(init_op)

    def train(self, epochs = 800):

        for i in range(1, epochs + 1):

            self.sess.run(self.dataset.iterator.initializer, feed_dict = {
                self.dataset.files: self.dataset.trainfiles
            })

            while True:

                try:
                    _, acc = self.sess.run([self.train_op, self.accuracy], feed_dict = {
                        self.dataset.files: self.dataset.trainfiles,
                        self.dropout1: 0.25, \
                        self.dropout2: 0.25, \
                        self.dropout3: 1 - 0.298224
                    })

                except tf.errors.OutOfRangeError:
                    print(i)
                    break

    def close(self):
        self.sess.close()
