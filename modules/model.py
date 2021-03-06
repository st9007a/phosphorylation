#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from modules.tf_builtin import *

class MusiteDeepModel():

    def __init__(self, dataset, logdir):

        self.dataset = dataset
        self.logdir = logdir

        self.sess = None
        self.tb = None

        # model option
        self.dropout_1 = tf.placeholder(tf.float32)
        self.dropout_2 = tf.placeholder(tf.float32)
        self.dropout_3 = tf.placeholder(tf.float32)
        self.training = tf.placeholder(tf.bool)

        self.build_model()
        self.session_init()

    def build_model(self):
        print('Start: build model')

        data_pipeline = self.dataset.iterator.get_next()

        x = data_pipeline[0]
        y = tf.reshape(data_pipeline[1], [-1, 2])

        x_reshape = tf.reshape(x, [-1, 1, 33, 22])

        h_conv1 = conv_layer(x_reshape, training = self.training, num_fms = 200, filter_size = [1, 22], strides = [1, 22], dropout = self.dropout_1)
        h_conv2 = conv_layer(h_conv1, training = self.training, num_fms = 150, filter_size = [9, 1], strides = [1, 1], dropout = self.dropout_2)
        h_conv3 = conv_layer(h_conv2, training = self.training, num_fms = 200, filter_size = [10, 1], strides = [1, 1])

        fm_domain_entry = tf.reshape(h_conv3, [-1, 200, 33])
        seq_domain_entry = tf.transpose(fm_domain_entry, perm = [0, 2, 1])

        seq_att, seq_att_w0, seq_att_w1 = attention(seq_domain_entry, dim = 8)
        fm_att, fm_att_w0, fm_att_w1 = attention(fm_domain_entry, dim = 10)
        merged = tf.concat([seq_att, fm_att], axis = 1)

        w_merge1 = weight_var([200 + 33, 149])
        b_merge1 = bias_var([149])
        h_merge1 = merged @ w_merge1 + b_merge1
        h_merge1 = fused_batch_normalization(h_merge1, self.training)
        h_merge1 = tf.nn.relu(h_merge1)

        h_merge1 = tf.nn.dropout(h_merge1, keep_prob = self.dropout_3)

        w_merge2 = weight_var([149, 8])
        b_merge2 = bias_var([8])
        h_merge2 = h_merge1 @ w_merge2 + b_merge2
        h_merge2 = fused_batch_normalization(h_merge2, self.training)
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

        self.loss, self.loss_update = tf.metrics.mean(loss)
        self.acc, self.acc_update = tf.metrics.accuracy(labels = tf.argmax(y, 1), predictions = tf.argmax(self.predict, 1))
        self.auc, self.auc_update = tf.metrics.auc(labels = y[:,1], predictions = self.predict[:,1])

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_op = tf.train.AdamOptimizer(5e-4).minimize(loss)

        tf.summary.scalar('Accuracy', self.acc)
        tf.summary.scalar('AUC', self.auc)
        tf.summary.scalar('Loss', self.loss)

        self.tb = tf.summary.merge_all()

        print('End: build model')

    def session_init(self):
        print('Start: initial session')

        self.sess = tf.Session()

        self.train_writer = tf.summary.FileWriter(self.logdir + '/train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(self.logdir + '/test')

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.sess.run(init_op)
        print('End: initial session')

    def validation(self, epoch):
        metrics_op = [self.acc, self.auc, self.loss, self.tb]

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(self.dataset.iterator.initializer, feed_dict = {
            self.dataset.files: self.dataset.testfiles,
            self.dataset.batch_size: 10000
        })

        while True:

            try:
                self.sess.run([self.acc_update, self.auc_update, self.loss_update], feed_dict = {
                    self.dataset.files: self.dataset.testfiles,
                    self.dataset.batch_size: 10000,
                    self.dropout_1: 1,
                    self.dropout_2: 1,
                    self.dropout_3: 1,
                    self.training: False,
                })

            except:
                break

        _, _, _, tb = self.sess.run(metrics_op)
        self.test_writer.add_summary(tb, epoch)

    def train(self, epochs):

        update_op = [self.acc_update, self.auc_update, self.loss_update]
        metrics_op = [self.acc, self.auc, self.loss, self.tb]

        for i in range(1, epochs + 1):
            print(str(i) + '/' + str(epochs), end = '\r')

            self.sess.run(tf.local_variables_initializer())
            self.sess.run(self.dataset.iterator.initializer, feed_dict = {
                self.dataset.files: self.dataset.trainfiles,
                self.dataset.batch_size: 200
            })

            self.sess.run([self.train_op] + update_op, feed_dict = {
                self.dataset.files: self.dataset.trainfiles,
                self.dataset.batch_size: 200,
                self.dropout_1: 0.25,
                self.dropout_2: 0.25,
                self.dropout_3: 1 - 0.298224,
                self.training: True,
            })

            _, _, _, tb = self.sess.run(metrics_op)
            self.train_writer.add_summary(tb, i - 1)

            while True:

                try:
                    self.sess.run(self.train_op, feed_dict = {
                        self.dataset.files: self.dataset.trainfiles,
                        self.dataset.batch_size: 200,
                        self.dropout_1: 0.25,
                        self.dropout_2: 0.25,
                        self.dropout_3: 1 - 0.298224,
                        self.training: True,
                    })

                except tf.errors.OutOfRangeError:
                    break

            self.validation(i)

    def close(self):
        self.sess.close()
