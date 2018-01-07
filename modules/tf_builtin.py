#!/usr/bin/python3
import tensorflow as tf

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
    energy = [t @ w0 + b0 for t in tensor_unpack]

    w1 = weight_var([dim, 1])
    b1 = bias_var([1])
    energy = [t @ w1 + b1 for t in energy]

    energy = tf.stack(energy, axis = 1)
    energy = tf.reshape(energy, [-1, input_length])
    energy = tf.nn.softmax(energy, 1)

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

def conv_layer(input_tensor, num_fms, filter_size, strides, dropout = None, act = tf.nn.relu, padding = 'SAME'):
    curr = int(input_tensor.get_shape()[1])
    w = weight_var(filter_size + [curr, num_fms])
    b = bias_var([num_fms])
    h = tf.nn.conv2d(input_tensor, w, strides = strides, padding = padding, data_format = 'NCHW') + b
    h = batch_normalization(h)

    dropout_layer = None
    if dropout is True:
        dropout_layer = tf.placeholder(tf.float32)
        h = tf.nn.dropout(h, keep_prob = dropout_layer)
    h = act(h)

    return h, dropout_layer
