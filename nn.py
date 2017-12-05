#!/usr/bin/python3
import tensorflow as tf

def weight_var(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_var(shape):
    return tf.Variable(tf.constant(0.1, shape = shape))

x = tf.placeholder(tf.float32, shape = [None, 33, 21, 1])
y = tf.placeholder(tf.float32, shape = [None, 1])

w_conv1 = weight_var([1, 21, 1, 200])
b_conv1 = bias_var([200])
h_conv1 = tf.nn.conv2d(x, w_conv1, strides = [1, 1, 1, 1], padding = 'VALID')
h_conv1 = tf.nn.dropout(h_conv1, keep_prob = 0.75)
h_conv1 = tf.nn.relu(h_conv1) + b_conv1

w_conv2 = weight_var([9, 1, 200, 150])
b_conv2 = bias_var([150])
h_conv2 = tf.nn.conv2d(h_conv1, w_conv2, strides = [1, 1, 1, 1], padding = 'VALID')
h_conv2 = tf.nn.dropout(h_conv2, keep_prob = 0.75)
h_conv2 = tf.nn.relu(h_conv2) + b_conv2

w_conv3 = weight_var([10, 1, 150, 200])
b_conv3 = bias_var([200])
h_conv3 = tf.nn.conv2d(h_conv2, w_conv3, strides = [1, 1, 1, 1], padding = 'VALID')
h_conv3 = tf.nn.relu(h_conv3) + b_conv3

seq_domain_entry = tf.reshape(h_conv3, [-1, 16, 200])
fm_domain_entry = tf.transpose(seq_domain_entry, perm = [0, 2, 1])

def attention(tensor):
    input_dim = int(tensor.get_shape()[2])
    input_length = int(tensor.get_shape()[1])

    tensor_unpack = tf.unstack(tensor, axis = 1)

    w0 = weight_var([input_dim, 10])
    b0 = bias_var([10])
    energy = [tf.matmul(t, w0) + b0 for t in tensor_unpack]

    w1 = weight_var([10, 1])
    b1 = bias_var([1])
    energy = [tf.matmul(t, w1) + b1 for t in energy]

    energy = tf.stack(energy, axis = 1)
    energy = tf.reshape(energy, [-1, input_length])
    energy = tf.nn.softmax(energy)

    weighted_sum = tf.reduce_sum(tensor * tf.expand_dims(energy, -1), 1)
    return weighted_sum

merged = tf.concat([attention(seq_domain_entry), attention(fm_domain_entry)], axis = 1)

w_output = weight_var([200 + 16, 1])
b_output = bias_var([1])
raw_output = tf.matmul(merged, w_output) + b_output

