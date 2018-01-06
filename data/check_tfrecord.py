#!/usr/bin/python3
import tensorflow as tf

for example in tf.python_io.tf_record_iterator("baseline/train.tfrecord"):
    result = tf.train.Example.FromString(example)
    print(result)
    exit()
