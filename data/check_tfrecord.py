#!/usr/bin/env python3
import tensorflow as tf
import sys

for example in tf.python_io.tf_record_iterator(sys.argv[1]):
    result = tf.train.Example.FromString(example)
    print(result)
    exit()
