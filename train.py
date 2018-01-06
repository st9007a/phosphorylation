#!/usr/bin/python3
import datetime
import tensorflow as tf
from modules.model import MusiteDeepModel
from modules.dataset import Dataset

if __name__ == '__main__':

    data = Dataset(
        trainfiles = ['data/baseline/train.tfrecord'], \
        testfiles = ['data/baseline/test.tfrecord'], \
        batch_size = 200, \
        parallel_call = 4 \
    )

    model = MusiteDeepModel(dataset = data, logdir = 'log/baseline/onehot21-1')

    start = datetime.datetime.now()
    model.train(epochs = 500)
    end = datetime.datetime.now()
    print(end - start)

    model.close()
