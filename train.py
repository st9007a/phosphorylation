#!/usr/bin/python3
import datetime
import tensorflow as tf
import os

from argparse import ArgumentParser

from modules.model import MusiteDeepModel
from modules.dataset import Dataset

if __name__ == '__main__':

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--force', '-f', action = 'store_true', help = 'Force to write log in LOGDIR')
    arg_parser.add_argument('--logdir', '-l', help = 'Setup tensorboard log directory')
    arg_parser.add_argument('--encode', '-e', default = 'identity', help = \
                            'Setup encodeing function. default run without encode\n'
                            'Option: identity onehot21 onehot22'
                            )

    args = arg_parser.parse_args()
    if args.logdir == None:
        print('Need a logdir. See python3 train.py --help')
        exit()
    elif os.path.isdir(args.logdir) and not args.force:
        print('logdir is exist')
        exit()

    data = Dataset(
        encode = args.encode, \
        trainfiles = ['data/baseline/train.tfrecord'], \
        testfiles = ['data/baseline/test.tfrecord'], \
        parallel_call = 4 \
    )

    model = MusiteDeepModel(dataset = data, logdir = args.logdir)

    start = datetime.datetime.now()
    model.train(epochs = 500)
    end = datetime.datetime.now()
    print(end - start)

    model.close()
