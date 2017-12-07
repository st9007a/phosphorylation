#!/usr/bin/python3
import numpy as np
from nn import Classifier
from utils import next_batch

clf = Classifier()

clf.train(200, 1000000)


