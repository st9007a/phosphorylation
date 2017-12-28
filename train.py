#!/usr/bin/python3
import numpy as np
from nn import MusiteDeepClassifier

clf = MusiteDeepClassifier()
clf.train(200, 10000)
