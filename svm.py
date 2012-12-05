# svm.py
# Bill Waldrep, December 2012
#
# SVM training with SMO

import numpy as np
from numpy import linalg as lin
from matplotlib import pyplot as pl

class LinKernel:
  def eval(self, x1, x2):
    pass

class RBFKernel:
  def __init__(self, sigma):
    self.sigma = sigma

  def eval(self, x1, x2):
    pass

class SVM:
  def __init__(self, kernel, C):
    self.k = kernel
    self.C = C

  def train(self, X, y):
    pass

  def classify(self, X):
    pass

  def test(self, X, y):
    pass
