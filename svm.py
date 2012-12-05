# svm.py
# Bill Waldrep, December 2012
#
# SVM training with SMO

# numerical computing
import numpy as np
from numpy import linalg as lin

# plotting data
from matplotlib import pyplot as pl

# generate test data
from sklearn.datasets.samples_generator import make_blobs

class LinKernel:
  # Linear Kernel
  def eval(self, x1, x2):
    return np.dot(x1,x2)

class RBFKernel:
  # Gaussian Kernel
  def __init__(self, sigma):
    self.sigma = sigma
    self.gamma = 1 / (2.0 * sigma**2)

  def eval(self, x1, x2):
    return np.exp(-gamma * lin.norm(x1 - x2)**2)

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
