# svm.py
# Bill Waldrep, December 2012
#
# SVM training with SMO

# numerical computing
import numpy as np
from numpy import linalg as lin

# SMO class for training
from smo import SMO

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
  # Support Vector Machine Classifier
  def __init__(self, kernel, C):
    self.k = kernel
    self.C = C
    self.optimizer = SMO(kernel, C)

  def train(self, X, y):
    # store training examples
    self.supv = X
    self.supv_y = y

    # use the SMO module to compute alphas and b
    self.alphas = self.optimizer.compute_alphas(X,y)
    self.b = self.optimizer.threshold

  def _eval(self, x):
    # evaluate the SVM on a single example
    ret = 0
    for i, a in enumerate(self.alphas):
      # ignore non-support vectors
      if a != 0:
        ret += a * self.supv_y[i] * self.k.eval(x,self.supv[i])
    return ret + self.b

  def eval(self, X):
    # evaluate a matrix of example vectors
    return np.vectorize(self._eval)(X)

  def classify(self, X):
    # classify a matrix of example vectors
    return np.sign(self.eval(X))

  def test(self, X, y):
    # find the percentage of misclassified examples
    error = np.zeros(len(X))
    guess = self.classify(X)
    error[guess != y] = 1
    return np.float(np.sum(error)) / len(X)
