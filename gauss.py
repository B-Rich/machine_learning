# gauss.py
# Bill Waldrep, November 2012
#
# Gaussian Process regression with a Gaussian kernel

import math
import numpy as np
import numpy.linalg as lin
import matplotlib
from matplotlib import pyplot as pl

def load_data():
  filename = "data/motor.dat"
  skip = 37
  cols = (0,1)
  data = np.loadtxt(filename, skiprows=skip, usecols=cols)
  return data

def plot_data(data):
  fig = pl.figure()
  pl.plot(data[:,0], data[:,1], 'r:', label='foobar')
  pl.xlabel('x')
  pl.ylabel('y')
  pl.legend(loc='upper left')

  pl.show()

class GaussianKernel:
  def __init__(self, sigma=0.5):
    self.sigma = sigma
    self.denom = -1 * sigma**2

  def eval(self, x1, x2):
    return math.exp(lin.norm(x1 + x2) / self.denom)

  def build_matrix(self, X):
    n = len(X)
    K = np.zeros((n,n))
    for i in xrange(n):
      for j in xrange(n):
        K[i][j] = self.eval(X[i],X[j])
    return K

class GaussianProcess:
  def __init__(self, noise=0.5, sigma=0.5):
    self.sig = noise
    self.kernel = GaussianKernel(sigma)

  def fit(self, x):
    self.K = self.kernel.build_matrix(x)

def magic(data):
  sigma = 0.5
  x = data[:,0]
  y = data[:,1]
  gp = GaussianProcess(noise=0.5)
  gp.fit(x)
  pred = y

  fig = pl.figure()
  pl.plot(x,y,'r.',markersize=10,label=u'Observations')
  pl.plot(x,pred, 'b-', label=u'Prediction')
  pl.fill(np.concatenate([x,x[::-1]]), \
          np.concatenate([pred - 1.96 * sigma,
                         (pred + 1.96 * sigma)[::-1]]), \
          alpha=.5, fc='b', ec='None', label='95% confidence interval')
  pl.xlabel('$x$')
  pl.ylabel('$y$')
  pl.ylim(-10,20)
  pl.legend(loc='upper left')

magic(load_data())
plot_data(load_data())
