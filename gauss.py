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
    self.denom = -2 * self.sigma**2
    self.vec_eval = np.vectorize(self.eval)

  def eval(self, x1, x2):
    return math.exp(lin.norm(x1 - x2)**2 / self.denom)

  def eval_row(self, x1, x):
    return np.mat(self.vec_eval(x1,x))

  def build_matrix(self, X):
    return np.mat(self.vec_eval(X[:,np.newaxis],X))

class GaussianProcess:
  def __init__(self, noise=0.5, sigma=0.5):
    self.sig = noise**2
    self.kernel = GaussianKernel(sigma)

  def fit(self, X, y):
    self.tX, self.ty = np.mat(X),np.mat(y)
    self.Kinv = lin.inv(self.kernel.build_matrix(X) + (self.sig * np.identity(len(X))))

  def _predict(self, x):
    kt = self.kernel.eval_row(x,self.tX)
    y = kt * self.Kinv * self.ty.T
    vy = self.kernel.eval(x,x) - (kt * self.Kinv * kt.T)
    return y, vy

  def predict(self, X):
    return np.vectorize(self._predict)(X)

  def eval(self, data):
    x,y = data[:,0], data[:,1]
    pred, var = self.predict(x)
    # return squared error loss
    return ((y - pred)**2).sum()

def plot_regression(data,sigma,noise):
  x = data[:,0]
  y = data[:,1]
  gp = GaussianProcess(noise=noise, sigma=sigma)
  mesh = np.linspace(np.min(x),np.max(x),num=150)
  gp.fit(x,y)
  pred, var = gp.predict(mesh)
  sigma = np.sqrt(var)
  print gp.eval(data)

  fig = pl.figure()
  pl.plot(x,y,'r.',markersize=10,label=u'Observations')
  pl.plot(mesh,pred, 'b-', label=u'Prediction')
  pl.fill(np.concatenate([mesh,mesh[::-1]]), \
          np.concatenate([pred - 1.96 * sigma,
                         (pred + 1.96 * sigma)[::-1]]), \
          alpha=.5, fc='b', ec='None', label='95% confidence interval')
  pl.xlabel('$x$')
  pl.ylabel('$y$')
  pl.legend(loc='upper left')
  pl.show()

plot_regression(load_data(),3.0,0.5)
#plot_data(load_data())
