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
    self.noise = noise
    self.kernel = GaussianKernel(sigma)
    self.sigma = sigma

  def fit(self, data):
    self.tX, self.ty = np.mat(data[:,0]),np.mat(data[:,1])
    self.Kinv = lin.inv(self.kernel.build_matrix(data[:,0]) + (self.noise**2 * np.identity(len(data[:,0]))))

  def _predict(self, x):
    kt = self.kernel.eval_row(x,self.tX)
    y = kt * self.Kinv * self.ty.T
    vy = self.kernel.eval(x,x) - (kt * self.Kinv * kt.T)
    return y, vy

  def predict(self, X):
    return np.vectorize(self._predict)(X)

  def eval(self, data):
    """return squared error loss"""
    pred, var = self.predict(data[:,0])
    return ((data[:,1] - pred)**2).sum()

class ParameterEstimator:
  def __init__(self, data, classifier=GaussianProcess, kfolds=5):
    self.c = classifier
    self.k = kfolds
    self.data = data.copy()
    np.random.shuffle(self.data)
    self.partitions = np.array_split(self.data, self.k)

  def _eval_corner(self, sigma, noise):
    err = np.zeros(self.k)
    for i in range(self.k):
      test = self.partitions[i]
      train = np.vstack([self.partitions[x] for x in range(self.k) if x != i])
      c = self.c(sigma=sigma,noise=noise)
      c.fit(train)
      err[i] = c.eval(test)
    return np.mean(err)

  def grid_search(self, sigmas, noises):
    min_err = np.inf
    for sig in sigmas:
      for noise in noises:
        err = self._eval_corner(sig, noise)
        print "sig: %f, noise %f, err %f, best %f" % (sig, noise, err, min_err)
        if err < min_err:
          self.sigma = sig
          self.noise = noise
          min_err = err
    self.final = self.c(sigma=self.sigma, noise=self.noise)

  def plot_result(self, data):
    x,y = data[:,0], data[:,1]
    mesh = np.linspace(np.min(x), np.max(x), num=150)
    self.final.fit(data)
    pred, var = self.final.predict(mesh)
    sigma = np.sqrt(var)
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

p = ParameterEstimator(load_data(),kfolds=5)
p.grid_search(sigmas=np.linspace(2.5,8.5,num=100),noises = np.linspace(0,2,num=20))
print "sigma:", p.final.sigma
print "noise:", p.final.noise
p.plot_result(load_data())
