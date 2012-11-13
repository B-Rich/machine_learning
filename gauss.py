# gauss.py
# Bill Waldrep, November 2012
#
# Gaussian Process regression with a Gaussian kernel

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

    # constant value
    self.denom = -2 * self.sigma**2

    # eval generalized to take vector inputs
    self.vec_eval = np.vectorize(self.eval)

  def eval(self, x1, x2):
    # compute the covariance between two inputs using the Gaussian kernel
    return math.exp(lin.norm(x1 - x2)**2 / self.denom)

  def eval_row(self, x1, x):
    # call eval with x1 and every value in x
    return np.mat(self.vec_eval(x1,x))

  def build_matrix(self, X):
    # compute the covariance of all pairs
    return np.mat(self.vec_eval(X[:,np.newaxis],X))

class GaussianProcess:
  def __init__(self, noise=0.5, sigma=0.5):
    # save these values so we can print them later
    self.noise = noise
    self.sigma = sigma

    # initialize kernel
    self.kernel = GaussianKernel(sigma)

  def fit(self, data):
    # store training dataset
    self.tX, self.ty = np.mat(data[:,0]),np.mat(data[:,1])

    # compute covariance matrix
    K = self.kernel.build_matrix(data[:,0]) + (self.noise**2 * np.identity(len(data[:,0])))

    # account for noise
    self.K = K + (self.noise**2 * np.identity(len(data[:,0])))
    
    # inversion is expensive, so we store it
    self.Kinv = lin.inv(self.K)

  def _predict(self, x):
    # make a prediction for a particular x
    # we use kt several times, so compute it once
    kt = self.kernel.eval_row(x,self.tX)

    # compute the predicted y
    y = kt * self.Kinv * self.ty.T

    # compute the predicted variance
    vy = self.kernel.eval(x,x) - (kt * self.Kinv * kt.T)
    return y, vy

  def predict(self, X):
    # makes predictions for a vector of inputs
    return np.vectorize(self._predict)(X)

  def eval(self, data):
    #return squared error loss
    pred, var = self.predict(data[:,0])
    return ((data[:,1] - pred)**2).sum()

class ParameterEstimator:
  def __init__(self, data, regressor=GaussianProcess, kfolds=5):
    self.r = regressor
    self.k = kfolds

    # initialize the data partitions
    self.data = data.copy()
    np.random.shuffle(self.data)
    self.partitions = np.array_split(self.data, self.k)

  def _eval_point(self, sigma, noise):
    # evaluate the error for a particular sigma/noise combination
    err = np.zeros(self.k)
    for i in range(self.k):
      test = self.partitions[i]
      
      # the training set is all data not in the i'th partition
      train = np.vstack([self.partitions[x] for x in range(self.k) if x != i])
      c = self.r(sigma=sigma,noise=noise)
      c.fit(train)

      # save the error on this partition
      err[i] = c.eval(test)

    # return the mean over all testing partitions
    return np.mean(err)

  def grid_search(self, sigmas, noises):
    min_err = np.inf

    # check all combinations of parameters
    for sig in sigmas:
      for noise in noises:
        err = self._eval_point(sig, noise)
        if err < min_err:
          self.sigma = sig
          self.noise = noise
          min_err = err

    # initialize a regressor with the optimal parameters to plot our result
    self.final = self.r(sigma=self.sigma, noise=self.noise)
    self.final.fit(data)

  def plot_result(self, data):
    x,y = data[:,0], data[:,1]

    # make predictions at 150 points along our curve
    mesh = np.linspace(np.min(x), np.max(x), num=150)
    pred, var = self.final.predict(mesh)
    sigma = np.sqrt(var)

    # matplotlib nastiness
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

##################################################
############# Excecute Regression ################
##################################################

# Initialize the estimator with 5 folds
p = ParameterEstimator(load_data(),kfolds=5)

# search for sigma \in [2.5, 8.5] and noise \in [0, 2]
p.grid_search(sigmas=np.linspace(2.5,8.5,num=100),noises = np.linspace(0,2,num=20))

# print the final values
print "sigma:", p.final.sigma
print "noise:", p.final.noise

# plot the resulting curve
p.plot_result(load_data())
