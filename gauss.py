# gauss.py
# Bill Waldrep, November 2012
#
# Gaussian Process regression with a Gaussian kernel

import numpy as np
import scipy as sci
import math as m
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

class Gauss:
  def __init__(self, sigma=0.5):
    self.sig = sigma

  def get_kernel_diff(self, x1, x2):
    top = -1 * m.sqrt(x1**2 + x2**2)**2
    return m.exp(top/(2*self.sig**2))

  def build_kernel(self, data):
    n = len(data)
    self.K = np.reshape(np.zeros(n**2), (n,n))
    for i in xrange(n):
      for j in xrange(n):
        self.K[i][j] = self.get_kernel_diff(data[i],data[j])




plot_data(load_data())
