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

plot_data(load_data())
