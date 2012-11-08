# gauss.py
# Bill Waldrep, November 2012
#
# Gaussian Process regression with a Gaussian kernel

import numpy as np
import scipy as sci
import math as m
import matplotlib as mplot

def load_data():
  filename = "data/motor.dat"
  skip = 37
  cols = (1,2)
  data = np.loadtxt(filename, skiprows=skip, usecols=cols)
  return data
