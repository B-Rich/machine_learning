import numpy as np
import matplotlib
from matplotlib import pyplot as pl

fname = 'a.out'
data = np.loadtxt(fname)

def plot_alphas(d):
  fig = pl.figure()
  pl.plot(d[:,0], d[:,1], 'r-')
  pl.xlabel('T')
  pl.ylabel('Alpha')
  pl.show()

def plot_errors(d):
  fig = pl.figure()
  pl.plot(d[:,0], d[:,2], 'b-', label='training error')
  pl.plot(d[:,0], d[:,3], 'r-', label='testing error')
  pl.xlabel('T')
  pl.ylabel('Error')
  pl.legend(loc='upper right')
  pl.show()

plot_alphas(data)
plot_errors(data)
