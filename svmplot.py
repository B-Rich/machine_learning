# svmplot.py
# Bill Waldrep, December 2012
#
# Utility functions for running/testing the svm
# class and plotting the graphs requested for 
# homework 4.

# numeric stuff
import numpy as np

# plotting data
from matplotlib import pyplot as pl

# generate test data
from sklearn.datasets.samples_generator import make_blobs

# actual svm module
import svm

def make2dData(n_samples):
  # generate a 2d 2-class dataset
  X,y = make_blobs(n_samples=n_samples, centers=2)
  y[y==0] = -1
  return X,y

def readSatData(fname='tr'):
  data = []
  with open('data/satimage.scale.' + fname, 'r') as f:
    for line in f.readlines():
      s = line.split()
      row = np.zeros(37)
      # set label
      if s[0] == '6':
        row[0] = 1
      else:
        row[0] = -1
      hand = 1
      for i in range(1,37):
        k,v = s[hand].split(':')
        if int(k) == i:
          hand += 1
          row[i] = float(v)
        else:
          row[i] = 0
      data.append(row)
  data = np.vstack(data)
  y = data[:,0]
  X = data[:,1:]
  return X,y

r = svm.RBFKernel(2)
s = svm.SVM(10,kernel=r)
#X,y = make2dData(200)
#c.train(X,y)
#print c.alphas
#print c.findC(X,y,count=5)
X,y = readSatData()
#c = s.findC(X,y,count=10,kfolds=3)
c = 8.3
s = svm.SVM(c,kernel=svm.RBFKernel(2))
s.train(X,y)
Xt,yt = readSatData('t')
print "final error", s.test(Xt,yt)
#print "final error", s.test(X,y)
