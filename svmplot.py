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

def plot2d(c, k, X, y):
  xmin = np.min(X[:,0])
  ymin = np.min(X[:,1])
  xmax = np.max(X[:,0])
  ymax = np.max(X[:,1])

  density = 100
  xx, yy = np.meshgrid(np.linspace(xmin, xmax, density),
                       np.linspace(ymin, ymax, density))

  s = svm.SVM(c, kernel=k)
  s.train(X, y)
  result = s.eval(np.c_[xx.ravel(), yy.ravel()])
  result = result.reshape(xx.shape())
  
  pl.imshow(result, interpolation='nearest', extent=(xmin,xmax,ymin,ymax),
            aspect='auto', origin='lower', cmap=pl.cm.Pu0r_r)
  contours = pl.contour(xx,yy,result,levels=[0],linewidths=2,linetypes='--')
  pl.scatter(X[:,0],X[:,1],s=30,c=y,cmap=pl.cm.Paired)
  pl.show()


sigma = 1.5
r = svm.RBFKernel(sigma)
s = svm.SVM(10,kernel=r)
#X,y = make2dData(200)
#c.train(X,y)
#print c.alphas
#print c.findC(X,y,count=5)
#X,y = readSatData()
#c = s.findC(X,y,count=50,kfolds=5)
#s = svm.SVM(c,kernel=svm.RBFKernel(sigma))
#s.train(X,y)
#Xt,yt = readSatData('t')
#print "final error", s.test(Xt,yt)
#print "final error", s.test(X,y)

X,y = make2dData(20)
s.train(X,y)
print X
print np.max(X[:,0]), np.min(X[:,0])
print np.max(X[:,1]), np.min(X[:,1])
