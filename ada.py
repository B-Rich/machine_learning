# ada.py
# Bill Waldrep, November 2012
#
# Implementation of AdaBoost with decision stubs
# as weak learners.

import numpy as np
import scipy as sci
import math as m
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_data(suffix='train.csv'):
  data_file = 'data/ada_x_' + suffix
  label_file = 'data/ada_y_' + suffix
  delim = ','
  data = np.loadtxt(data_file, delimiter=delim)
  labels = np.loadtxt(label_file, delimiter=delim)
  return data, labels

def show_image(img):
  """Plot the input as a greyscale
  image for sanity checking"""
  temp = np.reshape(x[3], (28,28))
  plt.imshow(temp, cmap = cm.Greys_r)
  plt.show()

class DecisionStump:
  def __init__(self, dimension=-1, thresh=-1, inequal=-1):
    self.dim = dimension
    self.t = thresh
    self.iq = inequal

  def classify(self, data):
    # start by labeling everything 1
    results = np.ones(len(data),'int')
    if self.iq == "gte":
      # flip the labels of everything greater than or equal to the threshold
      results[data[:,self.dim] >= self.t] = -1
    else:
      # flip the labels of everything less than or equal to the threshold
      results[data[:,self.dim] <= self.t] = -1
    return results

  def train(self, data, label, D):
      rows, cols = np.shape(data)
      min_error = np.inf
      for dim in xrange(cols):
        # get all possible thresholds we could split on
        vals = np.unique(data[:,dim])
        for t in vals:
          for flip in ["gte", "lte"]:
            candidate = DecisionStump(dim,t,flip)
            guess = candidate.classify(data)
            # initialize to zero error
            error = np.zeros(cols,'int')
            error[guess != label] = 1
            weighted = np.dot(error,D)
            #print "dim %d, t %d, iq %s, w %f, min %f" % (dim, t, flip, weighted, min_error)
            if weighted < min_error:
              min_error = weighted
              self.dim = dim
              self.t = t
              self.iq = flip

  def debug(self):
    print self.dim, self.t, self.iq

# Debugging
x,y = load_data()
d = DecisionStump()
D = np.ones(len(x[0]),'float')
d.train(x,y,D)
d.debug()
