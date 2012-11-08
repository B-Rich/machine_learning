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
