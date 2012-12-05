# svmplot.py
# Bill Waldrep, December 2012
#
# Utility functions for running/testing the svm
# class and plotting the graphs requested for 
# homework 4.

# numeric stuff
import numpy as np

# generate test data
from sklearn.datasets.samples_generator import make_blobs

# actual svm module
import svm

def make_2d_data(n_samples):
  # generate two clusters of 
  return make_blobs(n_samples=n_samples, centers=2)


