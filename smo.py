# smo.py
# Bill Waldrep, December 2012
#
# Sequential Minimal Optimization

# numerical computing
import numpy as np
from numpy import linalg as lin

class SMO:
  def __init__(self, kernel, C):
    self.k = kernel
    self.C = C

  def compute_alphas(self, X, y):
    raise NotImplementedError
