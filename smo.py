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
    self.threshold = 0

  def compute_alphas(self, X, y):
    # store training examples
    self.examples = X
    self.ex_labels = y

    # initialize array of alphas
    self.alphas = np.zeros(len(X))

    # set flags for main loop
    dirty = False
    examineAll = True

    # main training loop
    while dirty or examineAll:
      dirty = False
      if examineAll:
        # consider all examples
        for i in len(self.examples):
          dirty = examineEx(i) or dirty
        examineAll = False
      else:
        # consider suspicious examples
        for i, alpha in enumerate(self.alphas):
          if alpha != 0 and alpha != self.C:
            dirty = examineEx(i) or dirty
        # if nothing changed recheck the whole training set
        if not dirty:
          examineAll = True

    return self.alphas
