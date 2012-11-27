# hmm.py
# Bill Waldrep, November 2012
#
# Hidden Markov Model for text processing

import numpy as np
import numpy.linalg as lin

def load_data(filename='data/Alice.txt'):
  with open(filename, 'r') as f:
    s = map(lambda x:x.strip('\r\n'),f.readlines())
    return s
