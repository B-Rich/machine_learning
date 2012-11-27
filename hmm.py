# hmm.py
# Bill Waldrep, November 2012
#
# Hidden Markov Model for text processing

import numpy as np
import numpy.linalg as lin

def load_data(filename='data/Alice.txt'):
  """Load the file as a list of strings"""
  with open(filename, 'r') as f:
    s = map(lambda x:x.strip('\r\n'),f.readlines())
    return s

def normalize(prob_vec):
  """Normalize a probability vector"""
  s = np.sum(prob_vec)
  return(prob_vec / s)

def normalize_matrix(mat):
  """Normalize each row in mat"""
  return np.apply_along_axis(normalize, 1, mat)

def log_normalize(log_prob_vec):
  """Normalize a log probability vector without underflow"""
  log_sum = np.logaddexp.reduce(log_prob_vec)
  return(log_prob_vec - log_sum)

def index_obs(obs_char):
  """Convert observed character to observed node index"""
  if obs_char == '*':
    return 27
  elif obs_char == ' ':
    return 26
  else:
    return ord(obs_char) - ord('a')

def show_obs(obs_index):
  """Convert observed node index back to character"""
  if obs_index == 27:
    return '*'
  elif obs_char == ' ':
    return ' '
  else:
    return chr(obs_index + ord('a'))

class HiddenMarkovModel:
  def __init__(self, start_vec, transitions, emissions):
    self.num_hidden = len(transitions)
    self.num_observed = len(emissions)

