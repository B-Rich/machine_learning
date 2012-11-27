# hmm.py
# Bill Waldrep, November 2012
#
# Hidden Markov Model for text processing

import numpy as np
import numpy.linalg as lin
from random import random

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

def log_normalize_matrix(mat):
  """Normalize each row without underflow"""
  return np.apply_along_axis(log_normalize, 1, mat)

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
    self.start = start_vec
    self.tran = transitions
    self.obs = emissions
    self.num_hidden = len(transitions)
    self.num_observed = 26

 def viterbi(self, emissions):
   """Calculate most likely sequence of hidden states
      to result in 'ys'"""
   obs_count = len(ys)
   states = range(self.num_hidden):

   # initialize DP table
   table = np.zeros((self.num_hidden, obs_count))
   path = np.zeros((self.num_hidden, obs_count))

   # fill in first row
   for s in states:
     # add log values
     table[s,0] = self.start[s] + self.obs[s][ys[0]]
     path[s,0] = s

   # compute rest of table
   for t in range(1,obs_count):
     for s in states:
       guess = -1 * np.inf
       for ns in states:
         # again, add log values
         p = table[ns, t-1] + self.tran[ns,s] + self.obs[s][ys[t]]
         if p < guess:
           guess = p
           path[s,t] = ns
       table[s,t] = guess

   # reconstruct path
   fpath = []
   for t in range(obs_count).reverse():
     p = table[0,t]
     g = 0
     for s in states:
       if table[s,t] > p:
         p = table[s,t]
         g = s
     fpath.append(g)

   return fpath.reverse()
