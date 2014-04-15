from abcpy.helpers import gaussian_logpdf
import numpy as np
import pylab as pp

def heavyside( x ):
  y = np.zeros( x.shape )
  I = pp.find( x >= 0 )
  y[I] = 1
  return y
  
def log_gaussian_kernel( x, y, epsilon ):
  # will require a multivariate solution for full cov epsilon
  return np.sum(gaussian_logpdf( x, y, epsilon  ))
  
def one_sided_gaussian_kernel( x, y, epsilon ):
  d = x - y; d = -d
  
  h = heavyside(d)
  
  I = pp.find(h == 0)
  
  logprob = np.zeros( d.shape )
  
  logprob[I] = -0.5*pow( d[I]/epsilon, 2 )
  
  return logprob
  
def one_sided_exponential_kernel( x, y, epsilon ):
  d = x - y; d = -d
  
  h = heavyside(d)
  
  I = pp.find(h == 0)
  
  logprob = np.zeros( d.shape )
  
  logprob[I] = pow( d[I]/epsilon, 1 )
  
  return logprob