import numpy as np
import scipy as sp
import pylab as pp

import scipy
from scipy import special
from scipy import stats
from scipy import integrate
import pdb

sqrt2 = np.sqrt( 2.0 )
sqrt_2pi = np.sqrt( 2.0 * np.pi )
log_sqrt_2pi = np.sqrt(sqrt_2pi)

def bin_errors_1d( bins, true_centered_probability, samples ):
  N = len(samples)
  cnts, bins = np.histogram( samples ,bins=bins)
  
  n = cnts.sum()
  
  probability = cnts / (float(N))
  
  missed = N - n
  
  double_error = np.sum(np.abs( true_centered_probability - probability )) + missed/float(N)
  
  #pdb.set_trace()
  return double_error/2.0
    
def logsumexp(x,dim=0):
    """Compute log(sum(exp(x))) in numerically stable way."""
    #xmax = x.max()
    #return xmax + log(exp(x-xmax).sum())
    if dim==0:
        xmax = x.max(0)
        return xmax + np.log(np.exp(x-xmax).sum(0))
    elif dim==1:
        xmax = x.max(1)
        return xmax + np.log(np.exp(x-xmax[:,newaxis]).sum(1))
    else: 
        raise 'dim ' + str(dim) + 'not supported'
        
def gamma_logprob( x, alpha, beta ):
  if all(x>0):
    if beta > 0:
      return alpha*np.log(beta) - special.gammaln( alpha ) + (alpha-1)*np.log(x) - beta*x
    else:
      assert False, "Beta is zero"
  else:
    if x.__class__ == np.array:
      I = pp.find( x > 0.0 )
      lp = -np.inf*np.zeros( x.shape )
      lp[I] = alpha*np.log(beta) - special.gammaln( alpha ) + (alpha-1)*np.log(x[I]) - beta*x[I]
      return lp
    else:
      print "*****************"
      print "gamma_logprob returning -INF"
      print "*****************"
      return -np.inf

def gen_gamma_logpdf( alpha, beta ):
  def gamma_logpdf( theta ):
    return gamma_logprob( theta, alpha, beta  )
  return gamma_logpdf
        
def gen_gamma_cdf( alpha, beta ):
  g = stats.gamma(alpha,0,1.0/beta)
  def gamma_cdf( theta ):
    return g.cdf(theta)
  return gamma_cdf