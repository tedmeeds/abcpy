from simulation_response_kernel import SimulationResponseKernel
from abcpy.helpers import gaussian_logpdf

import numpy as np
import pylab as pp
import pdb

class EpsilonGaussianResponseKernel( SimulationResponseKernel ):
    
  def load_params( self, params ):
    if params.has_key("epsilon"):
      self.epsilon = params["epsilon"]
    
  def loglikelihood( self, observation_statistics, pseudo_statistics ):
    sh = observation_statistics.shape
    if len(sh) > 1:
      N,J = sh
    else:
      N = sh[0]
      J = 1
    
    psh = pseudo_statistics.shape
    if len(psh) > 1:
      S,J = psh
    else:
      if J == 1:
        S = psh[0]
      else:
        S = 1
    
    if N > 1:
      assert S == 1,"only allow one greater than the other"
    elif S > 1:
      N=S
          
    # start with the difference d =  y - y_star
    d = pseudo_statistics - observation_statistics
    
    # assume every observation is outside of tube
    loglikelihood = -np.inf*np.ones( (N,J) )
  
    for n in range(N):
      loglikelihood[n,:] = gaussian_logpdf( d[n], 0, self.epsilon ) 
    
    return loglikelihood.sum(1)