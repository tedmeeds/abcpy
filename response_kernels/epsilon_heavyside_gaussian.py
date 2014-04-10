from abcpy.response_kernel import SimulationResponseKernel
from abcpy.helpers import gaussian_logpdf, heavyside

import numpy as np
import pylab as pp
import pdb

class EpsilonHeavysideGaussianResponseKernel( SimulationResponseKernel ):
    
  def load_params( self, params ):
    self.down = 1
    self.up   = 0
    
    if params.has_key("epsilon"):
      self.epsilon = params["epsilon"]
    if params.has_key("direction"):
      if params["direction"] == "down":
        self.direction = self.down
      elif params["direction"] == "up":
        self.direction = self.up
      else:
        assert False, "unknown direction"
    else:
      self.direction = self.down
      
    
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
      S = psh[0]
      J = 1
    
    # if S>1:
    #   pdb.set_trace()
    if N > 1:
      assert S == 1,"only allow one greater than the other"
    elif S > 1:
      N=S
          
    # start with the difference d =  y - y_star
    d = pseudo_statistics - observation_statistics
    
    if self.direction == self.up:
      d *= -1
      
    # assume every observation is outside of tube
    loglikelihood = np.zeros( (N,J) )
  
    for n in range(N):
      h = heavyside( d[n] )
      loglikelihood[n,:] = np.log( 1.0 - h + h*np.exp(  -0.5*pow( d[n]/self.epsilon, 2 ) ) )
    
    #pdb.set_trace()
    return loglikelihood