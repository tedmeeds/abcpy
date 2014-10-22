from simulation_response_kernel import SimulationResponseKernel
import numpy as np
import pylab as pp
import pdb

class EpsilonTubeResponseKernel( SimulationResponseKernel ):
    
  def load_params( self, params ):
    if params.has_key("epsilon"):
      self.epsilon = params["epsilon"]
      self.lower_epsilon = -self.epsilon
      self.upper_epsilon = self.epsilon
      
    if params.has_key( "upper_epsilon" ) and params.has_key( "lower_epsilon" ):
      self.lower_epsilon = params["lower_epsilon"]
      self.upper_epsilon = params["upper_epsilon"]
    
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
    

    if N > 1:
      assert S == 1,"only allow one greater than the other"
    elif S > 1:
      N=S
          
    # start with the difference d =  y - y_star
    d = pseudo_statistics - observation_statistics
    
    # assume every observation is outside of tube
    loglikelihood = -np.inf*np.ones( (N,J) )
  
    for n in range(N):
      # find those inside tube and give loglikelihood of 0
      I = pp.find( ( d[n] <= self.upper_epsilon ) and (d[n] >= self.lower_epsilon) )
      loglikelihood[n,I] = 0 
    
    return loglikelihood