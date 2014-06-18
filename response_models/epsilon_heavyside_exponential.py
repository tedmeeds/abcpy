from abcpy.response_models.epsilon_heavyside_gaussian import EpsilonHeavysideGaussianResponseModel # import SimulationResponseModel
from abcpy.helpers import mvn_logpdf, mvn_diagonal_logpdf, mvn_diagonal_logcdf
from abcpy.helpers import gaussian_logpdf, heavyside
import numpy as np
import pdb
  
class EpsilonHeavysideExponentialResponseModel( EpsilonHeavysideGaussianResponseModel ):
  
    
  def loglikelihood_kernel( self, observation_statistics, pseudo_statistics ):
    # start with the difference d =  y - y_star
    J = len(pseudo_statistics)
    d = pseudo_statistics - observation_statistics
    
    if self.direction == self.up:
      d *= -1
    
    N = 1
    # assume every observation is outside of tube
    loglikelihood = np.zeros( J )
  
    for j in range(J):
      h = heavyside( d[j] )
      if h > 0.5:
        loglikelihood[j] = -d[j]/self.epsilon[j]
        
      if self.cliff_at is not None:
        if pseudo_statistics[j] == self.cliff_at[j]:
          loglikelihood[j] = -np.inf
    #pdb.set_trace()
    return np.sum(loglikelihood)