from abcpy.models.metropolis_hastings_model import AdaptiveMetropolisHastingsModel
import numpy as np
import pylab as pp
  
class SurrogateMetropolisHastingsModel( AdaptiveMetropolisHastingsModel ):
    
  def load_params( self, params ):
    super(SurrogateMetropolisHastingsModel, self).load_params(params)
    self.surrogate = params["surrogate"]
    
  def acquire_points( self ):
    if np.random.rand() < 0.5:
      self.proposed.acquire( self.deltaS )
    else:
      self.current.acquire( self.deltaS )