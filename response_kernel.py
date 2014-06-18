
# =========================================================================== #
#
#  SimulationResponseKernel: object models the output of a simulation call.  
#
# =========================================================================== #
class SimulationResponseKernel( object ):
  def __init__( self, params ):
    self.params = params
    self.load_params( params )
    
  def load_params( self, params ):
    raise NotImplementedError
    
  def loglikelihood( self, observations, pseudo_observations ):
    raise NotImplementedError
        
  def add( self, thetas, pseudo_statistics, observation_statistics ):
    pass