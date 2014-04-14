
# =========================================================================== #
#
#  SimulationResponseModel: object models the output of a simulation call.  
#
# =========================================================================== #
class SimulationResponseModel( object ):
  def __init__( self, params ):
    self.params = params
    self.load_params(params)
    
  def load_params( self, params ):
    raise NotImplementedError
    
  def response_model_rand( self ):
    raise NotImplementedError
    
  def update( self, pseudo_observations, thetas ):
    raise NotImplementedError
    
  def loglikelihood( self, observations ):
    raise NotImplementedError
    
  def loglikelihood_rand( self, observations, N = 1 ):
    raise NotImplementedError