
# =========================================================================== #
#
#  SimulationResponseModel: object models the output of a simulation call.  
#
# =========================================================================== #
class SimulationResponseModel( object ):
  def __init__( self, params ):
    self.is_model = True
    self.params = params
    self.load_params(params)
    self.pseudo_statistics = []
    self.thetas = []
    
  def is_empty( self ):
    raise NotImplementedError
    
  def load_params( self, params ):
    raise NotImplementedError
    
  #def response_model_rand( self ):
  #  raise NotImplementedError
    
  def add(self, thetas, pseudo_statistics, observation_statistics ):
    raise NotImplementedError
    
  def update( self ):
    raise NotImplementedError
    
  def loglikelihood( self, theta, bservations ):
    raise NotImplementedError
    
  def loglikelihood_rand( self, theta, observations, N = 1 ):
    raise NotImplementedError
    
  def update_post_mh( self, observation_group, simulation_statistics, params ):
    pass