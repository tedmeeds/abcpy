class GaussianResponseModel( SimulationResponseModel ):
  
  def load_params( self, params ):
    raise NotImplementedError
    
  def response_model_rand( self ):
    raise NotImplementedError
    
  def update( self, pseudo_observations, thetas ):
    raise NotImplementedError
    
  def loglikelihood_of_observations( self, observations ):
    raise NotImplementedError
    
  def loglikelihood_of_observations_rand( self, observations, N = 1 ):
    raise NotImplementedError