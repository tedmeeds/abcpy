class BaseSurrogate( object ):
  def __init__( self, params ):
    self.params = params
    self.nbr_sim_calls_this_iter = 0
    self.init_with_params( params )
    
  def init_with_params( self, params ):
    self.obs_statistics    = params["obs_statistics"]
    self.run_sim_and_stats = params["run_sim_and_stats_func"]
    
  def loglik_differences_rand( self, to_theta, from_theta, M ):
    raise NotImplementedError
        
  def acquire_points( self, to_theta, from_theta, M ):
    raise NotImplementedError
    