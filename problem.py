
# =========================================================================== #
#
#  BaseProblem: A "problem" wraps the simulator call, statistics functions, 
#               viewing functions in one object.
#
# =========================================================================== #
class BaseProblem( object ):
  def __init__(self, params ):
    self.params = params
    self.load_params( params )
    self.initialized = False
  
  # extract info about specific for this problem
  def load_params( self, params ):
    raise NotImplementedError  
    
  # "create" problem or load observations  
  def initialize( self ):
    raise NotImplementedError
    
  # take samples/staistics etc and "view" this particular problem
  def view_results( self, states_object ):
    raise NotImplementedError
    
  def get_observations( self ):
    raise NotImplementedError
    
  def get_obs_statistics( self ):
    raise NotImplementedError
    
  # run simulation at parameter setting x, return outputs
  def simulation_function( self, x ):
    raise NotImplementedError
    
  # pass outputs through statistics function, return statistics
  def statistics_function( self, outputs ):
    raise NotImplementedError
    
  # return size of statistics vector for this problem
  def get_nbr_statistics( self ):
    raise NotImplementedError