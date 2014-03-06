from abcpy.problem import BaseProblem
import numpy as np
import scipy as sp
import pylab as pp

class ExponentialProblem( BaseProblem ):
  # extract info about specific for this problem
  def load_params( self, params ):
    # Gamma prior parameters
    self.alpha = params["alpha"]
    self.beta  = params["beta"]
    
    # "true" parameter setting
    self.theta_star = params["theta_star"]
    
    # number of samples per simulation
    self.N = params["N"]
    
    # random seed for observations
    self.seed = None
    if params.has_key("seed"):
      self.seed = params["seed"]  
    
  # "create" problem or load observations  
  def initialize( self ):
    assert self.initialized is False, "Ensure we only call this once..."
    
    # set random seed (to reproduce results)
    if self.seed is not None:
      np.random.randn( self.seed )
    
    # generate observations and statistics
    self.observations   = self.simulation_function( self.theta_star )
    self.obs_statistics = self.statistics_function( self.observation )
    
    # done initialization
    self.initialized = True
    
  def get_observations( self ):
    assert self.initialized, "Not initialized..."
    return self.observations
    
  def get_obs_statistics( self ):
    assert self.initialized, "Not initialized..."
    return self.obs_statistics
      
  # run simulation at parameter setting theta, return outputs
  def simulation_function( self, theta ):
    return np.random.exponential( theta, self.N )
    
  # pass outputs through statistics function, return statistics
  def statistics_function( self, outputs ):
    return np.array( [np.mean( outputs )] )
    
  # return size of statistics vector for this problem
  def get_nbr_statistics( self ):
    return 1
      
  # take samples/staistics etc and "view" this particular problem
  def view_results( self, states_object ):
    raise NotImplementedError