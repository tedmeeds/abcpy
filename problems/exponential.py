from abcpy.problem import BaseProblem
from abcpy.plotting import *
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
    self.obs_statistics = self.statistics_function( self.observations )
    
    self.posterior_mode = (self.observations.sum() + self.alpha)/(self.N + self.alpha + self.beta)
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
    return np.random.exponential( 1.0/theta, self.N ) # 1/theta because of how python does exponential draws
    
  # pass outputs through statistics function, return statistics
  def statistics_function( self, outputs ):
    return np.array( [np.mean( outputs )] )
    
  # return size of statistics vector for this problem
  def get_nbr_statistics( self ):
    return 1
  
  # theta_rand
  def theta_prior_rand( self, N = 1 ):
    return np.random.gamma( self.alpha, 1.0/self.beta, N ) # 1/beta cause of how python implements
        
  # take samples/staistics etc and "view" this particular problem
  def view_results( self, states_object, burnin = 1 ):
    # plotting params
    nbins       = 20
    alpha       = 0.5
    label_size  = 8
    linewidth   = 3
    linecolor   = "r"
    
    # extract from states
    thetas = states_object.get_thetas(burnin=burnin)
    stats  = states_object.get_statistics(burnin=burnin)
    
    # plot sample distribution of thetas, add vertical line for true theta, theta_star
    f = pp.figure()
    sp = f.add_subplot(111)
    pp.hist( thetas, nbins, normed = True, alpha = alpha )
    ax = pp.axis()
    pp.vlines( thetas.mean(), ax[2], ax[3], color="b", linewidths=linewidth)
    #pp.vlines( self.theta_star, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    pp.vlines( 1.0/self.posterior_mode, ax[2], ax[3], color="g", linewidths=linewidth )
    pp.xlabel( "theta" )
    pp.ylabel( "P(theta)" )
    pp.axis(ax)
    set_label_fonsize( sp, label_size )
    
    # return handle to figure for further manipulation
    return f