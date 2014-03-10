from abcpy.state import BaseState

import numpy as np
import scipy as sp
import pylab as pp

# =========================================================================== #
#
#  DistanceEpsilonState: computes likelihood based on average of S kernel calls.
#                      Usage: kernel-abc-mcmc
#
# =========================================================================== #
class DistanceEpsilonState( BaseState ):
  
  def new( self, theta, params ):
    return DistanceEpsilonState( theta, params )
    
  def load_params( self, params ):

    
    self.theta_prior_rand_func = params["theta_prior_rand_func"]
    self.obs_statistics        = params["obs_statistics"]
    self.simulation_function   = params["simulation_function"]
    self.statistics_function   = params["statistics_function"]
    self.S                     = params["S"]
    self.disc                  = None
    
  def discrepancy( self, theta = None ):
    if (theta is None) and (self.disc is not None):
      return self.disc
    
    # note we are changing the state here, not merely calling a function
    if theta is not None:
      self.theta = theta
    
    # approximate likelihood by average over S simulations    
    self.discrepancies  = np.zeros(self.S)
    self.sim_outs       = []
    self.stats          = []
    
    for s in range(self.S):
      # simulation -> outputs -> statistics -> discrepancy
      self.sim_outs.append( self.simulation_function( self.theta ) ); self.add_sim_call()
      self.stats.append( self.statistics_function( self.sim_outs[-1] ) )
      self.discrepancies[s] = self.distance( self.stats[-1], self.obs_statistics )
    self.disc = np.mean( self.discrepancies )
    self.stats = np.array(self.stats)
    self.statistics = np.mean( self.stats, 0 )
    return self.disc
  
  def get_statistics(self):
    return self.statistics
     
  def distance( self, x, y ):
    return np.linalg.norm( x - y )
  
  def prior_rand(self):
    return self.theta_prior_rand_func()
    
