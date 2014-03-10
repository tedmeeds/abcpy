from abcpy.state import BaseState
from abcpy.helpers import logsumexp

import numpy as np
import scipy as sp
import pylab as pp

# =========================================================================== #
#
#  KernelEpsilonState: computes likelihood based on average of S kernel calls.
#                      used for kernel-abc-mcmc, maybe others...
#
# =========================================================================== #
class KernelEpsilonState( BaseState ):
  
  def new( self, theta, params ):
    return KernelEpsilonState( theta, params )
    
  def load_params( self, params ):
    # prior and proposal distribution functions
    self.theta_prior_rand_func      = params["theta_prior_rand_func"]
    self.theta_prior_logpdf_func    = params["theta_prior_logpdf_func"]
    self.theta_proposal_rand_func   = params["theta_proposal_rand_func"]
    self.theta_proposal_logpdf_func = params["theta_proposal_logpdf_func"]
    
    # observations, simulator, statistic functions
    self.obs_statistics        = params["obs_statistics"]
    self.simulation_function   = params["simulation_function"]
    self.statistics_function   = params["statistics_function"]
    
    # kernel and its epsilon
    self.log_kernel_func  = params["log_kernel_func"]
    self.epsilon          = params["epsilon"]
    
    # params specific to loglikelihood ans state updates
    self.S                = params["S"]
    self.is_marginal      = params["is_marginal"]
    
    # 
    self.loglik           = None
    
  def loglikelihood( self, theta = None ):
    if (theta is None) and (self.loglik is not None):
      return self.loglik
    
    # note we are changing the state here, not merely calling a function
    if theta is not None:
      self.theta = theta
    
    # approximate likelihood by average over S simulations    
    self.loglikelihoods = np.zeros(self.S)
    self.sim_outs       = []
    self.stats          = []
    for s in range(self.S):
      # simulation -> outputs -> statistics -> loglikelihood
      self.sim_outs.append( self.simulation_function( self.theta ) ); self.add_sim_call()
      self.stats.append( self.statistics_function( self.sim_outs[-1] ) )
      self.loglikelihoods[s] = self.log_kernel_func( self.stats[-1], self.obs_statistics, self.epsilon )
    self.stats      = np.array(self.stats)
    self.statistics = np.mean( self.stats, 0 )
    self.loglik     = logsumexp( self.loglikelihoods ) - np.log(self.S)
    
    return self.loglik
  
  def prior_rand(self):
    return self.theta_prior_rand_func()
    
  def logprior( self, theta = None ):
    if theta is None:
      return self.theta_prior_logpdf_func( self.theta )
    else:
      return self.theta_prior_logpdf_func( theta )
    
  def proposal_rand( self, from_theta ):
    return self.theta_proposal_rand_func( from_theta )
        
  def logproposal( self, to_theta, from_theta ):
    return self.theta_proposal_logpdf_func( to_theta, from_theta )
    
  def get_statistics(self):
    return self.statistics
     