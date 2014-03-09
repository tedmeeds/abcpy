from abcpy.state import BaseState
from abcpy.helpers import logsumexp, log_pdf_full_mvn, gaussian_logpdf

import numpy as np
import scipy as sp
import pylab as pp

import pdb

# =========================================================================== #
#
#  SyntheticLikelihoodState: also calculates estimators
#
# =========================================================================== #
class SyntheticLikelihoodModelState( BaseState ):
  
  def new( self, theta, params ):
    return SyntheticLikelihoodModelState( theta, params )
    
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
    #self.log_kernel_func  = params["log_kernel_func"]
    self.epsilon          = params["epsilon"]
    
    # params specific to loglikelihood ans state updates
    self.S                = params["S"]
    self.is_marginal      = params["is_marginal"]
    
    # 
    self.loglik           = None
  
  def run_sim_and_stats( self, nbr_points ):
    theta = self.theta
    sim_outs       = []
    stats          = []
    thetas         = []
    for s in range(nbr_points):
      # simulation -> outputs -> statistics -> loglikelihood
      sim_outs.append( self.simulation_function( theta ) ); self.nbr_sim_calls+=1
      stats.append( self.statistics_function( sim_outs[-1] ) )
      thetas.append( theta )
    sim_outs   = np.array(sim_outs)
    stats      = np.array(stats)
    thetas     = np.array(thetas)
    return thetas, sim_outs, stats
    
  def acquire( self, nbr_points ):
    thetas, sim_outs, stats = self.run_sim_and_stats( nbr_points )
  
    if len(self.stats) == 0:
      self.stats      = stats
      self.sim_ouputs = sim_outs
    else:
      self.stats      = np.vstack( (self.stats, stats) )
      self.sim_ouputs = np.vstack( (self.sim_ouputs, sim_outs) )
      
      
  def update_model( self ): 
    self.statistics   = np.mean( self.stats, 0 )
    self.mu_stats     = self.statistics
    self.cov_stats    = np.cov( self.stats.T )
    self.cov_mu_stats = self.cov_stats / self.S
    
  #    
  # def precomputes( self, theta = None ):
  #   if (theta is None) and (self.loglik is not None):
  #     return self.loglik
  #   
  #   # note we are changing the state here, not merely calling a function
  #   if theta is not None:
  #     self.theta = theta
  #   
  #   # approximate likelihood by average over S simulations    
  #   self.loglikelihoods = np.zeros(self.S)
  #   self.acquire( self.S )
  #   self.update_model()
  #   
  #   #self.loglik = log_pdf_full_mvn( self.obs_statistics, self.mu_stats, self.cov_stats )
  #   self.loglik = self.loglikelihood_under_model()
    
  def model_parameters(self):
    #self.precomputes()
    return self.mu_stats, self.cov_stats, self.cov_mu_stats, self.S
  
  def loglikelihood_rand( self, M ):
    if self.loglik is None:
      self.loglik = self.loglikelihood()
      
    logliks = np.zeros(M)
    stats = self.obs_statistics
    std_stats = self.epsilon+np.sqrt(self.cov_stats)
    std_mu_stats = np.sqrt(self.cov_mu_stats)
    for m in xrange(M):
      mu_stats = self.mu_stats + std_mu_stats*np.random.randn()
      logliks[m] = self.loglikelihood_under_model(stats, mu_stats, std_stats)
    return logliks
    
  def loglikelihood_under_model(self, stats, mu_stats, std_stats ): 
    return np.squeeze(gaussian_logpdf( stats, mu_stats, std_stats ))
      
  def loglikelihood( self, theta = None ):
    if len(self.stats) == 0:
      self.acquire( self.S )
      self.update_model()
      self.loglik = self.loglikelihood_under_model( self.obs_statistics, self.mu_stats, self.epsilon+np.sqrt(self.cov_stats) )

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
     