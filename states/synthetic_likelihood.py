from abcpy.state import BaseState
from abcpy.helpers import logsumexp, log_pdf_full_mvn, gaussian_logpdf, inv_wishart_rnd, wishart_rnd

import numpy as np
import scipy as sp
import pylab as pp

import pdb

# =========================================================================== #
#
#  SyntheticLikelihoodState: also calculates estimators
#
# =========================================================================== #
class SyntheticLikelihoodState( BaseState ):
  
  def __init__(self, theta, params ):
    super(SyntheticLikelihoodState, self).__init__(theta, params)
    self.stats    = []
    self.sim_outs = []
    
  def new( self, theta, params ):
    return SyntheticLikelihoodState( theta, params )
    
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
    
    self.hierarchy             = "just_gaussian"
    if params.has_key("hierarchy_type"):
      self.hierarchy = params["hierarchy_type"]
      
    # kernel and its epsilon
    self.epsilon          = params["epsilon"]
    
    # params specific to loglikelihood ans state updates
    self.S                = params["S"]
    self.is_marginal      = params["is_marginal"]
    
    # 
    self.loglik           = None
  
  def log_posterior( self ):
    return self.loglikelihood() + self.logprior()
    
  def run_sim_and_stats( self, theta, nbr_points ):
    #theta = self.theta
    sim_outs       = []
    stats          = []
    thetas         = []
    for s in range(nbr_points):
      # simulation -> outputs -> statistics -> loglikelihood
      sim_outs.append( self.simulation_function( theta ) ); self.add_sim_call()
      stats.append( self.statistics_function( sim_outs[-1] ) )
      thetas.append( theta )
    sim_outs   = np.array(sim_outs)
    stats      = np.array(stats)
    thetas     = np.array(thetas)
    return thetas, sim_outs, stats
    
  def acquire( self, nbr_points ):
    thetas, sim_outs, stats = self.run_sim_and_stats( self.theta, nbr_points )
  
    #pdb.set_trace()
    if len(self.stats) == 0:
      self.stats      = stats
      self.sim_ouputs = sim_outs
    else:
      self.stats      = np.vstack( (self.stats, stats) )
      self.sim_ouputs = np.vstack( (self.sim_ouputs, sim_outs) )
    
    self.update_model()
    
    return thetas, sim_outs, stats
      
      
  def update_model( self ): 
    self.statistics   = np.mean( self.stats, 0 )
    self.mu_stats     = self.statistics
    self.n_stats      = len(self.stats)
    dif = self.stats - self.mu_stats
    self.sum_sq_stats = np.dot( dif.T, dif )
    #self.cov_stats    = np.cov( self.stats.T, ddof = self.n_stats-1 )
    self.cov_stats    = self.sum_sq_stats / (self.n_stats-1 )
    self.cov_mu_stats = self.cov_stats / self.n_stats
    
  def model_parameters(self):
    #self.precomputes()
    return self.mu_stats, self.n_stats, self.sum_sq_stats, self.cov_stats, self.cov_mu_stats
  
  def loglikelihood_rand( self, M ):
    if self.loglik is None:
      self.loglik = self.loglikelihood()
      
    logliks = np.zeros(M)
    stats = self.obs_statistics
    std_stats = self.epsilon+np.sqrt(self.cov_stats)
    std_mu_stats = np.sqrt(self.cov_mu_stats)
    for m in xrange(M):
      if self.hierarchy == "jeffreys":
        cov_stats = wishart_rnd( self.n_stats-1, self.sum_sq_stats )
        std_stats = self.epsilon+np.sqrt(cov_stats)
        std_mu_stats = np.sqrt(cov_stats/self.n_stats)
        mu_stats = self.mu_stats + std_mu_stats*np.random.randn()
      elif self.hierarchy == "just_gaussian":
        mu_stats = self.mu_stats + std_mu_stats*np.random.randn()
      else:
        assert False, "no other type yet"
      logliks[m] = self.loglikelihood_under_model(stats, mu_stats, std_stats)
    return logliks
    
  def loglikelihood_under_model(self, stats, mu_stats, std_stats ): 
    return np.squeeze(gaussian_logpdf( stats, mu_stats, std_stats ))
      
  def loglikelihood( self, theta = None ):
    if len(self.stats) == 0:
      self.acquire( self.S )
      #self.update_model()
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
     