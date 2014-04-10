from abcpy.state import BaseState
from abcpy.helpers import logsumexp, log_pdf_full_mvn, gaussian_logpdf, inv_wishart_rnd, wishart_rnd, invgamma_rnd

import numpy as np
import scipy as sp
import pylab as pp

import pdb

# =========================================================================== #
#
#  ModelBasedLikelihoodState: also calculates estimators
#
# =========================================================================== #
class ModelBasedLikelihoodState( BaseState ):
  
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
    
    self.zero_out_cross_stats_terms = False
    if params.has_key("zero_cross_terms"):
      self.zero_out_cross_stats_terms = params["zero_cross_terms"]
    
    # observations, simulator, statistic functions
    self.obs_statistics        = params["obs_statistics"]
    self.simulation_function   = params["simulation_function"]
    self.statistics_function   = params["statistics_function"]
    self.nbr_stats             = len(self.obs_statistics)
    self.hierarchy             = params["hierarchy_type"] #"just_gaussian"
    if params.has_key("hierarchy_type"):
      self.hierarchy = params["hierarchy_type"]
      
    # kernel and its epsilon
    self.epsilon          = params["epsilon"]
    self.jitter           = 1e-6
    
    # params specific to loglikelihood ans state updates
    self.S                = params["S"]
    self.is_marginal      = params["is_marginal"]
      
    if self.hierarchy == "jeffreys" and self.zero_out_cross_stats_terms is False:
      assert self.S > self.nbr_stats, "we need at  S > nbr stats for wishart samples"
    # 
    self.loglik           = None
  
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
    
  # def acquire( self, nbr_points ):
  #   thetas, sim_outs, stats = self.run_sim_and_stats( self.theta, nbr_points )
  # 
  #   #pdb.set_trace()
  #   if len(self.stats) == 0:
  #     self.stats      = stats
  #     self.sim_ouputs = sim_outs
  #   else:
  #     self.stats      = np.vstack( (self.stats, stats) )
  #     self.sim_ouputs = np.vstack( (self.sim_ouputs, sim_outs) )
  #   
  #   self.update_model()
  #   
  #   return thetas, sim_outs, stats
      
      
  def update_model( self ): 
    raise NotImplementedError
# self.statistics   = np.mean( self.stats, 0 )
#     self.mu_stats     = self.statistics
#     self.n_stats      = len(self.stats)
#     dif = self.stats - self.mu_stats
#     self.sum_sq_stats = np.dot( dif.T, dif )
#     
#     if self.zero_out_cross_stats_terms:
#       s = np.diag(self.sum_sq_stats)
#       self.sum_sq_stats = s + self.jitter*np.random.rand(self.nbr_stats )
#     else:
#       self.sum_sq_stats += np.diag(self.jitter*np.random.rand(self.nbr_stats ))
#       
#     self.cov_stats    = self.sum_sq_stats / (self.n_stats-1 )
#     self.cov_mu_stats = self.cov_stats / self.n_stats
#     
#     print "  ",self.n_stats
#     #print self.mu_stats, self.cov_stats 
          
    
  def model_parameters(self):
    raise NotImplementedError
    #self.precomputes()
    #return self.mu_stats, self.n_stats, self.sum_sq_stats, self.cov_stats, self.cov_mu_stats
  
  # def loglikelihood_rand( self, M ):
  #   if self.loglik is None:
  #     self.loglik = self.loglikelihood()
  #     
  #   logliks = np.zeros(M)
  #   stats = self.obs_statistics
  #   cov_stats = self.cov_stats
  #   for m in xrange(M):
  #     logliks[m] = self.loglikelihood_under_model_rand()
  #     
  #     if self.hierarchy == "jeffreys":
  #       if self.nbr_stats == 1 or self.zero_out_cross_stats_terms:
  #         a = invgamma_rnd( float(self.n_stats-1), 0.5, len(self.sum_sq_stats) )
  #         cov_stats = float(self.n_stats-1)*self.sum_sq_stats/a
  #         
  #         if any(cov_stats<0) or any( np.isinf(cov_stats)) or any(np.isnan(cov_stats)):
  #           print a
  #           print cov_stats
  #           pdb.set_trace()
  #           
  #         cov_mu_stats = cov_stats/self.n_stats
  #         std_mu_stats = np.sqrt(cov_mu_stats)
  #         mu_stats  =  self.mu_stats + std_mu_stats*np.random.randn()
  #         #pdb.set_trace()
  #       else:
  #         cov_stats = inv_wishart_rnd( self.n_stats-1, self.sum_sq_stats )
  #         cov_mu_stats = cov_stats/self.n_stats
  # 
  #         mu_stats = np.random.multivariate_normal( self.mu_stats, cov_mu_stats )
  #       
  #       
  #     elif self.hierarchy == "just_gaussian":
  #       if self.zero_out_cross_stats_terms:
  #         std_mu_stats = np.sqrt(self.cov_mu_stats)
  #         std_stats = np.sqrt(self.cov_stats)
  #         mu_stats = self.mu_stats + (self.epsilon/self.n_stats+std_mu_stats)*np.random.randn(self.nbr_stats)
  #       else:
  #         mu_stats = np.random.multivariate_normal( self.mu_stats, self.cov_mu_stats )
  #       #mu_stats = self.mu_stats + std_mu_stats*np.random.randn()
  #     else:
  #       assert False, "no other type yet"
  #       
  #     if self.nbr_stats == 1 or self.zero_out_cross_stats_terms:
  #       logliks[m] = self.loglikelihood_under_model(self.obs_statistics, mu_stats, self.epsilon+std_stats )
  #       #pdb.set_trace()
  #     else:
  #       logliks[m] = self.mv_loglikelihood_under_model( self.obs_statistics, mu_stats, pow(self.epsilon,2)*np.eye(self.nbr_stats)+cov_stats )
  #   return logliks
  
  # def likelihood_function_rand(self):
  #   # sampler mean according to central limit theorem: mean \sim normal( pop_mean, pop_cov / S )
  #   if self.mean_randomness_type == "central_limit_full_gaussian":
  #     return self.likelihood_function_central_limit_gaussian_rand( diag = False )
  #     
  #   # for central_limit_diag_gaussian, zero-out the off diagonal terms  
  #   elif self.mean_randomness_type == "central_limit_diag_gaussian":
  #     return self.likelihood_function_central_limit_gaussian_rand( diag = True )
  #     
  #   # compute a random kernel density estimate at this parameter location
  #   elif self.mean_randomness_type == "conditional_kde":
  #     return self.likelihood_function_kernel_density_estimate_rand()
  
  def loglikelihood_under_model_rand(self):
    # generate likelihood function, pass this state so function has access to everything
    likelihood_function   = self.likelihood_function_rand( self )
    return likelihood_function.log_pdf()
  
  # def mv_loglikelihood_under_model( self, stats, mu_stats, cov_stats ):
  #   return log_pdf_full_mvn( stats, mu_stats, cov_stats )
  #   
  # def loglikelihood_under_model(self, stats, mu_stats, std_stats ): 
  #   return np.sum(np.squeeze(gaussian_logpdf( stats, mu_stats, std_stats )))
      
  def loglikelihood( self, theta = None ):
    if len(self.stats) == 0:
      self.acquire( self.S )
      #self.update_model()
      #pdb.set_trace()
      if self.nbr_stats == 1 or self.zero_out_cross_stats_terms:
        self.loglik = self.loglikelihood_under_model( self.obs_statistics, self.mu_stats, self.epsilon+np.sqrt(self.cov_stats) )
      else:
        self.loglik = self.mv_loglikelihood_under_model( self.obs_statistics, self.mu_stats, pow(self.epsilon,2)*np.eye(self.nbr_stats)+self.cov_stats )

    return self.loglik
    

     