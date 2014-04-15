from abcpy.response_model import SimulationResponseModel
from abcpy.helpers import log_pdf_full_mvn, log_pdf_diag_mvn, normcdf

import numpy as np
import pdb

def mvn_logpdf( X, mu, cov, invcov = None, logdet = None ):
  return log_pdf_full_mvn( X, mu, cov, invcov, logdet )
  
def mvn_diagonal_logpdf( X, mu, stddevs ):
  return log_pdf_diag_mvn( X, mu, stddevs )
  
def mvn_diagonal_logcdf( X, mu, stddevs ):
  logpdf = 0.0
  #pdb.set_trace()
  for x,mu,std in zip( X.T, mu, stddevs ):
    cdf = np.squeeze( normcdf( x, mu, std ) )
    if cdf ==0:
      logpdf += np.log(1e-12)
    else:
      logpdf += np.log( cdf )
  return logpdf
  
  #return log_pdf_diag_mvn( X, mu, stddevs )
  
class GaussianResponseModel( SimulationResponseModel ):
  
  def load_params( self, params ):
    self.likelihood_type = "logpdf"  # by default the likelihood is the density of observations under gaussian density
    self.diagonalize     = False     # by default, use full covariance
    self.epsilon         = 0.0
    
    # check for non-default parameters
    if params.has_key("diagonalize"):
      self.diagonalize = params["diagonalize"]
      
    if params.has_key("likelihood_type"):
      self.likelihood_type = params["likelihood_type"]
      
    if params.has_key("epsilon"):
      self.epsilon = params["epsilon"]
   
  def new( self, params ):   
    m = GaussianResponseModel( params )
    return m
    
  def response_model_rand( self ):
    raise NotImplementedError
    
  def update( self, thetas, pseudo_statistics, observation_statistics ):
    # ignore thetas, observation_statistics
    
    # compute mean, stats_cov, mean_cov
    S,J = pseudo_statistics.shape
    assert S > 1, "must have at least 2 simulations"
    
    self.pstats_mean     = pseudo_statistics.mean(0)
    d = pseudo_statistics - self.pstats_mean 
    self.pstats_sumsq    = np.dot( d.T, d )
    self.pstats_cov      = self.pstats_sumsq / (S-1) + self.epsilon*np.eye(J)
    self.pstats_mean_cov = self.pstats_cov / S
    self.pstats_icov     = np.linalg.inv( self.pstats_cov )
    self.pstats_logdet   = -np.log( np.linalg.det( self.pstats_cov ) )  # neg cause of mvn parameterization
    
    # TODO: other options are to put priors
    
    # zero-out the off-diagonals;  make the statistics independent.
    if self.diagonalize:
      self.pstats_stddevs      = np.sqrt( np.diag( self.pstats_cov) )
      self.pstats_mean_stddevs = np.sqrt( np.diag( self.pstats_mean_cov) )
    
  def loglikelihood( self, observations ):
    if self.likelihood_type == "logpdf":
      if self.diagonalize:
        return mvn_diagonal_logpdf( observations, self.pstats_mean, self.pstats_stddevs )
      else:
        return mvn_logpdf( observations, self.pstats_mean, self.pstats_cov, self.pstats_icov, self.pstats_logdet )
    elif self.likelihood_type == "logcdf":
      if self.diagonalize:
        return mvn_diagonal_logcdf( observations, self.pstats_mean, self.pstats_stddevs )
      else:
        return mvn_logcdf( observations, self.pstats_mean, self.pstats_cov )
    else:
      raise NotImplementedError
    
  def loglikelihood_rand( self, observations, N = 1 ):
    random_logliks = np.zeros( N )
    J = J = len(self.pstats_mean)
    if self.likelihood_type == "logpdf":
      if self.diagonalize:
        means = self.pstats_mean + self.pstats_mean_stddevs*np.random.randn( N, J )
        for n in range(N):
          loglike_n = mvn_diagonal_logpdf( observations, means[n], self.pstats_stddevs )
          random_logliks[n] = loglike_n.sum() # sum over observations
      else:
        means = np.random.multivariate_normal( self.pstats_mean, self.pstats_mean_cov, N )
        for n in range(N):
          loglike_n = mvn_logpdf( observations, means[n], self.pstats_cov, self.pstats_icov, self.pstats_logdet )
          random_logliks[n] = loglike_n.sum() # sum over observations
    elif self.likelihood_type == "logcdf":
      if self.diagonalize:
        means = np.random.multivariate_normal( self.pstats_mean, self.pstats_mean_cov, N )
        for n in range(N):
          loglike_n = mvn_diagonal_logcdf( observations, means[n], self.pstats_stddevs )
          random_logliks[n] = loglike_n.sum() # sum over observations
      else:
        means = self.pstats_mean + self.pstats_mean_stddevs*np.random.randn( N, J )
        for n in range(N):
          loglike_n = mvn_logcdf( observations, means[n], self.pstats_cov, self.pstats_icov, self.pstats_logdet )
          random_logliks[n] = loglike_n.sum() # sum over observations
    else:
      raise NotImplementedError
      
    return random_logliks