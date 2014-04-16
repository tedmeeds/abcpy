from abcpy.response_model import SimulationResponseModel
from abcpy.helpers import mvn_logpdf, mvn_diagonal_logpdf, mvn_diagonal_logcdf

import numpy as np
import pdb
  
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
    
  def is_empty( self ):
    if len(self.pseudo_statistics) == 0:
      return True
    else:
      return False
      
  def add( self, thetas, pseudo_statistics, observation_statistics ):
    if len( self.pseudo_statistics ) == 0:
      self.pseudo_statistics = pseudo_statistics.copy()
      self.thetas = thetas.copy()
    else:
      self.pseudo_statistics = np.vstack( (self.pseudo_statistics, pseudo_statistics) )
      self.thetas = np.vstack( (self.thetas, thetas) )
      
    self.update()
       
  def update( self ):
    self.make_estimators( )
    
  def make_estimators( self, theta = None ):
    pseudo_statistics = self.pseudo_statistics
    
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
    
  def loglikelihood( self, theta, observations ):
    self.make_estimators( theta )
    
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
    
  def loglikelihood_rand( self, theta, observations, N = 1 ):
    self.make_estimators( theta )
    
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