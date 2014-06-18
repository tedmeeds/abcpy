from abcpy.helpers import gaussian_logpdf, mvn_diagonal_logpdf, mvn_diagonal_logcdf, mvn_diagonal_logcdfcomplement
from abcpy.surrogates.surrogate import BaseSurrogate
from progapy.gps.product_gaussian_process import ProductGaussianProcess
from progapy.gps.basic_regression import BasicRegressionGaussianProcess

import numpy as np
import pylab as pp
import pdb

class GaussianProcessSurrogate(BaseSurrogate): 

  def is_empty(self):
    if self.gp.N == 0:
      return True
    else:
      return False
      
  def load_params( self, params ):
    self.gp             = params["gp"]
    
    if params.has_key("prior_std"):
      self.prior_std = params["prior_std"]
    else:
      self.prior_std = 0.0
  
  def add( self, thetas, pseudo_statistics ):
    if len(thetas.shape) == 1:
      D = len(thetas)
      S1 = 1
    else:
      S1,D = thetas.shape
    S,J = pseudo_statistics.shape
    
    if S1 != S:
      assert S1 == 1, "if there are many stats, make sure only one theta"
      THETAS = np.array( [thetas.copy() for s in range(S)] )
    else:
      THETAS = thetas.reshape( (S,D) )
      
    #pdb.set_trace()
    self.gp.add_data( THETAS, pseudo_statistics.reshape((S,J)) )  
    
  def posterior_at( self, theta ):
    mus, mu_covs, mu_covs_plus_noise = self.gp.full_posterior_mean_and_data( theta.reshape((1,len(theta))) )
    
    # this could be all redundent, but try and get into correct form
    J = len(mus)
    mu                = np.zeros( J )
    mu_cov            = np.zeros( (J,J) )
    mu_cov_plus_noise = np.zeros( (J,J) )
    for j in range(J):
      mu[j]                  = mus[j]
      mu_cov[j,j]            = np.diag( mu_covs[j] )
      mu_cov_plus_noise[j,j] = np.diag( mu_covs_plus_noise[j] )

    return  mu, mu_cov, mu_cov_plus_noise
    
  def logpdf( self, theta, observations, prior_std = 0.0, response_is_loglikelihood = False ):
    # use the expected mean and the full uncertainty to compute log-likelihood
    mu, mu_cov, mu_cov_plus_noise = self.posterior_at( theta )
    
    if response_is_loglikelihood:
      return np.sum(mu)
    else:
      stddevs = np.diag( mu_cov_plus_noise ) + self.prior_std
      logpdf_across_J = mvn_diagonal_logpdf( observations, mu, stddevs )
      return np.sum(logpdf_across_J)
      
  def logcdf( self, theta, observations, response_is_loglikelihood = False ):
    # use the expected mean and the full uncertainty to compute log-likelihood
    mu, mu_cov, mu_cov_plus_noise = self.posterior_at( theta )
    
    stddevs = np.diag( mu_cov_plus_noise ) + self.prior_std
    logpdf_across_J = mvn_diagonal_logcdf( observations, mu, stddevs )
    return np.sum(logpdf_across_J)
    
  def logcdfcomplement( self, theta, observations, response_is_loglikelihood = False ):
    # use the expected mean and the full uncertainty to compute log-likelihood
    mu, mu_cov, mu_cov_plus_noise = self.posterior_at( theta )
    
    stddevs = np.diag( mu_cov_plus_noise ) + self.prior_std
    logpdf_across_J = mvn_diagonal_logcdfcomplement( observations, mu, stddevs )
    return np.sum(logpdf_across_J)
      
  def logpdf_rand( self, theta, observations, N = 1, prior_std = 0.0, response_is_loglikelihood = False ):
    # use the expected mean and the full uncertainty to compute log-likelihood
    mu, mu_cov, mu_cov_plus_noise = self.posterior_at( theta )
    
    means   = np.random.multivariate_normal( mu, mu_cov, N )
    
    if response_is_loglikelihood:
      return means
    else:
      stddevs = np.diag( mu_cov_plus_noise - mu_cov ) + self.prior_std
    
      logliks = np.zeros( N )
      for n in range(N):
        logliks[n] = np.sum( mvn_diagonal_logpdf( observations, means[n], stddevs ) )
      return logliks
    
  def logcdf_rand( self, theta, observations, N = 1, response_is_loglikelihood = False ):
    # use the expected mean and the full uncertainty to compute log-likelihood
    mu, mu_cov, mu_cov_plus_noise = self.posterior_at( theta )
    
    means   = np.random.multivariate_normal( mu, mu_cov, N )
    stddevs = np.diag( mu_cov_plus_noise - mu_cov ) + self.prior_std
    
    logliks = np.zeros( N )
    for n in range(N):
      logliks[n] = np.sum( mvn_diagonal_logcdf( observations, means[n], stddevs ) )
    return logliks
    
  def logcdfcomplement_rand( self, theta, observations, N = 1, response_is_loglikelihood = False ):
    # use the expected mean and the full uncertainty to compute log-likelihood
    mu, mu_cov, mu_cov_plus_noise = self.posterior_at( theta )
    
    means   = np.random.multivariate_normal( mu, mu_cov, N )
    stddevs = np.diag( mu_cov_plus_noise - mu_cov ) + self.prior_std
    
    logliks = np.zeros( N )
    for n in range(N):
      logliks[n] = np.sum( mvn_diagonal_logcdfcomplement( observations, means[n], stddevs ) )
    return logliks
  #   
  # def loglikelihood( self, theta ):
  #   log_p = 0
  #   #Xtest = np.hstack( theta ).reshape( (1, len(theta) ))
  #   mus, mu_covs, data_covs = self.gp.full_posterior_mean_and_data( theta.reshape((1,len(theta))) )
  #   for j in range(len(self.obs_statistics)):
  #     y = self.obs_statistics[j]
  #     #data_stds = np.sqrt(np.diag( data_covs[j] ))
  #     data_stds = np.sqrt( np.diag( data_covs[j] - mu_covs[j] ) )
  #     log_p += np.sum( np.squeeze(gaussian_logpdf( y, mus, data_stds+self.epsilon )))
  #   return log_p
   
  def update(self):
    if np.random.rand()<0.25:
      self.gp.train()
    #pass
    #self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )  
    #self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )
    #print "UPDATING SURROGATE!!!!!"
    if self.gp.N < 100:
      p = self.gp.kernel.get_params()
      if np.random.rand()<0.25:
        self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":2} )
        self.gp.kernel.set_params(p)
    #thetas = self.gp.sample( method = "slice", params = {"nbrSteps":3,"N":2,"MODE":2,"set_to_last_sample":True}) 
    #self.gp.set_params( thetas[-1])
    
  def update_post_mh( self, observation_group, simulation_statistics, params ):
    if observation_group.params.has_key("update_ystar_down"):
      # for each training input, compute means, find min
      old_ystar = observation_group.ystar[0]
      mus, mu_covs, mu_covs_plus_noise = self.gp.full_posterior_mean_and_data( self.gp.X )
      ystar = np.min( mus )
      observation_group.ystar = np.array([ystar])
      observation_group.observation_statistics = np.array([ystar])
      print "changed ystar from ", old_ystar, ' to ', ystar
      
    elif observation_group.params.has_key("update_ystar_up"):
      # for each training input, compute means, find max
      old_ystar = observation_group.ystar[0]
      mus, mu_covs, mu_covs_plus_noise = self.gp.full_posterior_mean_and_data( self.gp.X )
      ystar = np.max( mus )
      observation_group.ystar = np.array([ystar])
      observation_group.observation_statistics = np.array([ystar])
      print "changed ystar from ", old_ystar, ' to ', ystar
    
    if params is not None and params.has_key("length_scale_update"):
      p = self.gp.kernel.get_params()
      alpha = 0.9999 #(self.gp.N - 1.0 ) / self.gp.N
      p[1:] *= alpha
      self.gp.kernel.set_params(p)
      print "changing kernel length scale"
      