from abcpy.helpers import gaussian_logpdf, mvn_diagonal_logpdf, mvn_diagonal_logcdf
from abcpy.surrogates.surrogate import BaseSurrogate
from progapy.gps.product_gaussian_process import ProductGaussianProcess
from progapy.gps.basic_regression import BasicRegressionGaussianProcess

import numpy as np
import pylab as pp
import pdb

class GaussianProcessSurrogate(BaseSurrogate): 

  def load_params( self, params ):
    self.gp             = params["gp"]
  
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
      THETAS = thetas.reshape( (D,S) )
      
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
    
  def logpdf( self, theta, observations, prior_std = 0.0 ):
    # use the expected mean and the full uncertainty to compute log-likelihood
    mu, mu_cov, mu_cov_plus_noise = self.posterior_at( theta )
    
    stddevs = np.diag( mu_cov_plus_noise ) + prior_std
    logpdf_across_J = mvn_diagonal_logpdf( observations, mu, stddevs )
    return np.sum(logpdf_across_J)
      
  def logcdf( self, theta, observations ):
    # use the expected mean and the full uncertainty to compute log-likelihood
    mu, mu_cov, mu_cov_plus_noise = self.posterior_at( theta )
    
    stddevs = np.diag( mu_cov_plus_noise ) + prior_std
    logpdf_across_J = mvn_diagonal_logcdf( observations, mu, stddevs )
    return np.sum(logpdf_across_J)
      
  def logpdf_rand( self, theta, observations, N = 1, prior_std = 0.0 ):
    # use the expected mean and the full uncertainty to compute log-likelihood
    mu, mu_cov, mu_cov_plus_noise = self.posterior_at( theta )
    
    means   = np.random.multivariate_normal( mu, mu_cov, N )
    stddevs = np.diag( mu_cov_plus_noise - mu_cov ) + prior_std
    
    logliks = np.zeros( N )
    for n in range(N):
      logliks[n] = np.sum( mvn_diagonal_logpdf( observations, means[n], stddevs ) )
    return logliks
    
  def logcdf_rand( self, theta, observations, N = 1 ):
    # use the expected mean and the full uncertainty to compute log-likelihood
    mu, mu_cov, mu_cov_plus_noise = self.posterior_at( theta )
    
    means   = np.random.multivariate_normal( mu, mu_cov, N )
    stddevs = np.diag( mu_cov_plus_noise - mu_cov ) + prior_std
    
    logliks = np.zeros( N )
    for n in range(N):
      logliks[n] = np.sum( mvn_diagonal_logcdf( observations, means[n], stddevs ) )
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
    pass
    #self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )  
    #self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )
    #print "UPDATING SURROGATE!!!!!"
    #if self.gp.N < 100:
    #  self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":2} )
    #thetas = self.gp.sample( method = "slice", params = {"nbrSteps":3,"N":2,"MODE":2,"set_to_last_sample":True}) 
    #self.gp.set_params( thetas[-1])