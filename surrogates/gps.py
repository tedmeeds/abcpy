from abcpy.helpers import gaussian_logpdf
from abcpy.surrogates.surrogate import BaseSurrogate
from progapy.gps.product_gaussian_process import ProductGaussianProcess
from progapy.gps.basic_regression import BasicRegressionGaussianProcess

import numpy as np
import pylab as pp
import pdb

class GaussianProcessSurrogate(BaseSurrogate): 

  def init_with_params( self, params ):
    super(GaussianProcessSurrogate, self).init_with_params(params)
    self.gp             = params["gp"]
    self.obs_statistics = params["obs_statistics"]
    self.update_rate    = params["update_rate"]
    self.since_last_update    = 0
    
  def loglik_differences_rand( self, to_theta, from_theta, M, force_update = False ):
    
    if self.since_last_update >= self.update_rate or force_update is True:
      self.update()
      self.since_last_update = 0
      
    D = len(to_theta)
    
    # make Xtest matrix
    Xtest = np.hstack( (to_theta, from_theta) ).reshape( (2, D ))
    
    # compute gp posterior at test points
    mus, mu_covs, data_covs = self.gp.full_posterior_mean_and_data( Xtest )
    
    # J := nbr gps (one for each statistic), N:= nbr locations (always 2), D:= nbr outputs per gp (since independent, this should be 1 always)
    J,N,Ny = mus.shape
    
    # some error checking we can remove later
    assert J == len(mu_covs), "should be same"
    assert J == len(data_covs), "should be same"
    assert N == mu_covs.shape[1], "should be same"
    assert N == mu_covs.shape[2], "should be same"
    assert N == data_covs.shape[1], "should be same"
    assert N == data_covs.shape[2], "should be same"
    assert Ny == 1, "independence, please"
    
    # init to zeros
    loglik_difs = np.zeros( M )
    
    # for now, we assume independence across J outputs
    for j in range(J):
      y = self.obs_statistics[j]
      data_stds = np.sqrt(np.diag( data_covs[j] ))
      mu_var = np.diag( mu_covs[j] )
      mu_c = np.diag( mu_var )
      #mu_c = mu_covs[j]
      
      # samples means, a M by 2 matrix, first column are proposal means, second current means
      #means = np.random.multivariate_normal( np.squeeze(mus[j]), mu_covs[j], M )
      means = np.random.multivariate_normal( np.squeeze(mus[j]), mu_c, M )
      
      proposal_logliks = gaussian_logpdf( y, means[:,0], data_stds[0] )
      current_logliks  = gaussian_logpdf( y, means[:,1], data_stds[1] )
      loglik_difs +=  proposal_logliks - current_logliks

    if self.params.has_key("pause"):  
      print mus, mu_covs, data_covs
      pdb.set_trace()
    return loglik_difs
    
        
  def acquire_points( self, to_theta, from_theta, M, force_update = False ):
    print "Acquiring %d points"%(M)
    for m in range(M):
      if np.random.rand() < 0.5:
        thetas, sim_outs, stats = self.run_sim_and_stats( to_theta, 1 )
      else:
        thetas, sim_outs, stats = self.run_sim_and_stats( from_theta, 1 )
      self.nbr_sim_calls_this_iter += 1
      self.since_last_update += 1
      #print "N before: ", self.gp.N
      self.gp.add_data( thetas, stats )  
      print "   gp now has N: ", self.gp.N
      
      
   
  def update(self):
    #self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )  
    #self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )
    thetas = self.gp.sample( method = "slice", params = {"nbrSteps":3,"N":2,"MODE":2,"set_to_last_sample":True}) 
    #gp.set_params( thetas[-1])