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
    self.epsilon        = params["epsilon"]
  
  def loglikelihood( self, theta ):
    log_p = 0
    #Xtest = np.hstack( theta ).reshape( (1, len(theta) ))
    mus, mu_covs, data_covs = self.gp.full_posterior_mean_and_data( theta.reshape((1,len(theta))) )
    for j in range(len(self.obs_statistics)):
      y = self.obs_statistics[j]
      #data_stds = np.sqrt(np.diag( data_covs[j] ))
      data_stds = np.sqrt( np.diag( data_covs[j] - mu_covs[j] ) )
      log_p += np.sum( np.squeeze(gaussian_logpdf( y, mus, data_stds+self.epsilon )))
    return log_p
      
  def loglik_differences_rand( self, to_theta, from_theta, M, force_update = False ):

      
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
      mu_stds = np.sqrt(mu_var)
      data_stds = np.sqrt( np.diag( data_covs[j] - mu_covs[j] ) )
      #mu_c = mu_covs[j]
      
      # samples means, a M by 2 matrix, first column are proposal means, second current means
      #means = np.random.multivariate_normal( np.squeeze(mus[j]), mu_covs[j], M )
      #means = np.random.multivariate_normal( np.squeeze(mus[j]), mu_c, M )
      means1 = mus[j,0] + mu_stds[0]*np.random.randn( M )
      means2 = mus[j,1] + mu_stds[1]*np.random.randn( M )
      
      proposal_logliks = gaussian_logpdf( y, means1, data_stds[0] + self.epsilon )
      current_logliks  = gaussian_logpdf( y, means2, data_stds[1] + self.epsilon )
      #proposal_logliks = gaussian_logpdf( y, means[:,0], data_stds[0] )
      #current_logliks  = gaussian_logpdf( y, means[:,1], data_stds[1] )
      loglik_difs +=  proposal_logliks - current_logliks
      #pdb.set_trace()

    if self.params.has_key("pause"):  
      print mus, mu_covs, data_covs
      pdb.set_trace()
    return loglik_difs
    
        
  def acquire_points( self, to_theta, from_theta, M, force_update = False ):
    print "Acquiring %d points"%(M)
    dif_vector = to_theta-from_theta
    a = np.random.rand()
    if a < 0.5:
      at_vector = from_theta
    else:
      at_vector = to_theta
    #at_vector = from_theta + a*dif_vector
    all_thetas = []
    all_stats  = []
    nstats = len(self.obs_statistics)
    for m in range(M):
      a = np.random.rand()
      if a < 0.5:
        at_vector = from_theta
      else:
        at_vector = to_theta
      thetas, sim_outs, stats = self.run_sim_and_stats( at_vector, 1 )
      all_thetas.append(at_vector)
      all_stats.append( np.squeeze(stats) )
      # if np.random.rand() < 0.5:
#         thetas, sim_outs, stats = self.run_sim_and_stats( to_theta, 1 )
#       else:
#         thetas, sim_outs, stats = self.run_sim_and_stats( from_theta, 1 )
      self.nbr_sim_calls_this_iter += 1
      self.since_last_update += 1
      #print "N before: ", self.gp.N
    self.gp.add_data( np.array(all_thetas), np.array(all_stats).reshape((len(all_stats),nstats)) )  
    print "   gp now has N: ", self.gp.N
    self.since_last_update = self.gp.N
    if (self.since_last_update >= self.update_rate and self.since_last_update < 1000) or force_update is True:
      self.update()
      self.since_last_update = 0
      self.update_rate *= 2
    
  def offline_simulation( self, theta ):
    thetas, sim_outs, stats = self.run_sim_and_stats( theta, 1 )
    #pdb.set_trace()
    return stats     
      
   
  def update(self):
    #self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )  
    #self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":10} )
    print "UPDATING SURROGATE!!!!!"
    self.gp.optimize( method = "minimize", params = {"maxnumlinesearch":2} )
    #thetas = self.gp.sample( method = "slice", params = {"nbrSteps":3,"N":2,"MODE":2,"set_to_last_sample":True}) 
    #self.gp.set_params( thetas[-1])