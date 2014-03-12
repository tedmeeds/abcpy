from abcpy.problem import BaseProblem
from abcpy.plotting import *
from abcpy.helpers import *
import numpy as np
import scipy as sp
import pylab as pp

import pdb

class BlowflyProblem( BaseProblem ):
  # extract info about specific for this problem
  def load_params( self, params ):
    # which blowfly data are we using
    self.blowfly_filename = params["blowfly_filename"]
    #self.blowfly_index    = params["blowfly_index"]
    
    # prior parameters (Gamma distribution)
    #self.alpha = params["alpha"]
    #self.beta  = params["beta"]
    
    # proposal params (LogNormal)
    #if params.has_key("q_stddev"):
    #  self.proposal_std    = params["q_stddev"]
    #self.proposal_rand   = lognormal_rand
    #self.proposal_logpdf = lognormal_logpdf
    #self.proposal_rand   = positive_normal_rand
    #self.proposal_logpdf = normal_logpdf
    
    # "true" parameter setting
    #self.theta_star = params["theta_star"]
    
    # number of samples per simulation
    #self.N = params["N"]
    
  # "create" problem or load observations  
  def initialize( self ):
    assert self.initialized is False, "Ensure we only call this once..."
    
    # set random seed (to reproduce results)
    # if self.seed is not None:
    #   np.random.randn( self.seed )
    
    # load observations and generate its statistics
    self.observations   = np.loadtxt( self.blowfly_filename )[:,-1] # last column has values
    self.obs_statistics = self.statistics_function( self.observations )
    
    self.T = len(self.observations)
    # 
    # self.min_range           = 0.05
    # self.max_range           = 0.15
    # self.range               = (self.min_range,self.max_range)
    # self.fine_bin_width      = 0.0001
    # self.coarse_bin_width    = 0.001
    # 
    # self.fine_theta_range    = np.arange( self.min_range, self.max_range+self.fine_bin_width, self.fine_bin_width )
    # self.coarse_theta_range  = np.arange( self.min_range, self.max_range+self.coarse_bin_width, self.coarse_bin_width )
    # 
    # self.nbins_coarse   = len(self.coarse_theta_range)
    # self.nbins_fine     = len(self.fine_theta_range)
    # self.log_posterior  = gamma_logprob( self.fine_theta_range, self.alpha+self.N, self.beta+self.observations.sum() )
    # self.posterior      = np.exp(self.log_posterior)
    # self.posterior_mode = (self.N + self.alpha)/(self.observations.sum() + self.beta)
    # 
    # self.true_posterior_logpdf_func = gen_gamma_logpdf(self.alpha+self.N,self.beta+self.observations.sum())
    # self.true_posterior_cdf_func    = gen_gamma_cdf(self.alpha+self.N,self.beta+self.observations.sum())
    # 
    # self.posterior_bars_range = self.coarse_theta_range[:-1] + 0.5*self.coarse_bin_width
    # self.posterior_cdf        = self.true_posterior_cdf_func( self.coarse_theta_range )
    # self.posterior_bars       = (self.posterior_cdf[1:] - self.posterior_cdf[:-1])/self.coarse_bin_width
    # self.posterior_cdf_bins   = self.posterior_cdf[1:] - self.posterior_cdf[:-1]
    # done initialization
    self.initialized = True
    
  def get_observations( self ):
    assert self.initialized, "Not initialized..."
    return self.observations
    
  def get_obs_statistics( self ):
    assert self.initialized, "Not initialized..."
    return self.obs_statistics
      
  # run simulation at parameter setting theta, return outputs
  def simulation_function( self, theta ):
    
    N0, P, tau, sigma_p, sigma_d, delta = theta
  
    T = self.T
  
    var_d  = sigma_d**2
    prec_d = 1.0 / var_d

    var_p  = sigma_p**2
    prec_p = 1.0 / var_p

    burnin = 50
    lag = int(np.floor(tau))
    if (float(tau)-float(lag)>0.5):
      lag = lag + 1

    N = np.zeros( lag+burnin+T, dtype=float)
    #print N0
    N[0] = N0

    for i in range(lag):
      N[i] = 180.0

    for i in xrange(burnin+T):
      t = i + lag

      eps_t = gamma_rnd( prec_d, prec_d )
      e_t   = gamma_rnd( prec_p, prec_p )

      #tau_t = max(0,t-int(tau))
      tau_t = t - lag
      N[t] = P*N[tau_t]*np.exp(-N[tau_t]/N0)*e_t + N[t-1]*np.exp(-delta*eps_t)
  
    return N[-(T+1):]

  # pass outputs through statistics function, return statistics
  def statistics_function( self, outputs ):
    nstats = 4
    s = np.zeros( nstats, dtype = float )
    s[0] = outputs.mean() / 1000.0
    s[1] = (s[0] - np.median(outputs))/ 1000.0
    mx,mn = peakdet(outputs/outputs.std(), 1.5 )
    s[2] = float(len(mx))
    s[3] = np.log(np.max(outputs+1)/1000.0)
    return s
    # def blowstats( N ):
    #       maxlags = 11
    #   
    #       lags = 2*np.array( [6,6,6,1,1] )
    #       pows = np.array( [1,2,3,1,2] )
    #       n_reg_coeffs = len(lags)
    #   
    #       nstats = 2 + maxlags + n_reg_coeffs + 1
    #       #nstats -= maxlags
    #       nstats = 4
    #       s = np.zeros( nstats, dtype = float )
    #   
    #       s[0] = N.mean() / 1000.0
    #       s[1] = (s[0] - np.median(N))/ 1000.0
    #   
    #       mx,mn = peakdet(N/N.std(), 1.5 )
    #       #mx,mn = peakdet( N, 20000.0 )
    #       #pdb.set_trace()
    #       s[2] = float(len(mx))
    #       #lag,ac,dum1,dum2 = pp.acorr( N/1000.0, normed=True, maxlags=maxlags)
    #   
    #       #s[2:2+maxlags] = ac[:maxlags]
    #   
    #       #reg_coeffs = compute_regression( N/1000.0, lags, pows )
    #   
    #       #d1 = np.diff( N/1000.0 )
    #       #d2 = np.diff( d1 )
    #   
    #       #s[2] = np.mean(d1)
    #       #s[3] = np.mean(d2)
    #       #pdb.set_trace()
    #       #s[2:2+5] = reg_coeffs
    #       #s[2+maxlags:-1] = reg_coeffs
    #   
    #       #s[-2] = float( len(pp.find( np.abs(np.diff(np.sign(np.diff(N))))>0)) )
    #       s[-1] = np.log(np.max(N+1)/1000.0)
    #       #s[-2] = np.log(np.min(N+1)/1000.0)
    #       #s[-2] = np.max(np.abs( N[5:]-N[:-5] ))/1000.0
    #       #print "ADD turning points"
    #   
    #       return s
    return np.array( [np.mean( outputs )] )
    
  # return size of statistics vector for this problem
  def get_nbr_statistics( self ):
    return 1
  
  # theta_rand
  def theta_prior_rand( self, N = 1 ):
    return np.random.gamma( self.alpha, 1.0/self.beta, N ) # 1/beta cause of how python implements
    
  # theta_rand
  def theta_prior_logpdf( self, theta ):
    return gamma_logprob( theta, self.alpha, self.beta ) # 1/beta cause of how python implements
      
  def theta_proposal_rand( self, theta ):
    return self.proposal_rand( theta, self.proposal_std )
    
  def theta_proposal_logpdf( self, to_theta, from_theta ):
    return self.proposal_logpdf( to_theta, from_theta, self.proposal_std )
    #return self.proposal_logpdf( to_theta, np.log(from_theta), self.proposal_std )
      
  # take samples/staistics etc and "view" this particular problem
  def view_results( self, states_object, burnin = 1 ):
    # plotting params
    nbins       = 20
    alpha       = 0.5
    label_size  = 8
    linewidth   = 3
    linecolor   = "r"
    
    # extract from states
    thetas = states_object.get_thetas(burnin=burnin)
    stats  = states_object.get_statistics(burnin=burnin)
    nsims  = states_object.get_sim_calls(burnin=burnin)
    
if __name__ == "__main__":
  N0      = 3721.0 #np.exp(6.0)
  sigma_p = 0.1 #np.exp(-0.5)
  sigma_d = 0.1 #np.exp(-0.75) # smoothness
  tau     = 15.0
  P       = 12.0 #np.exp(2.0)
  delta   = 0.9# np.exp(-1.8)
  Nr      = 5
  
  N0      = 450.0 #np.exp(6.0)
  sigma_p = 1.5#np.exp(-0.5)
  sigma_d = 0.5 #np.exp(-0.75) # smoothness
  tau     = 15.0
  P       = 2.25 #np.exp(2.0)
  delta   = 0.24# np.exp(-1.8)
  
  theta_test = np.array( [N0, P, tau, sigma_p, sigma_d, delta] )
  params = {}
  params["blowfly_filename"] = "./problems/blowfly/bf1.txt"
  b = BlowflyProblem( params, force_init = True )
  test_obs = b.simulation_function( theta_test )
  pp.figure(1)
  pp.clf()
  pp.plot( b.observations/1000.0 )
  pp.plot( test_obs / 1000.0)
  
  pp.show()
  