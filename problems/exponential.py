from abcpy.problem import BaseProblem
from abcpy.plotting import *
from abcpy.helpers import *
import numpy as np
import scipy as sp
import pylab as pp

import pdb

class ExponentialProblem( BaseProblem ):
  # extract info about specific for this problem
  def load_params( self, params ):
    # prior parameters (Gamma distribution)
    self.alpha = params["alpha"]
    self.beta  = params["beta"]
    
    # proposal params (LogNormal)
    if params.has_key("q_stddev"):
      self.proposal_std    = params["q_stddev"]
    self.proposal_rand   = lognormal_rand
    self.proposal_logpdf = lognormal_logpdf
    
    # "true" parameter setting
    self.theta_star = params["theta_star"]
    
    # number of samples per simulation
    self.N = params["N"]
    
    
    # random seed for observations
    self.seed = None
    if params.has_key("seed"):
      self.seed = params["seed"]  
    
  # "create" problem or load observations  
  def initialize( self ):
    assert self.initialized is False, "Ensure we only call this once..."
    
    # set random seed (to reproduce results)
    if self.seed is not None:
      np.random.randn( self.seed )
    
    # generate observations and statistics
    self.observations   = self.simulation_function( self.theta_star )
    self.obs_statistics = self.statistics_function( self.observations )
    
    self.min_range = 0.05
    self.max_range = 0.15
    self.range = (self.min_range,self.max_range)
    self.fine_bin_width      = 0.0001
    self.coarse_bin_width    = 0.001
    
    self.fine_theta_range    = np.arange( self.min_range, self.max_range+self.fine_bin_width, self.fine_bin_width )
    self.coarse_theta_range  = np.arange( self.min_range, self.max_range+self.coarse_bin_width, self.coarse_bin_width )
    
    self.nbins_coarse   = len(self.coarse_theta_range)
    self.nbins_fine     = len(self.fine_theta_range)
    self.log_posterior  = gamma_logprob( self.fine_theta_range, self.alpha+self.N, self.beta+self.observations.sum() )
    self.posterior      = np.exp(self.log_posterior)
    self.posterior_mode = (self.N + self.alpha)/(self.observations.sum() + self.beta)
    
    self.true_posterior_logpdf_func = gen_gamma_logpdf(self.alpha+self.N,self.beta+self.observations.sum())
    self.true_posterior_cdf_func = gen_gamma_cdf(self.alpha+self.N,self.beta+self.observations.sum())
    
    self.posterior_bars_range = self.coarse_theta_range[:-1] + 0.5*self.coarse_bin_width
    self.posterior_cdf = self.true_posterior_cdf_func( self.coarse_theta_range )
    self.posterior_bars = (self.posterior_cdf[1:] - self.posterior_cdf[:-1])/self.coarse_bin_width
    self.posterior_cdf_bins = self.posterior_cdf[1:] - self.posterior_cdf[:-1]
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
    return np.random.exponential( 1.0/theta, self.N ) # 1/theta because of how python does exponential draws
    
  # pass outputs through statistics function, return statistics
  def statistics_function( self, outputs ):
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
    return self.proposal_rand( np.log(theta), self.proposal_std )
    
  def theta_proposal_logpdf( self, to_theta, from_theta ):
    return self.proposal_logpdf( to_theta, np.log(from_theta), self.proposal_std )
      
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
    
    # plot sample distribution of thetas, add vertical line for true theta, theta_star
    f = pp.figure()
    sp = f.add_subplot(111)
    pp.plot( self.fine_theta_range, self.posterior, linecolor+"-", lw = 1)
    ax = pp.axis()
    pp.hist( thetas, self.nbins_coarse, range=self.range,normed = True, alpha = alpha )
    
    pp.fill_between( self.fine_theta_range, self.posterior, color="m", alpha=0.5)
    
    pp.plot( self.posterior_bars_range, self.posterior_bars, 'ro')
    pp.vlines( thetas.mean(), ax[2], ax[3], color="b", linewidths=linewidth)
    #pp.vlines( self.theta_star, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    pp.vlines( self.posterior_mode, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    
    pp.xlabel( "theta" )
    pp.ylabel( "P(theta)" )
    pp.axis([self.range[0],self.range[1],ax[2],ax[3]])
    set_label_fonsize( sp, label_size )
    
    total_sims = states_object.nbr_sim_calls
    all_sims = nsims.sum()
    at_burnin = total_sims-all_sims
    errs = []
    time_ids = []
    nbr_sims = []
    
    for time_id in [1,5,10,50,100,500,1000,2000,5000,10000,15000,20000,30000,40000,50000]:
      if time_id <= len(thetas):
        errs.append( bin_errors_1d(self.coarse_theta_range, self.posterior_cdf_bins, thetas[:time_id]) )
        time_ids.append(time_id)
        nbr_sims.append(nsims[:time_id].sum()+at_burnin)
        
    errs = np.array(errs)
    time_ids = np.array(time_ids)
    nbr_sims = np.array(nbr_sims)
    
    f2 = pp.figure()
    sp1 = f2.add_subplot(2,2,1)
    pp.loglog( time_ids, errs, "bo-", lw=2)
    pp.xlabel( "nbr samples")
    pp.ylabel( "err")
    pp.grid('on')
    sp3 = f2.add_subplot(2,2,3)
    pp.semilogx( time_ids, errs, "bo-", lw=2)
    pp.xlabel( "nbr samples")
    pp.ylabel( "err")
    pp.grid('on')
    sp2 = f2.add_subplot(2,2,2)
    pp.loglog( nbr_sims, errs, "bo-", lw=2)
    pp.xlabel( "nbr samples")
    pp.ylabel( "err")
    pp.grid('on')
    sp4 = f2.add_subplot(2,2,4)
    pp.semilogx( nbr_sims, errs, "bo-", lw=2)
    pp.xlabel( "nbr samples")
    pp.ylabel( "err")
    pp.grid('on')
    #pdb.set_trace()
    print "ERROR  ",bin_errors_1d( self.coarse_theta_range, self.posterior_cdf_bins, thetas )
    # return handle to figure for further manipulation
    return f