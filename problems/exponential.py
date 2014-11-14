from abcpy.problem import BaseProblem
from abcpy.observation_group import ObservationGroup

from abcpy.plotting import *
from abcpy.helpers import *
import numpy as np
import scipy as sp
import pylab as pp

import pdb

def lognormal_logpdf( X, mu, sigma ):
  logpdf = -np.log(X) - 0.5*np.log(2*np.pi) - np.log(sigma) - 0.5*pow( (np.log(X)-mu)/sigma, 2 )
  return np.sum( logpdf )
  
def default_params():
  params = {}
  params["alpha"]           = 0.1
  params["beta"]            = 0.1
  params["theta_star"]      = 0.1
  params["N"]               = 500  # how many observations we draw per simulation
  params["q_stddev"]        = 0.5
  params["epsilon"]         = 0.1
  params["use_model"]       = False
  return params

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
    #self.proposal_rand   = positive_normal_rand
    #self.proposal_logpdf = normal_logpdf
    
    # "true" parameter setting
    self.theta_star = params["theta_star"]
    
    # number of samples per simulation
    self.N         = params["N"]
    
    self.epsilon   = params["epsilon"]
    self.use_model = params["use_model"]
    
  # "create" problem or load observations  
  def initialize( self ):
    assert self.initialized is False, "Ensure we only call this once..."
    
    # generate observations and statistics
    np.random.seed(0)
    self.observations   = self.simulation_function( self.theta_star )
    # reproduce by setting seed(0) : 10.086749290513298
    self.obs_statistics = np.array([10.086749290513298]) #self.statistics_function( self.observations )
    self.obs_sum        = 10.086749290513298*self.N
    
    self.min_range           = 0.05
    self.max_range           = 0.15
    self.range               = (self.min_range,self.max_range)
    self.fine_bin_width      = 0.0001
    self.coarse_bin_width    = 0.001
    
    self.fine_theta_range    = np.arange( self.min_range, self.max_range+self.fine_bin_width, self.fine_bin_width )
    self.coarse_theta_range  = np.arange( self.min_range, self.max_range+self.coarse_bin_width, self.coarse_bin_width )
    
    self.nbins_coarse   = len(self.coarse_theta_range)
    self.nbins_fine     = len(self.fine_theta_range)
    self.log_posterior  = gamma_logprob( self.fine_theta_range, self.alpha+self.N, self.beta+self.obs_sum )
    self.posterior      = np.exp(self.log_posterior)
    self.posterior_mode = (self.N + self.alpha)/(self.obs_sum + self.beta)
    
    self.true_posterior_logpdf_func = gen_gamma_logpdf(self.alpha+self.N,self.beta+self.obs_sum)
    self.true_posterior_cdf_func    = gen_gamma_cdf(self.alpha+self.N,self.beta+self.obs_sum)
    
    self.posterior_bars_range = self.coarse_theta_range[:-1] + 0.5*self.coarse_bin_width
    self.posterior_cdf        = self.true_posterior_cdf_func( self.coarse_theta_range )
    self.posterior_bars       = (self.posterior_cdf[1:] - self.posterior_cdf[:-1])/self.coarse_bin_width
    self.posterior_cdf_bins   = self.posterior_cdf[1:] - self.posterior_cdf[:-1]
    # done initialization
    self.initialized = True
    
  def get_observations( self ):
    assert self.initialized, "Not initialized..."
    return self.observations
    
  def get_obs_statistics( self ):
    assert self.initialized, "Not initialized..."
    return self.obs_statistics
    
  def get_obs_groups( self ):
    assert self.initialized, "Not initialized..."

    params = {"response_type":"gaussian",
              "response_params":{"epsilon":self.epsilon }
             }
    g = ObservationGroup( np.array([0]), self.get_obs_statistics(), params )
    return [g]   
    
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
    #pdb.set_trace()
    log_q_theta = np.log(theta) + np.random.randn( len(theta) )*self.proposal_std
    return np.exp(log_q_theta)
    #return self.proposal_rand( theta, self.proposal_std )
    
  def theta_proposal_logpdf( self, to_theta, from_theta ):
    return lognormal_logpdf( to_theta, np.log( from_theta), self.proposal_std )
    #return self.proposal_logpdf( to_theta, from_theta, self.proposal_std )
    #return self.proposal_logpdf( to_theta, np.log(from_theta), self.proposal_std )
  
  def compute_errors_at_times( self, times, thetas, sims ):
    errs = []
    time_ids = []
    nbr_sims = []
    
    for time_id in times:
      if time_id <= len(thetas):
        errs.append( bin_errors_1d(self.coarse_theta_range, self.posterior_cdf_bins, thetas[:time_id]) )
        time_ids.append(time_id)
        nbr_sims.append(sims[:time_id].sum())
        
    errs = np.array(errs)
    time_ids = np.array(time_ids)
    nbr_sims = np.array(nbr_sims)
    
    return errs, nbr_sims, time_ids 
    
  # take samples/staistics etc and "view" this particular problem
  def view_results( self, states_object, burnin = 1 ):
    # plotting params
    nbins       = 20
    alpha       = 0.5
    label_size  = 8
    linewidth   = 3
    linecolor   = "r"
    
    # extract from states
    thetas = states_object.get_thetas()[burnin:,:]
    stats  = states_object.get_statistics()[burnin:,:]
    nsims  = states_object.get_sim_calls()[burnin:]
    
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
    
    total_sims = states_object.get_sim_calls().sum()
    all_sims = nsims.sum()
    at_burnin = total_sims-all_sims
    errs = []
    time_ids = []
    nbr_sims = []
    
    for time_id in [1,5,10,25,50,75,100,200,300,400,500,750,1000,1500,2000,3000,4000,5000,7500,10000,12500,15000,17500,20000,25000,30000,35000,40000,45000,50000]:
      if time_id <= len(thetas):
        errs.append( bin_errors_1d(self.coarse_theta_range, self.posterior_cdf_bins, thetas[:time_id]) )
        time_ids.append(time_id)
        nbr_sims.append(nsims[:time_id].sum()+at_burnin)
        
    errs = np.array(errs)
    time_ids = np.array(time_ids)
    nbr_sims = np.array(nbr_sims)
    
    f2 = pp.figure()
    sp1 = f2.add_subplot(1,3,1)
    pp.semilogx( time_ids, errs, "bo-", lw=2)
    pp.xlabel( "nbr samples")
    pp.ylabel( "err")
    pp.grid('on')
    sp2 = f2.add_subplot(1,3,2)
    pp.semilogx( nbr_sims, errs, "ro-", lw=2)
    pp.xlabel( "nbr sims")
    pp.ylabel( "err")
    pp.grid('on')
    sp3 = f2.add_subplot(1,3,3)
    pp.semilogx( time_ids, errs, "bo-", lw=2)
    pp.semilogx( nbr_sims, errs, "ro-", lw=2)
    pp.xlabel( "time")
    pp.ylabel( "err")
    pp.grid('on')
    pp.show()
    #pdb.set_trace()
    print "ERROR    ",bin_errors_1d( self.coarse_theta_range, self.posterior_cdf_bins, thetas )
    #print "ACC RATE ", states_object.acceptance_rate()
    print "SIM      ", total_sims
    # return handle to figure for further manipulation
    return f
    
if __name__ == "__main__":
  # view problem and rejection samples
  pp.rc('text', usetex=True)
  pp.rc('font', family='serif')
  #plt.xlabel(r'\textbf{time} (s)')
  
  epsilon = 2.0
  
  params = default_params()
  p = ExponentialProblem(params)
  p.initialize()
  thetas = np.load("./uai2014/saved/exponential/rejection_eps2p0_thetas.npy")[:,1]
  pp.close("all")
  # f1 = pp.figure(1)
  # sp1 = f1.add_subplot(111)
  # pp.plot( p.fine_theta_range, p.posterior, "k--", lw = 2)
  # 
  # pp.hist( thetas, p.nbins_coarse, range=p.range,normed = True, alpha = 0.25 )
  # ax = pp.axis()
  # pp.axis([p.range[0],p.range[1],ax[2],ax[3]])
  
  N = 2000
  good = []
  bad = []
  good_stats = []
  bad_stats = []
  for n in range(N):
    theta = p.theta_prior_rand()
    sim_outs = p.simulation_function(theta)
    stats = p.statistics_function(sim_outs)
    if np.abs( p.obs_statistics - stats )<= epsilon:
      good.append( theta )
      good_stats.append(stats)
    else:
      bad.append(theta)
      bad_stats.append(stats)
      
  good = np.squeeze(np.array(good))
  bad = np.squeeze(np.array(bad))
  good_stats = np.squeeze(np.array(good_stats))
  bad_stats = np.squeeze(np.array(bad_stats))
  
  figsize=(9,6)
  #dpi=600
  # f2 = pp.figure(2,figsize=figsize, dpi=dpi)
  # #f2=pp.figure(figsize=(3,3),dpi=300)
  # sp = f2.add_subplot(111)
  # 
  # pp.plot( bad, bad_stats, 'b.', ms=5,alpha=0.25)
  # pp.plot( good, good_stats, 'ro', ms=5, alpha = 0.5)
  # pp.vlines( p.posterior_mode, 0, 20 )
  # pp.hlines( p.obs_statistics, 0, 1.0 )
  # pp.hlines( p.obs_statistics+epsilon, 0, 1.0, linestyles="--", lw=2 )
  # pp.hlines( p.obs_statistics-epsilon, 0, 1.0, linestyles="--", lw=2 )
  # pp.axis( [0,1.0,0,20])
  # 
  # 
  # set_tick_fonsize( sp, 6 )
  # set_label_fonsize( sp, 8 )
  # 
  from mpl_toolkits.axes_grid1 import host_subplot
  import mpl_toolkits.axisartist as AA
  import matplotlib.pyplot as plt
  
  f=pp.figure(figsize=figsize)

  fs=16
  host = host_subplot(111, axes_class=AA.Axes)
  plt.subplots_adjust(right=0.75)

  
  #host.plot( bad, bad_stats, 'r.', ms=5,alpha=0.25)
  host.plot( bad, bad_stats, 'bo', ms=7,alpha=0.25)
  host.plot( good, good_stats, 'ro', ms=15, alpha = 0.5)
  #host.vlines( p.posterior_mode, 0, 20 )
  host.hlines( p.obs_statistics, 0, 1.0 )
  host.hlines( p.obs_statistics+epsilon, 0, 1.0, linestyles="--", lw=4 )
  host.hlines( p.obs_statistics-epsilon, 0, 1.0, linestyles="--", lw=4 )
  #par2 = host.twinx()
  par1 = host.twinx()
  xx=np.linspace( 0.001, 0.2, 100 )
  
  #par1.plot( xx, np.exp( p.theta_prior_logpdf( xx) ), "b--", lw=1 )
  p2,=par1.plot( p.fine_theta_range, p.posterior, "b-", lw = 3)
  par1.hist( thetas, p.nbins_coarse, color="r",histtype="stepfilled",range=p.range,normed = True, alpha = 0.5 )
  
  par1.text(0.105,70,r'$\pi( \theta | y )$', fontsize=1.5*fs)
  par1.text(0.11,25,r'$\pi_{\epsilon}( \theta | y  )$', fontsize=1.5*fs)
  #pp.axis( [0,0.4,0,20])
  #par1.axis["right"].label.set_color("b")
  #par1.axis["right"].label.set_color(p2.get_color())
  host.set_xlim(0, 0.2)
  par1.set_xlim(0, 0.2)
  host.set_ylim(0, 20)
  par1.set_ylim(0, 250)
  host.axis["left"].set_label("x")
  host.axis["left"].label.set_rotation(90)
  
  #plt.xlabel(r'\textbf{time} (s)')
  #par1.axis["right"].set_label("p(theta|y)")
  par1.axis["right"].set_label(r'\textbf{$\pi( \theta | y )$}')
  host.axis["bottom"].set_label(r'\textbf{$\theta$}')
  host.axis["left"].label.set_fontsize(fs)
  host.axis["bottom"].label.set_fontsize(fs)
  host.axis["left"].major_ticklabels.set_fontsize(fs)
  host.axis["bottom"].major_ticklabels.set_fontsize(fs)
  par1.axis["right"].major_ticklabels.set_fontsize(fs)
  par1.axis["right"].label.set_fontsize(fs)
  par1.axis["right"].label.set_rotation(270)
  par1.axis["right"].label.set_color("b")
  
  host.set_title("Exponential problem", fontsize=20)
  f.savefig( "exponential_problem.eps", format="ps", dpi=600 ) #,bbox_inches="tight")
  #savefig( "test.png", format="png", dpi=300,bbox_inches="tight")
  
  # for tick in host.yaxis.get_major_ticks():
  #   tick.label.set_fontsize(6)
  # for tick in par1.yaxis.get_major_ticks():
  #   tick.label.set_fontsize(6)
  # 
  # set_tick_fonsize( host, 6 )
  # set_label_fonsize( host, 8 )
  # set_tick_fonsize( par1, 6 )
  # set_label_fonsize( par1, 8 )
  #     
  #fig, ax1 = pp.subplots(111)
  #ax2 = ax1.twinx()
  
  #sp2 = f2.add_subplot(111)
  # ax2.plot( p.fine_theta_range, p.posterior, "b-", lw = 2)
#   ax2.hist( thetas, p.nbins_coarse, range=p.range,normed = True, alpha = 0.25 )
#   ax = ax2.axis()
#   ax2.axis([p.range[0],p.range[1],0,300])
  pp.show()
  