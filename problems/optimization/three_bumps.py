from abcpy.problem import BaseProblem
from abcpy.plotting import *
from abcpy.helpers import *
from abcpy.observation_group import ObservationGroup
import numpy as np
import scipy as sp
import pylab as pp
log2pi = np.log(2*np.pi)
import pdb
logsum=logsumexp
def loglike_mog( X, mu, std ):
  N = len(X)

  lpdf = -0.5*log2pi - np.log(std)

  d = X-mu
  lpdf -= 0.5*d*d/std
  lpdf = lpdf.reshape( (N,))
  return lpdf
    
def mog_logliks(X, PI, M, STD ):
  N = len(X)
  K = len(PI)
  R = np.zeros( ( N, K ) )
  logliks = np.zeros( ( N, K ) )
  for k in range( K ):
    logliks[:,k] = loglike_mog( X, M[k], STD[k] )

  ls = logsum( np.log(PI) + logliks,1 ).reshape( (len(X),1))
  return ls
  
def mog_loglikelihood( x, pis, mus, stds ):
  return mog_logliks( x, pis, mus, stds )
  
def default_params():
  params = {}
  params["ystar"]           = 0.0
  
  # where along theta-axis are the bumps centered
  params["bump_centers"]    = np.array( [-1.5,0,1.5])
  # what are the standard deviations of the bumps
  params["bump_stds"]       = 0.25*np.array([1.0,1.0,1.0])
  # how are they weighted?
  params["bump_weights"]    = np.array([0.333,0.333,0.333])
  
  # gaussian noise stddev added at each theta location
  params["noise"]           = 0.0
  
  # prior over theta
  params["prior_mu"]        = 0.0
  params["prior_std"]       = 3.0
  
  # random walk steps
  params["q_stddev"]        = 0.5
  
  return params

class ThreeBumpsProblem( BaseProblem ):
  # extract info about specific for this problem
  def load_params( self, params ):
    self.obs_statistics = params["ystar"]
    self.ystar          = params["ystar"]
    
    self.epsilon = params["epsilon"]
    # prior parameters 
    self.prior_mu   = params["prior_mu"]
    self.prior_std  = params["prior_std"]
    
    # where along theta-axis are the bumps centered
    self.bump_centers = params["bump_centers"] 
    # what are the standard deviations of the bumps
    self.bump_stds    = params["bump_stds"]
    # how are they weighted?
    self.bump_weights = params["bump_weights"]
    
    # gaussian noise stddev added at each theta location
    self.noise = params["noise"]
    
    # proposal params (Gaussian)
    if params.has_key("q_stddev"):
      self.proposal_std    = params["q_stddev"]
    #self.proposal_rand   = normal_rand
    #self.proposal_logpdf = normal_logpdf
    
  # "create" problem or load observations  
  def initialize( self ):
    assert self.initialized is False, "Ensure we only call this once..."
    
    self.min_range           = -3.0
    self.max_range           =  3.0
    self.range               = (self.min_range,self.max_range)
    self.fine_bin_width      = 0.005
    self.coarse_bin_width    = 0.05
    
    self.fine_theta_range    = np.arange( self.min_range, self.max_range+self.fine_bin_width, self.fine_bin_width )
    self.coarse_theta_range  = np.arange( self.min_range, self.max_range+self.coarse_bin_width, self.coarse_bin_width )
    
    self.nbins_coarse   = len(self.coarse_theta_range)
    self.nbins_fine     = len(self.fine_theta_range)
    
    self.obs_statistics = self.simulation_mean_function( np.array([0.0]) )[0]
    self.ystar          = self.obs_statistics
    
    #self.log_posterior  = gamma_logprob( self.fine_theta_range, self.alpha+self.N, self.beta+self.obs_sum )
    #self.posterior      = np.exp(self.log_posterior)
    #self.posterior_mode = (self.N + self.alpha)/(self.obs_sum + self.beta)
    
    #self.true_posterior_logpdf_func = gen_gamma_logpdf(self.alpha+self.N,self.beta+self.obs_sum)
    #self.true_posterior_cdf_func    = gen_gamma_cdf(self.alpha+self.N,self.beta+self.obs_sum)
    
    #self.posterior_bars_range = self.coarse_theta_range[:-1] + 0.5*self.coarse_bin_width
    #self.posterior_cdf        = self.true_posterior_cdf_func( self.coarse_theta_range )
    #self.posterior_bars       = (self.posterior_cdf[1:] - self.posterior_cdf[:-1])/self.coarse_bin_width
    #self.posterior_cdf_bins   = self.posterior_cdf[1:] - self.posterior_cdf[:-1]
    # done initialization
    self.initialized = True
    
  def get_observations( self ):
    assert self.initialized, "Not initialized..."
    return self.get_obs_statistics()
    
  def get_obs_statistics( self ):
    assert self.initialized, "Not initialized..."
    return self.obs_statistics
  
  def get_obs_groups( self ):
    assert self.initialized, "Not initialized..."

    params = {"response_type":"gaussian",
              "response_params":{"epsilon":self.epsilon }
              }
   #   g = ObservationGroup( np.array([0]), self.get_obs_statistics(), params )
   #    return [g]
      
    g = ObservationGroup( np.array([0]), self.get_obs_statistics(), params)
    return [g]     
    
  def simulation_mean_function( self, theta ):
    return -mog_loglikelihood( theta, self.bump_weights, self.bump_centers, self.bump_stds )
    
  # run simulation at parameter setting theta, return outputs
  def simulation_function( self, theta ):
    #pdb.set_trace()
    return self.simulation_mean_function(theta) + self.noise*np.random.randn()
    
  # pass outputs through statistics function, return statistics
  def statistics_function( self, outputs ):
    return outputs
    
  # return size of statistics vector for this problem
  def get_nbr_statistics( self ):
    return 1
  
  # theta_rand
  def theta_prior_rand( self, N = 1 ):
    return self.prior_mu+self.prior_std*np.random.randn( N ) 
    
  # theta_rand
  def theta_prior_logpdf( self, theta ):
    return gaussian_logpdf( theta, self.prior_mu, self.prior_std) #gamma_logprob( theta, self.alpha, self.beta ) # 1/beta cause of how python implements
      
  def theta_proposal_rand( self, theta ):
    return theta + self.proposal_std*np.random.randn()
    
  def theta_proposal_logpdf( self, to_theta, from_theta ):
    return 0 #self.proposal_logpdf( to_theta, from_theta, self.proposal_std )
    #return self.proposal_logpdf( to_theta, np.log(from_theta), self.proposal_std )
  
  def log_posterior( self, theta, epsilon=None ):
    logprior = np.squeeze( self.theta_prior_logpdf( theta) )
    
    if epsilon is None:
      if self.noise > 0:
        likelihood = normcdf( self.ystar, self.simulation_mean_function( theta ), self.noise )
      else:
        likelihood = np.zeros( theta.shape )
        I = pp.find( self.ystar >= self.simulation_mean_function( theta ) )
        likelihood[I] = 1
    else:
      if self.noise > 0:
        likelihood = normcdf( self.ystar+epsilon, self.simulation_mean_function( theta ), self.noise )
      else:
        likelihood = np.zeros( theta.shape )
        I = pp.find( self.ystar+epsilon >= self.simulation_mean_function( theta ) )
        likelihood[I] = 1
    
    return logprior + np.squeeze(np.log( likelihood ))
  
  def posterior( self, theta, epsilon=None ):
    return np.exp( self.log_posterior(theta,epsilon))
      
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
  def view_results( self, states_object, burnin = 1, epsilon = None ):
    # plotting params
    nbins       = 100
    alpha       = 0.5
    label_size  = 8
    linewidth   = 3
    linecolor   = "b"
    
    # extract from states
    thetas = states_object.get_thetas()
    stats  = states_object.get_statistics()
    nsims  = states_object.get_sim_calls()
    
    # plot sample distribution of thetas, add vertical line for true theta, theta_star
    f = pp.figure()
    sp = f.add_subplot(111)
    #pp.plot( self.fine_theta_range, self.posterior, linecolor+"-", lw = 1)
    ax = pp.axis()
    pp.hist( thetas, 100, histtype='stepfilled',range=self.range, color="r",normed = True, alpha = 0.5 )
    pp.hist( thetas, 100, histtype='step',range=self.range, color="k",normed = True, lw=4 )
    posterior = self.posterior(self.fine_theta_range)
    Z = np.sum(0.5*(posterior[1:]+posterior[:-1])*self.fine_bin_width)
    pp.plot( self.fine_theta_range, self.posterior(self.fine_theta_range)/Z, lw=3, color="b")
    if epsilon is not None:
      posterior = self.posterior(self.fine_theta_range,epsilon)
      Z = np.sum(0.5*(posterior[1:]+posterior[:-1])*self.fine_bin_width)
      #pp.plot( self.fine_theta_range, self.posterior(self.fine_theta_range,epsilon)/Z, lw=3, color="m")
    #host = host_subplot(111, axes_class=AA.Axes)
    #pp.subplots_adjust(right=0.75)
    if epsilon is not None:
      pp.hlines( self.obs_statistics+epsilon, self.fine_theta_range[0], self.fine_theta_range[-1], linestyles="--", lw=2 )
    pp.hlines( self.obs_statistics, self.fine_theta_range[0], self.fine_theta_range[-1], linestyles="-", lw=4 )

    n_samples = 1500
    xx = -3 + 6*np.random.randn(n_samples)
    yy = self.simulation_mean_function(xx)+self.noise*np.random.randn(n_samples,1)
    #pdb.set_trace()
    #pp.plot( self.fine_theta_range, self.simulation_mean_function(self.fine_theta_range) )
    I = pp.find( yy < self.ystar[0] + self.epsilon )
    J = pp.find( yy >= self.ystar[0] + self.epsilon )
    pp.plot( xx[J], yy[J], 'bo', ms=8, alpha=0.75)
    pp.plot( xx[I], yy[I], 'ro', ms=20, alpha=0.95)
    pp.ylim( 0, 3 )
    pp.xlim( -3,3)
    
    pp.title("Three Bumps")
    for tick in sp.yaxis.get_major_ticks():
      tick.label.set_fontsize(24)

    for tick in sp.xaxis.get_major_ticks():
      tick.label.set_fontsize(24)
    sp.title.set_fontsize(20)  
    #sp = f.add_subplot(212)
    #Z=self.posterior(self.fine_theta_range).sum()*self.fine_bin_width
    #pp.plot( self.fine_theta_range, self.posterior(self.fine_theta_range)/Z)
    #pp.plot( self.fine_theta_range, self.posterior, linecolor+"-", lw = 1)
    #ax = pp.axis()
    # pp.fill_between( self.fine_theta_range, self.posterior, color="m", alpha=0.5)
    # 
    # pp.plot( self.posterior_bars_range, self.posterior_bars, 'ro')
    # pp.vlines( thetas.mean(), ax[2], ax[3], color="b", linewidths=linewidth)
    # #pp.vlines( self.theta_star, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    # pp.vlines( self.posterior_mode, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    # 
    # pp.xlabel( "theta" )
    # pp.ylabel( "P(theta)" )
    # pp.axis([self.range[0],self.range[1],ax[2],ax[3]])
    # set_label_fonsize( sp, label_size )
    # 
    # total_sims = states_object.nbr_sim_calls
    # all_sims = nsims.sum()
    # at_burnin = total_sims-all_sims
    # errs = []
    # time_ids = []
    # nbr_sims = []
    # 
    # for time_id in [1,5,10,25,50,75,100,200,300,400,500,750,1000,1500,2000,3000,4000,5000,7500,10000,12500,15000,17500,20000,25000,30000,35000,40000,45000,50000]:
    #   if time_id <= len(thetas):
    #     errs.append( bin_errors_1d(self.coarse_theta_range, self.posterior_cdf_bins, thetas[:time_id]) )
    #     time_ids.append(time_id)
    #     nbr_sims.append(nsims[:time_id].sum()+at_burnin)
    #     
    # errs = np.array(errs)
    # time_ids = np.array(time_ids)
    # nbr_sims = np.array(nbr_sims)
    # 
    # f2 = pp.figure()
    # sp1 = f2.add_subplot(1,3,1)
    # pp.semilogx( time_ids, errs, "bo-", lw=2)
    # pp.xlabel( "nbr samples")
    # pp.ylabel( "err")
    # pp.grid('on')
    # sp2 = f2.add_subplot(1,3,2)
    # pp.semilogx( nbr_sims, errs, "ro-", lw=2)
    # pp.xlabel( "nbr sims")
    # pp.ylabel( "err")
    # pp.grid('on')
    # sp3 = f2.add_subplot(1,3,3)
    # pp.semilogx( time_ids, errs, "bo-", lw=2)
    # pp.semilogx( nbr_sims, errs, "ro-", lw=2)
    # pp.xlabel( "time")
    # pp.ylabel( "err")
    # pp.grid('on')
    # 
    # #pdb.set_trace()
    # print "ERROR    ",bin_errors_1d( self.coarse_theta_range, self.posterior_cdf_bins, thetas )
    # print "ACC RATE ", states_object.acceptance_rate()
    # print "SIM      ", total_sims
    # return handle to figure for further manipulation
    return f
    
if __name__ == "__main__":
  # view problem and rejection samples
  pp.rc('text', usetex=True)
  pp.rc('font', family='serif')
  #plt.xlabel(r'\textbf{time} (s)')
  
  epsilon = 0.1
  
  params = default_params()
  p = ThreeBumpsProblem(params)
  p.initialize()
  #thetas = np.load("./uai2014/saved/exponential/rejection_eps2p0_thetas.npy")[:,1]
  pp.close("all")
  # f1 = pp.figure(1)
  # sp1 = f1.add_subplot(111)
  # pp.plot( p.fine_theta_range, p.posterior, "k--", lw = 2)
  # 
  # pp.hist( thetas, p.nbins_coarse, range=p.range,normed = True, alpha = 0.25 )
  # ax = pp.axis()
  # pp.axis([p.range[0],p.range[1],ax[2],ax[3]])
  
  
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
  host.hlines( p.obs_statistics+epsilon, p.fine_theta_range[0], p.fine_theta_range[-1], linestyles="--", lw=2 )
  host.hlines( p.obs_statistics, p.fine_theta_range[0], p.fine_theta_range[-1], linestyles="-", lw=2 )
  host.plot( p.fine_theta_range, p.simulation_mean_function(p.fine_theta_range) )
  
  #host.plot( bad, bad_stats, 'r.', ms=5,alpha=0.25)
  #host.plot( bad, bad_stats, 'bo', ms=7,alpha=0.25)
  
  #host.plot( good, good_stats, 'ro', ms=15, alpha = 0.5)
  #host.vlines( p.posterior_mode, 0, 20 )
  #host.hlines( p.obs_statistics, 0, 1.0 )
  
  #host.hlines( p.obs_statistics-epsilon, 0, 1.0, linestyles="--", lw=4 )
  #par2 = host.twinx()
  #par1 = host.twinx()
  #xx=np.linspace( 0.001, 0.2, 100 )
  
  # p2,=par1.plot( p.fine_theta_range, p.posterior, "b-", lw = 3)
  # par1.hist( thetas, p.nbins_coarse, color="r",histtype="stepfilled",range=p.range,normed = True, alpha = 0.5 )
  # 
  # par1.text(0.105,70,r'$\pi( \theta | y )$', fontsize=1.5*fs)
  # par1.text(0.11,25,r'$\pi_{\epsilon}( \theta | y  )$', fontsize=1.5*fs)
  # host.set_xlim(0, 0.2)
  # par1.set_xlim(0, 0.2)
  # host.set_ylim(0, 20)
  # par1.set_ylim(0, 250)
  # host.axis["left"].set_label("x")
  # host.axis["left"].label.set_rotation(90)
  # 
  # par1.axis["right"].set_label(r'\textbf{$\pi( \theta | y )$}')
  # host.axis["bottom"].set_label(r'\textbf{$\theta$}')
  # host.axis["left"].label.set_fontsize(fs)
  # host.axis["bottom"].label.set_fontsize(fs)
  # host.axis["left"].major_ticklabels.set_fontsize(fs)
  # host.axis["bottom"].major_ticklabels.set_fontsize(fs)
  # par1.axis["right"].major_ticklabels.set_fontsize(fs)
  # par1.axis["right"].label.set_fontsize(fs)
  # par1.axis["right"].label.set_rotation(270)
  # par1.axis["right"].label.set_color("b")
  # 
  # host.set_title("Exponential problem", fontsize=20)
  # f.savefig( "exponential_problem.eps", format="ps", dpi=600 ) #,bbox_inches="tight")

  pp.show()
  